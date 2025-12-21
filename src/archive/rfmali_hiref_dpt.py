import sys
import numpy as np
import ot  # Python Optimal Transport
from rfgap import RFGAP # Your custom module
from rfphate import PageRankPHATE
from scipy import sparse
from scipy.spatial.distance import cdist
from .hiref import HiRef_fast as HiRef
from .hiref import rank_annealing
import scipy
from sklearn import preprocessing
import graphtools

# Embedders
from phate import PHATE # Your custom module



# Fast and Sparse Semi-Supervised Manifold Alignment with Random Forest Semantic Optimal Transport
class RFMALI(object):
    def __init__(self, mu=0.5, n_components=2, knn_dist='precomputed_affinity', verbose=0,
                 random_state=None, n_jobs=-1):
        self.mu = mu
        self.n_components = n_components
        self.knn_dist = knn_dist
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        self.T_sparse = None
        self.W_combined = None
        self.embedding_ = None
    


    def _get_rfgap_posteriors(self, prox, y, labels):
        """
        Compute Manifold-Smoothed Posteriors using RF proximities.
        """
        
        n_samples = prox.shape[0]
        n_classes = len(labels)
        label_to_col = {lab: i for i, lab in enumerate(labels)}
        
        posteriors = np.zeros((n_samples, n_classes), dtype=float)
        
        # Class-wise summation (KDE on the Manifold)
        for lab in labels:
            col_idx = label_to_col[lab]
            class_mask = (y == lab)
            
            if not np.any(class_mask):
                continue
                
            # Efficient sparse slicing
            if sparse.issparse(prox):
                # Summing over the columns belonging to class 'lab'
                vec = prox[:, class_mask].sum(axis=1).A.ravel()
            else:
                vec = prox[:, class_mask].sum(axis=1)
                
            posteriors[:, col_idx] = vec
            
        # # Row Normalize -> Probability Distribution
        # row_sums = posteriors.sum(axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1.0
        # posteriors /= row_sums
        
        return posteriors
    

    def _get_diffusion_operators(self,
        K,
        n_landmark=2000,
        random_state=None,
        verbose=True,
        **graph_kwargs,
    ):
        """
        Build a graphtools Graph and return a unified (P_NM, P_MM) interface.

        - If the graph is a LandmarkGraph:
            P_MM = landmark diffusion operator  (M x M)
            P_NM = transitions from N points to M landmarks (N x M)
        - If the graph is a TraditionalGraph:
            P_MM = full diffusion operator P (N x N)
            P_NM = identity (N x N)  -> treat all points as landmarks.

        Parameters
        ----------
        K : array-like (N, N)
            Precomputed symmetric affinity / kernel matrix (usually sparse)
        n_landmark : int or None
            Number of landmarks to request. If None, graphtools may build a
            TraditionalGraph instead.
        random_state : int or None
        verbose : bool
        graph_kwargs : dict
            Extra kwargs passed to graphtools.Graph (e.g., knn, decay, etc.)

        Returns
        -------
        G : graphtools.Graph
        P_NM : ndarray (N x M) -- transitions from N points to M landmarks
        P_MM : ndarray (M x M) -- landmark diffusion operator
        """

        G = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark if n_landmark is not None and n_landmark < K.shape[0] else None,
            kernel_symm=None,
            random_state=random_state,
            verbose=verbose,
            **graph_kwargs,
        )

        # LandmarkGraph case
        if hasattr(G, "landmark_op") and hasattr(G, "transitions"):
            if verbose:
                print("Using LandmarkGraph operators (N→M and M→M).")
        
            P_MM = np.asarray(G.landmark_op)           # (M_eff, M_eff) usually
            P_NM = G.transitions.toarray()             # (N, M_cols)
            clusters = np.asarray(G.clusters).ravel()  # (N,)
            
            # ------------------------------------------------------------
            # ✅ Minimal robust fix for graphtools indexing mismatch:
            # clusters may contain ids not matching P_NM column indexing.
            # We only need contiguous ids in [0..M-1] where M = P_NM.shape[1].
            # ------------------------------------------------------------
            M = P_NM.shape[1]
            
            # remap cluster ids to consecutive integers (0..K-1)
            _, clusters_remap = np.unique(clusters, return_inverse=True)
            clusters = clusters_remap.astype(int)
            
            # if remapped clusters still has more unique ids than columns, clip/merge
            # (this happens when graphtools kept an extra cluster label but dropped its column)
            K = clusters.max() + 1
            if K > M:
                # merge the last cluster(s) into the last valid landmark index
                clusters = np.minimum(clusters, M - 1)
            
            # final sanity: clusters now in [0..M-1]
            if clusters.max() >= M or clusters.min() < 0:
                raise ValueError(f"After remap, clusters out of bounds: [{clusters.min()}, {clusters.max()}] vs M={M}")
            
            # also ensure P_MM matches M (sometimes landmark_op matches transitions cols)
            if P_MM.shape[0] != M:
                # safest: truncate to MxM (since transitions defines the actual landmark basis)
                P_MM = P_MM[:M, :M]
            
            return P_NM, P_MM, clusters

        else:
            # TraditionalGraph fallback: all points are landmarks if n_landmark >= N
            if verbose:
                print("Using TraditionalGraph: treating all N points as landmarks.")
            P_MM = G.P.toarray()               # N x N diffusion operator
            P_NM = np.eye(P_MM.shape[0], dtype=P_MM.dtype)  # identify, all points = landmarks
            clusters = np.arange(P_MM.shape[0])

        return P_NM, P_MM, clusters
    

    # MALI works on Full Diff Operators (I checked this), we suggest using Landmark Diff Op instead for scalability (O(M^3) but M<<N)    
    def compute_dpt(self, P):  # should we row-normalize M?
        n = P.shape[0]
        I = np.eye(n)
    
        # psi0 = 1 for row-stochastic P
        ones = np.ones(n, dtype=float)
    
        # stationary distribution phi0: left eigenvector for eigenvalue 1
        wL, lv = np.linalg.eig(P.T)
        j = np.argmin(np.abs(wL - 1.0))
        phi0 = lv[:, j].real
    
        # fix sign + normalize to sum to 1  (so phi0^T * 1 = 1)
        if phi0.sum() < 0:
            phi0 = -phi0
        phi0 = np.maximum(phi0, 0)
        s = phi0.sum()
        if s <= 0:
            raise ValueError("Failed to extract a valid stationary distribution.")
        phi0 = phi0 / s
    
        # deflate stationary component: P - 1 phi0^T
        P_deflated = P - np.outer(ones, phi0)
    
        # M = (I - P_deflated)^(-1) - I  (use solve, not inv)
        Mmat = np.linalg.solve(I - P_deflated, I) - I

        # min max normalize
        from sklearn.preprocessing import MinMaxScaler
        Mmat = MinMaxScaler().fit_transform(Mmat.T).T
    
        return Mmat
    

    def _get_balanced_posteriors(self,
        prox_nm,
        y,
        clusters,
        eps=1e-12,
    ):
        """
        prox_nm  : (N, M) point->landmark weights (P_NM @ M_MM)
        y        : (N,) labels
        clusters : (N,) landmark id per point
        """   
        N, M = prox_nm.shape
        labels = np.unique(y)
        C = labels.size
    
        # --- label indexing ---
        lab2idx = {lab: k for k, lab in enumerate(labels)}
        y_idx = np.fromiter((lab2idx[yy] for yy in y), count=N, dtype=int)
    
        # --- landmark class counts ---
        counts = np.zeros((M, C), dtype=float)
        np.add.at(counts, (clusters, y_idx), 1.0)
    
        # p(c | landmark m)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1.0)
        Q = counts / row_sums                      # (M, C)
    
        # --- global class prior p(c) ---
        class_prior = np.bincount(y_idx, minlength=C).astype(float)
        class_prior /= class_prior.sum()           # (C,)
    
        # --- point posteriors ---
        post = prox_nm @ Q                         # (N, C)
    
        # --- prior correction ---
        post /= np.maximum(class_prior[None, :], eps)
    
        return post
    

    def fit(self, x_a, y_a, x_b, y_b):
        """
        Fit the RFMALI model to the two datasets, assumed to be of the same size n.
        """
        self.n = x_a.shape[0]

        print("Fitting RFGAP on Domain A...")
        y_a = y_a.ravel() 
        self.rfgap_a = RFGAP(y=y_a, random_state=self.random_state,
                             non_zero_diagonal=True,
                             force_symmetric=True,  # Important for symmetry in joint affinity
                             n_jobs=self.n_jobs)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities()
        
        print("Fitting RFGAP on Domain B...")
        y_b = y_b.ravel()
        self.rfgap_b = RFGAP(y=y_b, random_state=self.random_state,
                             non_zero_diagonal=True,
                             force_symmetric=True,
                             n_jobs=self.n_jobs)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()

        P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(
            prox_a,
            n_landmark=2000,
            random_state=self.random_state,
            verbose=True)
        print(P_NM_a, P_MM_a, clusters_a)

        P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(
            prox_b,
            n_landmark=2000,
            random_state=self.random_state,
            verbose=True)
        
        # Build DPT Matrices (infinite aggregated transition matrices from Landmarks to Landmarks)
        M_a = self.compute_dpt(P_MM_a)
        M_b = self.compute_dpt(P_MM_b)
        print(M_a, M_b)

        # Build infinite aggregated transition matrices from Points to Landmarks by projection
        trans_a = P_NM_a.dot(M_a)  # N x M
        trans_b = P_NM_b.dot(M_b)  # N x M

        # Build C-dimensional Posteriors using the Landmark cluster assignments and the existing labels
        #NOTE: Assumption: y_a and y_b have the same label space, with at least one point per class in each domain
        # Build Balanced Posteriors using the Landmark cluster assignments and the existing labels
        print("Building Balanced Posteriors...")
        post_a = self._get_balanced_posteriors(trans_a, y_a, clusters_a)
        post_b = self._get_balanced_posteriors(trans_b, y_b, clusters_b)
        
        
        # Compute Optimal Transport with HiRef (fast near linear OT)
        print("Computing Optimal Transport...")
        rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n)
        frontier = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule)
        print(frontier)


        row_idx = np.array([int(p[0][0]) for p in frontier], dtype=np.int64)
        col_idx = np.array([int(p[1][0]) for p in frontier], dtype=np.int64)
        data = np.ones_like(row_idx, dtype=float)  # Values for the coupling: 1 for each matched pair
        self.T_sparse = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.n, self.n))


        # Fusion
        print("Building joint affinity matrix...")

        # Off-diagonal blocks
        # W_ab = 0.5 * (P_a @ T + T @ P_b)
        W_ab = (prox_a.dot(self.T_sparse) + self.T_sparse.dot(prox_b)) / 2
        W_ba = W_ab.T

        self.W_combined = sparse.bmat(
            [
                [self.mu * prox_a, (1-self.mu) * W_ab],
                [(1-self.mu) * W_ba, self.mu * prox_b]
            ],
            format="csr"
        )
        
        print("Model fit complete.")
        return self

    def fit_transform(self, x_a=None, y_a=None, x_b=None, y_b=None):
        if x_a is not None:
            self.fit(x_a, y_a, x_b, y_b)
            
        phate_op = PageRankPHATE(
            n_components=self.n_components,
            knn_dist=self.knn_dist,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=-1,
            beta=0.9,
        )
        self.embedding_ = phate_op.fit_transform(self.W_combined)
        return self.embedding_