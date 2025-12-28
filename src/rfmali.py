import numpy as np
from scipy import sparse
from sklearn import preprocessing

# Graph tools for DPT
import graphtools

# RF-GAP
from rfgap import RFGAP

# Embedders
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

# Clustering
from sklearn.cluster import MiniBatchKMeans

# Utils
from utils.utils import kernel2Dist
from utils.labels import LabelUtils

# OT solver
from .hiref import HiRef_fast as HiRef
from .hiref import rank_annealing

import sys

class RFMALI(object):
    '''RF-MALI: Random Forest-based MALI implementation for semi-supervised domain adaptation.'''
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 n_landmark=2000,
                 prior_correct=True,
                 semantic_norm='l1',  # normalization method for semantic vectors (supports 'l1' and 'l2')
                 embedder='spectral',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.n_landmark = n_landmark  # number of landmarks for DPT
        self.prior_correct = prior_correct
        self.semantic_norm = semantic_norm
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.rfgap_params = {
            'random_state': random_state,
            'prediction_type': 'classification',  # force classification mode
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'max_normalize': True,
            # 'class_weight': 'balanced',  # handle class imbalance in RF
            'verbose': 0,
            'n_jobs': n_jobs,
        }

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

    # ------------------------------------------------------------
    # Posterior builders
    # ------------------------------------------------------------        
    def _get_semantic_vectors(
        self,
        W,                     # (N, K) Adjacency. K=N (Full, sparse) or K=M (Landmarks, dense)
        y,                     # (N,) Labels
        labels,                # List of unique canonical labels
        clusters=None,         # (N,) Landmark IDs (optional, only for landmark path)
        eps=1e-12,
        prior_correct=True
    ):
        """
        Builds C-dim semantic vectors (posteriors) via unified matrix diffusion.
        
        Logic:
        1. Construct Signal Basis Y_encoded (K x C): 
            - If Full: One-hot encoding of labeled points.
            - If Landmarks: Probability distribution P(class | landmark).
        2. Diffuse: Post = W @ Y_encoded
        3. Correct: Divide by class priors to handle imbalance.
        4. Transform: Apply Metric scaling (Cosine/Hellinger).
        """
        # --- Setup Data & Labels ---
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)
        
        N, K = W.shape
        C = len(labels)
        
        # Map labels -> [0..C-1]
        lab2idx = {lab: k for k, lab in enumerate(labels)}
        
        # Filter strictly to labeled data
        y_lab = y[~mask_unl]
        if y_lab.size == 0:
            return np.zeros((N, C), dtype=float)

        try:
            y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(f"Found label {e} in y that is not in `labels`.")

        # --- Compute Priors (for correction) ---
        # We compute this regardless of path to support prior_correct
        counts_c = np.bincount(y_idx, minlength=C).astype(float)
        n_lab = float(counts_c.sum())
        
        # p_c = count / total
        class_prior = counts_c / max(n_lab, 1.0)
        # 1/p_c (used to normalize "Total Affinity" to "Average Affinity")
        inv_prior = 1.0 / np.maximum(class_prior, eps)

        # --- Construct Signal Basis Y_encoded (K x C) ---
        # This matrix represents the initial class signal on the graph nodes (points or landmarks)
        Y_encoded = np.zeros((K, C), dtype=np.float64)

        if clusters is None:
            # Case A: Full Graph (Basis = Points, K=N)
            # Create strict One-Hot encoding for labeled points
            # Y[i, c] = 1.0 if point i has label c, else 0
            labeled_indices = np.flatnonzero(~mask_unl)
            Y_encoded[labeled_indices, y_idx] = 1.0
            
        else:
            # Case B: Landmarks (Basis = Landmarks, K=M)
            # Aggregate counts: "Landmark k contains 5 Class A and 10 Class B"
            clusters = np.asarray(clusters).ravel()
            landmark_ids = clusters[~mask_unl] # Map labeled points to their landmarks
            
            # Fast unbuffered add
            np.add.at(Y_encoded, (landmark_ids, y_idx), 1.0)
            
            # Row-Normalize (Counts -> Probs P(c|m))
            row_sums = Y_encoded.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0 # Prevent div/0 for empty landmarks
            Y_encoded /= row_sums

        # --- Diffusion (The Projection) ---
        # W is sparse (N, N) or dense (N, K), Y is dense (K, C) -> Result is Dense (N, C)
        # This calculates the raw sum of affinities to class signals
        post = W.dot(Y_encoded) if sparse.issparse(W) else np.dot(W, Y_encoded)

        # --- Prior Correction ---
        if prior_correct:
            # Broadcast multiplication: post[:, c] *= (1 / p_c)
            # Converts "Total Affinity" -> "Density-Independent Affinity"
            post *= inv_prior[None, :]

        # --- Metric Transformation ---
        # Prepares vectors so standard Euclidean distance downstream matches desired metric
        if self.semantic_norm == 'l2':  # for Cosine distance, same as MALI but less theoretical justification here for MiniBatchKMeans
            # L2 normalize -> scale by 1/sqrt(2)
            # Resulting Euclidean dist = sqrt(1 - cos_sim)
            post = preprocessing.normalize(post, norm="l2", axis=1)
            post /= np.sqrt(2)

        elif self.semantic_norm == 'l1':
            # We JUST L1 normalize to ensure scale invariance between domains
            # This preserves the "relative confidence" while fixing the "density mismatch" bug
            # This also ensures MiniBatchKMeans computes 'valid' centroids (mean of probs = probs)
            post = preprocessing.normalize(post, norm="l1", axis=1)
        else:
            print(f"[WARN] Unknown normalization={self.semantic_norm}, skipping metric transform.")

        return post

    # ------------------------------------------------------------
    # DPT / Landmark diffusion machinery
    # ------------------------------------------------------------
    def _get_diffusion_operators(self, K, random_state=None, verbose=True, **graph_kwargs):
        """
        Returns P_NM (N x M), P_MM (M x M), clusters (N,)
        """
        n_landmark = self.n_landmark
        G = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark if n_landmark is not None and n_landmark < K.shape[0] else None,
            kernel_symm=None,
            random_state=random_state,
            verbose=verbose,
            **graph_kwargs,
        )
        if hasattr(G, "landmark_op") and hasattr(G, "transitions"):
            if verbose:
                print("Using LandmarkGraph operators (N→M and M→M).")
            P_MM = np.asarray(G.landmark_op)
            P_NM = G.transitions.toarray()
            clusters = np.asarray(G.clusters).ravel()

            # robust remap
            M = P_NM.shape[1]
            _, clusters_remap = np.unique(clusters, return_inverse=True)
            clusters = clusters_remap.astype(int)
            if clusters.max() + 1 > M:
                clusters = np.minimum(clusters, M - 1)
            if P_MM.shape[0] != M:
                P_MM = P_MM[:M, :M]
            return P_NM, P_MM, clusters

        if verbose:
            print("Using TraditionalGraph: treating all N points as landmarks.")
        P_MM = G.P.toarray()
        P_NM = np.eye(P_MM.shape[0], dtype=P_MM.dtype)
        clusters = np.arange(P_MM.shape[0])
        return P_NM, P_MM, clusters

    def _compute_dpt(self, P):
        """
        DPT-like aggregated transition matrix: (I - (P - 1 phi^T))^{-1} - I
        """
        n = P.shape[0]
        I = np.eye(n)
        ones = np.ones(n, dtype=float)
        wL, lv = np.linalg.eig(P.T)
        j = np.argmin(np.abs(wL - 1.0))
        phi0 = lv[:, j].real
        if phi0.sum() < 0:
            phi0 = -phi0
        phi0 = np.maximum(phi0, 0)
        s = phi0.sum()
        if s <= 0:
            raise ValueError("Failed to extract a valid stationary distribution.")
        phi0 = phi0 / s
        P_deflated = P - np.outer(ones, phi0)
        Mmat = np.linalg.solve(I - P_deflated, I) - I
        Mmat = preprocessing.MinMaxScaler().fit_transform(Mmat.transpose()).transpose()
        return Mmat

    # ------------------------------------------------------------
    # Compressed OT Logic
    # ------------------------------------------------------------
    def _solve_compressed_ot(self, post_fixed, post_to_compress):
        """
        Solves Optimal Transport where the larger domain (`post_to_compress`) is 
        compressed to match the size of the smaller domain (`post_fixed`).

        This method implements a locally adaptive version of the "Mass Rebalancing" 
        strategy described in the MALI paper (Section D).

        Mechanism:
        ----------
        1. Clustering as Mass Rebalancing:
            Instead of a global uniform weight, we cluster the large domain to find 
            representatives. The mass of each representative is distributed equally 
            among its cluster members.
            
            Weight(point p) = 1 / |Cluster_Size|
            
            - Dense regions -> Large Clusters -> Small individual weights.
            - Sparse regions -> Small Clusters -> Large individual weights.
            
            This automatically fulfills the theoretical suggestion to "increase the 
            masses of samples belonging to low density regions," ensuring alignment 
            is driven by geometric structure rather than sampling density.

        Returns:
            T_intermediate (sparse matrix): Shape (n_fixed, n_compress).
        
        -------------------------------------------------------------------------
        SCENARIO 1: N_a < N_b (Domain A is Small/Fixed)
        -------------------------------------------------------------------------
        - We call: _solve_compressed_ot(post_a, post_b)
        - Returns: T of shape (N_a, N_b).
        - This IS the final Coupling Matrix T.
        - Row Sums (N_a): Strictly 1.0 (Bijection to Centroids).
        - Col Sums (N_b): ~ 1/|Cluster| (Soft assignment via mass rebalancing).

        -------------------------------------------------------------------------
        SCENARIO 2: N_b < N_a (Domain B is Small/Fixed)
        -------------------------------------------------------------------------
        - We call: _solve_compressed_ot(post_b, post_a)
        - Returns: T_intermediate of shape (N_b, N_a).
        - We TRANSPOSE this in .fit() to get final T of shape (N_a, N_b).
        """
        n_fixed = post_fixed.shape[0]          # Size of Small Domain
        n_compress = post_to_compress.shape[0] # Size of Large Domain
        k = n_fixed  # Compress large domain to match small domain size

        # Compress Large Domain (Adaptive Mass Calculation)
        # We cluster the large domain to find 'k' representatives.
        # This implicitly defines the mass of each point based on local density.
        km = MiniBatchKMeans(n_clusters=k, random_state=self.random_state)
        z = km.fit_predict(post_to_compress)         # (n_compress,) assignments
        post_centroids = km.cluster_centers_         # (k, C) semantic centers

        #Solve OT (Bijective Mapping)
        # Map Small Domain <-> Centroids of Large Domain.
        # Since sizes match (n_fixed == k), HiRef forces a 1-to-1 matching.
        rank_schedule = rank_annealing.optimal_rank_schedule(n=n_fixed)
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            post_fixed, post_centroids,
            rank_schedule=rank_schedule,
            return_coupling=True
        )
        
        # T_coarse shape: (n_fixed, k)
        # Row sums = 1.0 (Perfect Bijection found)
        T_coarse = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_fixed, k)
        ).tocsr()

        # Build Membership Matrix (M)
        # Distribute centroid mass equally to constituent points in Large Domain.
        # M shape: (k, n_compress)
        counts = np.bincount(z, minlength=k).astype(np.float64)
        
        # Safe inverse for density weighting:
        # Weight = 1 / |Cluster_Size| (The "Mass Rebalancing" term)
        inv_counts = np.zeros_like(counts)
        inv_counts[counts > 0] = 1.0 / counts[counts > 0]
        
        M_membership = sparse.coo_matrix(
            (inv_counts[z], (z, np.arange(n_compress, dtype=np.int64))),
            shape=(k, n_compress)
        ).tocsr()

        # Lift to Full Resolution
        # T_intermediate = T_coarse @ M_membership
        # Shape: (n_fixed, n_compress)
        #
        # Verification:
        # - Row i (Small Domain point): Sums to 1.0.
        # - Col j (Large Domain point): Sums to 1/|Cluster|.
        T_intermediate = (T_coarse @ M_membership).tocsr()

        n_empty = int(np.sum(counts == 0))
        if n_empty > 0:
            print(f"[WARN] MiniBatchKMeans produced {n_empty}/{k} empty clusters.")

        return T_intermediate
    
    
    # ------------------------------------------------------------
    # Balanced Affinity Construction
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix just like in MALI, with max-normalized T as input (OT coupling matrix)
        """
        W_ab = (prox_a.dot(T) + T.dot(prox_b))   # (n_a x n_b)
        W_ba = W_ab.transpose()
        W_sym = sparse.bmat(
            [
                [self.mu * prox_a, (1-self.mu) * W_ab],
                [(1-self.mu) * W_ba,   self.mu * prox_b]
            ],
            format="csr"
        )
        return W_sym

    # ------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------
    def fit(self, x_a, x_b, y_a, y_b):
        self.n_a = x_a.shape[0]
        self.n_b = x_b.shape[0]
        self.n = self.n_a + self.n_b

        y_a = np.asarray(y_a).ravel()
        y_b = np.asarray(y_b).ravel()
        labels = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)
        self.classes_ = labels

        print("Fitting RFGAP on Domain A...") if self.verbose > 0 else None
        self.rfgap_a = RFGAP(**self.rfgap_params)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities()

        print("Fitting RFGAP on Domain B...") if self.verbose > 0 else None
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()

        print("Building C-dim vectors...") if self.verbose > 0 else None
        if not self.dpt:
            post_a = self._get_semantic_vectors(prox_a, y_a, labels, clusters=None, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(prox_b, y_b, labels, clusters=None, prior_correct=self.prior_correct)
        else:
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(prox_a, random_state=self.random_state, verbose=True)
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(prox_b, random_state=self.random_state, verbose=True)
            M_a = self._compute_dpt(P_MM_a)
            M_b = self._compute_dpt(P_MM_b)
            trans_a = P_NM_a.dot(M_a)
            trans_b = P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b, prior_correct=self.prior_correct)

        print("Computing Optimal Transport...") if self.verbose > 0 else None
        n_a, n_b = self.n_a, self.n_b

        if n_a == n_b:
            # --- Balanced Case (equivalent to n_a=n_b with m=1 in MALI) ---
            rank_schedule = rank_annealing.optimal_rank_schedule(n=n_a)
            (rows, cols, data), _ = HiRef.hiref_lr_fast(
                post_a, post_b,
                rank_schedule=rank_schedule,
                return_coupling=True
            )
            self.T_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(n_a, n_b)).tocsr()

        # Imbalanced cases, correspond to n_a =! n_b with m=1 in MALI
        elif n_a < n_b:
            # --- Compress B -> A ---
            # Returns T (n_a x n_b) directly
            self.T_sparse = self._solve_compressed_ot(post_fixed=post_a, post_to_compress=post_b)

        else: # n_b < n_a
            # --- Compress A -> B ---
            # Returns T (n_b x n_a). We need A->B, so we transpose.
            T_ba = self._solve_compressed_ot(post_fixed=post_b, post_to_compress=post_a)
            self.T_sparse = T_ba.T.tocsr()

        # ---------- DIAGNOSTICS ----------
        T = self.T_sparse.tocsr()
        if self.verbose:
            print("\nSPARSE COUPLING")
            print("---------------")
            print(f"Empty rows: {np.sum(np.diff(T.indptr) == 0)} / {T.shape[0]}")
            print(f"Empty cols: {np.sum(np.diff(T.tocsc().indptr) == 0)} / {T.shape[1]}")
            print("Total mass:", T.sum())
            # print row/col sums
            row_sums = np.asarray(T.sum(axis=1)).ravel()
            col_sums = np.asarray(T.sum(axis=0)).ravel()
            print(f"Row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")
            print(f"Col sums: min={col_sums.min():.4f}, max={col_sums.max():.4f}, mean={col_sums.mean():.4f}")
            print("Building joint affinity matrix...")

        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

        print("Model fit complete.") if self.verbose > 0 else None
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        print("Computing joint embedding...")

        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                kernel_symm=None,  # already use pre-symmetrized affinity
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=-1,
                beta=0.5,
            )
            self.embedding_ = phate_op.fit_transform(self.W)
            return self.embedding_

        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.embedding_ = embedder.fit_transform(self.W)
            return self.embedding_

        elif self.embedder == 'UMAP':
            DistM = kernel2Dist(self.W.toarray())
            self.embedding_ = UMAP(
                n_components=self.n_components,
                metric='precomputed',
                random_state=self.random_state
            ).fit_transform(DistM)
            return self.embedding_

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

    def get_embeddings(self):
        return self.embedding_