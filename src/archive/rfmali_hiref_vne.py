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
                 random_state=None, n_jobs=-1, dpt=True):
        self.mu = mu
        self.n_components = n_components
        self.knn_dist = knn_dist
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.dpt = dpt
        
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

        if self.dpt:
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                knn_dist=self.knn_dist,
                t='auto',
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=-1,
                beta=0.9,
                n_landmark=200,
                gamma=-1
            )

            # Source Domain A
            phate_op.fit(prox_a)
            U_NM_t_a = phate_op.diff_potential   # Potential coordinates NxM (-log P_MM_t) (inf dim PHATE)
            if isinstance(phate_op.graph, graphtools.graphs.LandmarkGraph):
                clusters = phate_op.graph.clusters
                # remap cluster ids to consecutive integers (0..K-1)
                _, clusters_remap = np.unique(clusters, return_inverse=True)
                clusters_a = clusters_remap.astype(int)
            else:
                clusters_a = np.arange(U_NM_t_a.shape[1])

            # Source Domain B
            phate_op.fit(prox_b)
            U_NM_t_b = phate_op.diff_potential   # Potential coordinates NxM (-log P_MM_t)
            if isinstance(phate_op.graph, graphtools.graphs.LandmarkGraph):
                clusters = phate_op.graph.clusters
                # remap cluster ids to consecutive integers (0..K-1)
                _, clusters_remap = np.unique(clusters, return_inverse=True)
                clusters_b = clusters_remap.astype(int)
            else:
                clusters_b = np.arange(U_NM_t_b.shape[1])


            # Build C-dimensional Posteriors using the Landmark cluster assignments and the existing labels
            #NOTE: Assumption: y_a and y_b have the same label space, with at least one point per class in each domain
            # Build Balanced Posteriors using the Landmark cluster assignments and the existing labels
            print("Building Balanced Posteriors...")
            post_a = self._get_balanced_posteriors(U_NM_t_a, y_a, clusters_a)
            post_b = self._get_balanced_posteriors(U_NM_t_b, y_b, clusters_b)
            
            
            # Compute Optimal Transport with HiRef (fast near linear OT)
            print("Computing Optimal Transport...")
            rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n)
            frontier = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule)
            print(frontier)
        
        else:
            # Compute Posteriors
            print("Building Posteriors...")
            labels_a = np.unique(y_a)
            labels_b = np.unique(y_b)
            post_a = self._get_rfgap_posteriors(prox_a, y_a, labels_a)
            post_b = self._get_rfgap_posteriors(prox_b, y_b, labels_b)

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