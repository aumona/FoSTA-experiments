import time
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

import graphtools
from .rfgap_old.forestkernel import ForestKernel
from .hiref.adaptive_HiRef import solve_surjection_hiref
from src.phate import PageRankPHATE
from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment.
    """
    
    def __init__(self, mu=0.5, kernel_method='gap',
                 force_nonzero_diag=False,
                 normalize_diagonal=False,
                 model_type='rf', euclidean_mode=False,
                 n_pca=100, n_neighbors=5, decay=40, knn_dist='euclidean',
                 t='auto', beta=0.9, n_estimators=500, prior_correct=True,
                 semantic_norm='l2', embedder='PHATE', n_components=2,
                 verbose=0, random_state=None, n_jobs=-1):
        
        self.mu = mu
        self.kernel_method = kernel_method
        self.normalize_diagonal = normalize_diagonal
        self.force_nonzero_diag = force_nonzero_diag
        self.model_type = model_type
        self.euclidean_mode = euclidean_mode
        self.n_pca, self.n_neighbors = n_pca, n_neighbors
        self.decay, self.knn_dist = decay, knn_dist
        self.prior_correct = prior_correct
        self.t, self.beta = t, beta
        self.semantic_norm = semantic_norm
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

        self.prox_a = self.prox_b = self.T_sparse = self.W = self.embedding_ = None
        self.classes_ = self.n = self.n_a = self.n_b = None
        self.runtime_summary_ = None

    def _get_semantic_vectors(self, W, y, labels, eps=1e-12, prior_correct=True):
        """Builds C-dim semantic vectors (posteriors) via unified matrix diffusion."""
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)
        N, C = W.shape[0], len(labels)
        lab2idx = {lab: k for k, lab in enumerate(labels)}

        if not np.any(~mask_unl): return np.zeros((N, C))

        y_idx = np.array([lab2idx[v] for v in y[~mask_unl]], dtype=int)
        counts_c = np.bincount(y_idx, minlength=C).astype(float)
        inv_prior = 1.0 / np.maximum(counts_c / max(counts_c.sum(), 1.0), eps)

        Y_encoded = np.zeros((N, C))
        Y_encoded[np.flatnonzero(~mask_unl), y_idx] = 1.0
        
        # Diffusion
        post = W.dot(Y_encoded)

        if self.verbose:
            r_sums = post.sum(axis=1)
            # Use scientific notation for mean to catch those giant numbers if they return
            print(f"[DEBUG] Semantic Row Sums: min={r_sums.min():.4f}, mean={r_sums.mean():.4e}")

        if prior_correct:
            post *= inv_prior[None, :]
        
        if self.semantic_norm in ['l1', 'l2']:
            post = preprocessing.normalize(post, norm=self.semantic_norm, axis=1)
            
        return post


    def _apply_alpha_sharpening(self, K, k, a):
        """
        Sparsity-preserving Alpha-Sharpening with Pythonic Max-Normalization.
        """
        K = K.tocsr()
        n_samples = K.shape[0]
        
        # 1. Row-normalize by MAX (Pythonic way)
        # This ensures the best neighbor in every row has affinity 1.0
        K = preprocessing.normalize(K, norm='max', axis=1)

        # 2. Identify k-th neighbor threshold
        sigmas = np.zeros(n_samples)
        for i in range(n_samples):
            row = K.data[K.indptr[i]:K.indptr[i+1]]
            if len(row) >= k:
                sigmas[i] = np.partition(row, -k)[-k]
            else:
                sigmas[i] = np.min(row) if len(row) > 0 else 1e-12

        # 3. Vectorized Power Sharpening
        sigmas = np.maximum(sigmas, 1e-12)
        sigma_repeats = np.repeat(sigmas, np.diff(K.indptr))
        
        # (K / sigma)^a, clipped at 1.0 to prevent numerical explosion
        sharpened_data = np.power(np.minimum(K.data / sigma_repeats, 1.0), a)
        
        K_new = sparse.csr_matrix((sharpened_data, K.indices, K.indptr), shape=K.shape)
        
        # 4. Symmetrize (PHATE Average)
        K_final = (K_new + K_new.T) / 2
        K_final.eliminate_zeros()
        
        return K_final


    def _get_geometry(self, x, y):
        """
        Extracts the raw geometry. We keep it unsharpened here to 
        preserve global semantic signal for transport/diffusion.
        """
        if self.euclidean_mode:
            # For Euclidean, we'll still use graphtools but with no decay yet
            return graphtools.Graph(x, knn=self.n_neighbors, decay=None).K

        fk = ForestKernel(n_estimators=self.n_estimators, kernel_method=self.kernel_method, 
                            random_state=self.random_state)
        fk.fit(x, y, idx_unlabeled=np.flatnonzero(LabelUtils.get_unlabeled_mask(y)))
        
        K = fk.get_kernel(normalize_diagonal=self.normalize_diagonal)
        K.data = np.clip(K.data, 0, 1) 
        return K
    
        
    

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """Constructs joint affinity matrix from domain geometries and transport plan."""
        # Note: prox_a and prox_b are already sharpened
        W_ab = prox_a.dot(T)
        W_ba = prox_b.dot(T.transpose())
        
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A", prox_a)
            print_mat_stats("Within-domain B", prox_b)
            print_mat_stats("Cross-domain A→B", W_ab)
            print_mat_stats("Cross-domain B→A", W_ba)

        return sparse.bmat([
            [(1-self.mu)*prox_a, self.mu*W_ab],
            [self.mu*W_ba, (1-self.mu)*prox_b]
        ], format="csr")

    def fit(self, x_a, x_b, y_a, y_b):
        """Fits FoSTA alignment across two domains."""
        self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
        self.classes_ = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)
    
        # 1. Get RAW Geometry (Broad proximities for better label diffusion)
        start = time.perf_counter()
        self.prox_a = self._get_geometry(x_a, y_a)
        self.prox_b = self._get_geometry(x_b, y_b)
        t_geom = time.perf_counter() - start
    
        # 2. Extract Semantic Vectors (using the broad RAW kernels)
        start = time.perf_counter()
        post_a = self._get_semantic_vectors(self.prox_a, y_a, self.classes_, prior_correct=self.prior_correct)
        post_b = self._get_semantic_vectors(self.prox_b, y_b, self.classes_, prior_correct=self.prior_correct)
        t_sem = time.perf_counter() - start
    
        # 3. Transport Alignment
        start = time.perf_counter()
        T = solve_surjection_hiref(post_a, post_b, verbose=self.verbose, random_state=self.random_state)
        self.T_sparse = T.tocsr()
        t_trans = time.perf_counter() - start
    
        # 4. Construct Joint Affinity (SHARPEN ONLY NOW)
        start = time.perf_counter()
        
        # Apply alpha-sharpening to narrow down the manifold for embedding
        prox_a_sharp = self._apply_alpha_sharpening(self.prox_a, k=self.n_neighbors, a=self.decay)
        prox_b_sharp = self._apply_alpha_sharpening(self.prox_b, k=self.n_neighbors, a=self.decay)
        
        # Build the final matrix using sharpened within-domain links
        self.W = self._build_balanced_affinity(prox_a_sharp, prox_b_sharp, self.T_sparse)
        t_joint = time.perf_counter() - start
    
        self.runtime_summary_ = {
            "geometry": t_geom, "semantics": t_sem, "transport": t_trans, 
            "joint_affinity": t_joint, "embedding": 0.0, 
            "total": t_geom + t_sem + t_trans + t_joint
        }
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """Fits alignment and computes embedding."""
        self.fit(x_a, x_b, y_a, y_b)
        start = time.perf_counter()

        if self.embedder == 'PHATE':
            self.embedding_ = PageRankPHATE(
                n_components=self.n_components, t=self.t, knn_dist='precomputed_affinity',
                kernel_symm='+', random_state=self.random_state, verbose=self.verbose,
                n_jobs=self.n_jobs, beta=self.beta
            ).fit_transform(self.W)
        elif self.embedder == 'spectral':
            self.embedding_ = SpectralEmbedding(
                n_components=self.n_components, affinity='precomputed',
                random_state=self.random_state, n_jobs=self.n_jobs
            ).fit_transform(self.W)
        elif self.embedder == 'UMAP':
            self.embedding_ = UMAP(
                n_components=self.n_components, metric='precomputed', 
                random_state=self.random_state
            ).fit_transform(kernel2Dist(self.W.toarray()))

        self.runtime_summary_["embedding"] = time.perf_counter() - start
        self.runtime_summary_["total"] += self.runtime_summary_["embedding"]
        
        if self.verbose: 
            self._print_runtime_summary()
        
        return self.embedding_

    def _print_runtime_summary(self):
        print(f"\nRUNTIME SUMMARY\nTotal: {self.runtime_summary_['total']:.3f}s")
        for k, v in self.runtime_summary_.items():
            if k == 'total': continue
            print(f"{k:15s} {v:8.3f}s ({100*v/self.runtime_summary_['total']:5.1f}%)")