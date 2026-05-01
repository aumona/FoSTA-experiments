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
    """FoSTA: Forest-guided Semantic Transport Alignment"""
    
    def __init__(self, mu=0.5, kernel_method='gap', force_nonzero_diag=False,
                 normalize_diagonal=False, model_type='rf', euclidean_mode=False,
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
        post = W.dot(Y_encoded)

        # --- Diagnostics: Pre-normalization row sums ---
        if self.verbose:
            r_sums = post.sum(axis=1)
            print(f"\n[DEBUG] Semantic Row Sums (pre-norm): min={r_sums.min():.4f}, max={r_sums.max():.4f}, mean={r_sums.mean():.4f}")

        if prior_correct:
            post *= inv_prior[None, :]
        
        if self.semantic_norm in ['l1', 'l2']:
            post = preprocessing.normalize(post, norm=self.semantic_norm, axis=1)
            
        return post

    def _row_normalize(self, M):
        row_sums = np.asarray(M.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        return sparse.diags(1.0 / row_sums) @ M

    def _top_k_sparsify(self, K, k=5):
        K = K.tocsr()
        n_samples = K.shape[0]
        new_data, new_indices = np.zeros(n_samples * k), np.zeros(n_samples * k, dtype=int)
        new_indptr = np.arange(0, (n_samples + 1) * k, k)

        for i in range(n_samples):
            start, end = K.indptr[i], K.indptr[i+1]
            d, idx = K.data[start:end], K.indices[start:end]
            n_row = len(d)
            if n_row > k:
                p = np.argpartition(d, -k)[-k:]
                new_data[i*k : i*k+k], new_indices[i*k : i*k+k] = d[p], idx[p]
            else:
                new_data[i*k : i*k+n_row], new_indices[i*k : i*k+n_row] = d, idx

        return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=K.shape).tocsr()

    def _fill_orphan_matches(self, T, post_a, post_b):
        orph_a = np.where(np.asarray(T.sum(axis=1)).ravel() == 0)[0]
        orph_b = np.where(np.asarray(T.sum(axis=0)).ravel() == 0)[0]

        if not (len(orph_a) or len(orph_b)): return T

        new_r, new_c = [], []
        if len(orph_a):
            idx = NearestNeighbors(n_neighbors=1).fit(post_b).kneighbors(post_a[orph_a], return_distance=False)
            new_r.extend(orph_a); new_c.extend(idx.ravel())
        if len(orph_b):
            idx = NearestNeighbors(n_neighbors=1).fit(post_a).kneighbors(post_b[orph_b], return_distance=False)
            new_r.extend(idx.ravel()); new_c.extend(orph_b)

        T_patch = sparse.csr_matrix((np.ones(len(new_r)), (new_r, new_c)), shape=T.shape)
        T_combined = (T + T_patch).tocsr()
        T_combined.data = np.ones_like(T_combined.data)
        return T_combined

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """Constructs joint affinity matrix with row-normalized blocks and diagnostics."""
        T_ab, T_ba = self._row_normalize(T), self._row_normalize(T.T.tocsr())
        
        # Sparsify and re-normalize intra-domain kernels
        prox_a_sparse = self._row_normalize(self._top_k_sparsify(prox_a, k=30))
        prox_b_sparse = self._row_normalize(self._top_k_sparsify(prox_b, k=30))

        W_ab, W_ba = prox_a_sparse @ T_ab, prox_b_sparse @ T_ba
        
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A", prox_a_sparse)
            print_mat_stats("Within-domain B", prox_b_sparse)
            print_mat_stats("Cross-domain A→B", W_ab)
            print_mat_stats("Cross-domain B→A", W_ba)

        return sparse.bmat([
            [(1-self.mu)*prox_a_sparse, self.mu*W_ab],
            [self.mu*W_ba, (1-self.mu)*prox_b_sparse]
        ], format="csr")

    def _get_geometry(self, x, y):
        if not self.euclidean_mode:
            fk = ForestKernel(n_estimators=self.n_estimators, kernel_method=self.kernel_method, 
                              model_type=self.model_type, force_nonzero_diag=self.force_nonzero_diag,
                              random_state=self.random_state, prediction_type='classification')
            mask = LabelUtils.get_unlabeled_mask(y)
            fk.fit(x, y, idx_unlabeled=np.flatnonzero(mask))
            K = fk.get_kernel(normalize_diagonal=self.normalize_diagonal)
            K.data = np.maximum(0, K.data)
            return K
        
        n_pca = min(self.n_pca, x.shape[1]) if self.n_pca and self.n_pca >= 100 else None
        return graphtools.Graph(x, n_pca=n_pca, knn=self.n_neighbors, decay=self.decay, 
                                distance=self.knn_dist, thresh=1e-4, n_jobs=self.n_jobs, 
                                random_state=self.random_state, verbose=False).K

    def fit(self, x_a, x_b, y_a, y_b):
        self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
        self.classes_ = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)

        # 1. Geometry
        start = time.perf_counter()
        self.prox_a, self.prox_b = self._get_geometry(x_a, y_a), self._get_geometry(x_b, y_b)
        t_geom = time.perf_counter() - start

        # 2. Semantics
        start = time.perf_counter()
        post_a = self._get_semantic_vectors(self.prox_a, y_a, self.classes_, prior_correct=self.prior_correct)
        post_b = self._get_semantic_vectors(self.prox_b, y_b, self.classes_, prior_correct=self.prior_correct)
        t_sem = time.perf_counter() - start

        # 3. Transport & Alignment
        start = time.perf_counter()
        T = solve_surjection_hiref(post_a, post_b, verbose=self.verbose, random_state=self.random_state)
        self.T_sparse = self._fill_orphan_matches(T, post_a, post_b)
        if self.verbose:
            print_mat_stats("\nCoupling Matrix T", self.T_sparse)
        t_trans = time.perf_counter() - start

        # 4. Joint Affinity
        start = time.perf_counter()
        self.W = self._build_balanced_affinity(self.prox_a, self.prox_b, self.T_sparse)
        t_joint = time.perf_counter() - start

        self.runtime_summary_ = {"geometry": t_geom, "semantics": t_sem, "transport": t_trans, 
                                 "joint_affinity": t_joint, "embedding": 0.0, 
                                 "total": t_geom + t_sem + t_trans + t_joint}
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        start = time.perf_counter()

        if self.embedder == 'PHATE':
            self.embedding_ = PageRankPHATE(n_components=self.n_components, t=self.t, knn_dist='precomputed_affinity',
                                            kernel_symm='+', random_state=self.random_state, verbose=self.verbose,
                                            n_jobs=self.n_jobs, beta=self.beta).fit_transform(self.W)
        elif self.embedder == 'spectral':
            self.embedding_ = SpectralEmbedding(n_components=self.n_components, affinity='precomputed',
                                                random_state=self.random_state, n_jobs=self.n_jobs).fit_transform(self.W)
        elif self.embedder == 'UMAP':
            self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', 
                                   random_state=self.random_state).fit_transform(kernel2Dist(self.W.toarray()))

        self.runtime_summary_["embedding"] = time.perf_counter() - start
        self.runtime_summary_["total"] += self.runtime_summary_["embedding"]
        if self.verbose: self._print_runtime_summary()
        
        return self.embedding_

    def _print_runtime_summary(self):
        print(f"\nRUNTIME SUMMARY\nTotal: {self.runtime_summary_['total']:.3f}s")
        for k, v in self.runtime_summary_.items():
            if k == 'total': continue
            print(f"{k:15s} {v:8.3f}s ({100*v/self.runtime_summary_['total']:5.1f}%)")