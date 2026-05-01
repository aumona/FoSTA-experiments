# import time
# import numpy as np
# from scipy import sparse
# from sklearn import preprocessing
# from sklearn.neighbors import NearestNeighbors
# from sklearn.manifold import SpectralEmbedding
# from umap import UMAP

# import graphtools
# from .rfgap_old.forestkernel import ForestKernel
# from .hiref.adaptive_HiRef import solve_surjection_hiref
# from src.phate import PageRankPHATE
# from utils.utils import kernel2Dist, print_mat_stats
# from utils.labels import LabelUtils

# class FoSTA:
#     """FoSTA: Forest-guided Semantic Transport Alignment"""
    
#     def __init__(self, mu=0.5, kernel_method='gap', force_nonzero_diag=False,
#                  normalize_diagonal=False, model_type='rf', euclidean_mode=False,
#                  n_pca=100, n_neighbors=5, decay=40, knn_dist='euclidean',
#                  t='auto', beta=0.9, n_estimators=500, prior_correct=True,
#                  semantic_norm='l2', embedder='PHATE', n_components=2,
#                  verbose=0, random_state=None, n_jobs=-1):
        
#         self.mu = mu
#         self.kernel_method = kernel_method
#         self.normalize_diagonal = normalize_diagonal
#         self.force_nonzero_diag = force_nonzero_diag
#         self.model_type = model_type
#         self.euclidean_mode = euclidean_mode
#         self.n_pca, self.n_neighbors = n_pca, n_neighbors
#         self.decay, self.knn_dist = decay, knn_dist
#         self.prior_correct = prior_correct
#         self.t, self.beta = t, beta
#         self.semantic_norm = semantic_norm
#         self.embedder = embedder
#         self.n_components = n_components
#         self.random_state = random_state
#         self.verbose = verbose
#         self.n_jobs = n_jobs
#         self.n_estimators = n_estimators

#         self.prox_a = self.prox_b = self.T_sparse = self.W = self.embedding_ = None
#         self.classes_ = self.n = self.n_a = self.n_b = None
#         self.runtime_summary_ = None

#     def _get_semantic_vectors(self, W, y, labels, eps=1e-12, prior_correct=True):
#         """Builds C-dim semantic vectors (posteriors) via unified matrix diffusion."""
#         y = np.asarray(y).ravel()
#         mask_unl = LabelUtils.get_unlabeled_mask(y)
#         N, C = W.shape[0], len(labels)
#         lab2idx = {lab: k for k, lab in enumerate(labels)}

#         if not np.any(~mask_unl): return np.zeros((N, C))

#         y_idx = np.array([lab2idx[v] for v in y[~mask_unl]], dtype=int)
#         counts_c = np.bincount(y_idx, minlength=C).astype(float)
#         inv_prior = 1.0 / np.maximum(counts_c / max(counts_c.sum(), 1.0), eps)

#         Y_encoded = np.zeros((N, C))
#         Y_encoded[np.flatnonzero(~mask_unl), y_idx] = 1.0
#         post = W.dot(Y_encoded)

#         # --- Diagnostics: Pre-normalization row sums ---
#         if self.verbose:
#             r_sums = post.sum(axis=1)
#             print(f"\n[DEBUG] Semantic Row Sums (pre-norm): min={r_sums.min():.4f}, max={r_sums.max():.4f}, mean={r_sums.mean():.4f}")

#         if prior_correct:
#             post *= inv_prior[None, :]
        
#         if self.semantic_norm in ['l1', 'l2']:
#             post = preprocessing.normalize(post, norm=self.semantic_norm, axis=1)
            
#         return post

#     def _row_normalize(self, M):
#         row_sums = np.asarray(M.sum(axis=1)).flatten()
#         row_sums[row_sums == 0] = 1.0
#         return sparse.diags(1.0 / row_sums) @ M

#     def _top_k_sparsify(self, K, k=5):
#         K = K.tocsr()
#         n_samples = K.shape[0]
#         new_data, new_indices = np.zeros(n_samples * k), np.zeros(n_samples * k, dtype=int)
#         new_indptr = np.arange(0, (n_samples + 1) * k, k)

#         for i in range(n_samples):
#             start, end = K.indptr[i], K.indptr[i+1]
#             d, idx = K.data[start:end], K.indices[start:end]
#             n_row = len(d)
#             if n_row > k:
#                 p = np.argpartition(d, -k)[-k:]
#                 new_data[i*k : i*k+k], new_indices[i*k : i*k+k] = d[p], idx[p]
#             else:
#                 new_data[i*k : i*k+n_row], new_indices[i*k : i*k+n_row] = d, idx

#         return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=K.shape).tocsr()

#     def _fill_orphan_matches(self, T, post_a, post_b):
#         orph_a = np.where(np.asarray(T.sum(axis=1)).ravel() == 0)[0]
#         orph_b = np.where(np.asarray(T.sum(axis=0)).ravel() == 0)[0]

#         if not (len(orph_a) or len(orph_b)): return T

#         new_r, new_c = [], []
#         if len(orph_a):
#             idx = NearestNeighbors(n_neighbors=1).fit(post_b).kneighbors(post_a[orph_a], return_distance=False)
#             new_r.extend(orph_a); new_c.extend(idx.ravel())
#         if len(orph_b):
#             idx = NearestNeighbors(n_neighbors=1).fit(post_a).kneighbors(post_b[orph_b], return_distance=False)
#             new_r.extend(idx.ravel()); new_c.extend(orph_b)

#         T_patch = sparse.csr_matrix((np.ones(len(new_r)), (new_r, new_c)), shape=T.shape)
#         T_combined = (T + T_patch).tocsr()
#         T_combined.data = np.ones_like(T_combined.data)
#         return T_combined

#     def _build_balanced_affinity(self, prox_a, prox_b, T):
#         """Constructs joint affinity matrix with row-normalized blocks and diagnostics."""
#         T_ab, T_ba = self._row_normalize(T), self._row_normalize(T.T.tocsr())
        
#         # Sparsify and re-normalize intra-domain kernels
#         prox_a_sparse = self._row_normalize(self._top_k_sparsify(prox_a, k=30))
#         prox_b_sparse = self._row_normalize(self._top_k_sparsify(prox_b, k=30))

#         W_ab, W_ba = prox_a_sparse @ T_ab, prox_b_sparse @ T_ba
        
#         if self.verbose:
#             print("\nJOINT AFFINITY BLOCK STATISTICS")
#             print("------------------------------")
#             print_mat_stats("Within-domain A", prox_a_sparse)
#             print_mat_stats("Within-domain B", prox_b_sparse)
#             print_mat_stats("Cross-domain A→B", W_ab)
#             print_mat_stats("Cross-domain B→A", W_ba)

#         return sparse.bmat([
#             [(1-self.mu)*prox_a_sparse, self.mu*W_ab],
#             [self.mu*W_ba, (1-self.mu)*prox_b_sparse]
#         ], format="csr")

#     def _get_geometry(self, x, y):
#         if not self.euclidean_mode:
#             fk = ForestKernel(n_estimators=self.n_estimators, kernel_method=self.kernel_method, 
#                               model_type=self.model_type, force_nonzero_diag=self.force_nonzero_diag,
#                               random_state=self.random_state, prediction_type='classification')
#             mask = LabelUtils.get_unlabeled_mask(y)
#             fk.fit(x, y, idx_unlabeled=np.flatnonzero(mask))
#             K = fk.get_kernel(normalize_diagonal=self.normalize_diagonal)
#             K.data = np.maximum(0, K.data)
#             return K
        
#         n_pca = min(self.n_pca, x.shape[1]) if self.n_pca and self.n_pca >= 100 else None
#         return graphtools.Graph(x, n_pca=n_pca, knn=self.n_neighbors, decay=self.decay, 
#                                 distance=self.knn_dist, thresh=1e-4, n_jobs=self.n_jobs, 
#                                 random_state=self.random_state, verbose=False).K

#     def fit(self, x_a, x_b, y_a, y_b):
#         self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
#         self.classes_ = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)

#         # 1. Geometry
#         start = time.perf_counter()
#         self.prox_a, self.prox_b = self._get_geometry(x_a, y_a), self._get_geometry(x_b, y_b)
#         t_geom = time.perf_counter() - start

#         # 2. Semantics
#         start = time.perf_counter()
#         post_a = self._get_semantic_vectors(self.prox_a, y_a, self.classes_, prior_correct=self.prior_correct)
#         post_b = self._get_semantic_vectors(self.prox_b, y_b, self.classes_, prior_correct=self.prior_correct)
#         t_sem = time.perf_counter() - start

#         # 3. Transport & Alignment
#         start = time.perf_counter()
#         T = solve_surjection_hiref(post_a, post_b, verbose=self.verbose, random_state=self.random_state)
#         self.T_sparse = self._fill_orphan_matches(T, post_a, post_b)
#         if self.verbose:
#             print_mat_stats("\nCoupling Matrix T", self.T_sparse)
#         t_trans = time.perf_counter() - start

#         # 4. Joint Affinity
#         start = time.perf_counter()
#         self.W = self._build_balanced_affinity(self.prox_a, self.prox_b, self.T_sparse)
#         t_joint = time.perf_counter() - start

#         self.runtime_summary_ = {"geometry": t_geom, "semantics": t_sem, "transport": t_trans, 
#                                  "joint_affinity": t_joint, "embedding": 0.0, 
#                                  "total": t_geom + t_sem + t_trans + t_joint}
#         return self

#     def fit_transform(self, x_a, x_b, y_a, y_b):
#         self.fit(x_a, x_b, y_a, y_b)
#         start = time.perf_counter()

#         if self.embedder == 'PHATE':
#             self.embedding_ = PageRankPHATE(n_components=self.n_components, t=self.t, knn_dist='precomputed_affinity',
#                                             kernel_symm='+', random_state=self.random_state, verbose=self.verbose,
#                                             n_jobs=self.n_jobs, beta=self.beta).fit_transform(self.W)
#         elif self.embedder == 'spectral':
#             self.embedding_ = SpectralEmbedding(n_components=self.n_components, affinity='precomputed',
#                                                 random_state=self.random_state, n_jobs=self.n_jobs).fit_transform(self.W)
#         elif self.embedder == 'UMAP':
#             self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', 
#                                    random_state=self.random_state).fit_transform(kernel2Dist(self.W.toarray()))

#         self.runtime_summary_["embedding"] = time.perf_counter() - start
#         self.runtime_summary_["total"] += self.runtime_summary_["embedding"]
#         if self.verbose: self._print_runtime_summary()
        
#         return self.embedding_

#     def _print_runtime_summary(self):
#         print(f"\nRUNTIME SUMMARY\nTotal: {self.runtime_summary_['total']:.3f}s")
#         for k, v in self.runtime_summary_.items():
#             if k == 'total': continue
#             print(f"{k:15s} {v:8.3f}s ({100*v/self.runtime_summary_['total']:5.1f}%)")





import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, svds
import ot

import graphtools
from forestkernel import ForestKernel

from src.phate import PageRankPHATE
from phate import vne
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref

class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment
    Exact match to reference class logic, using W_lab and Row-Normalization.
    """

    def __init__(
        self,
        mu=0.5,
        kernel_method="gap",
        model_type="rf",
        n_estimators=500,
        n_pca=100,
        n_neighbors=5,
        decay=40,
        knn_dist="euclidean",
        t="auto",
        beta=0.7,
        t_sem_a=None,
        t_sem_b=None,  # None/0: no diffusion, "auto": VNE, or positive integer
        t_sem_max=30,  # maximum t range to consider if t_sem="auto"
        average_semantic_diffusion=True,  # if True, average the semantic vectors across all diffusion scales up to t_sem instead of just taking the final one
        prior_correct=True,
        l2_normalize=True,
        embedder="PHATE",
        n_components=2,
        ot_solver="hiref",
        hierarchy_depth=6,
        max_Q=int(2**10),
        max_rank=16,
        entR=0,
        m=1,
        distance="cosine",
        verbose=1,
        random_state=None,
        n_jobs=-1,
    ):
        # Shared global parameters
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        # ForestKernel parameters
        self.kernel_method = kernel_method
        if self.kernel_method not in {"kerf", "gap"}:
            raise ValueError("FoSTA currently supports only kernel_method='kerf' or 'gap'.")
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.kernel_params = {
            "random_state": random_state,
            "prediction_type": "classification",
            "n_estimators": self.n_estimators,
            "kernel_method": "kerf" if self.kernel_method == "original" else self.kernel_method,
            "model_type": self.model_type,
            "bootstrap": True
        }

        self.n_pca = n_pca
        self.n_neighbors = n_neighbors
        self.decay = decay
        self.knn_dist = knn_dist
        self.l2_normalize = l2_normalize
        self.t_sem_a = t_sem_a
        self.t_sem_b = t_sem_b
        self.t_sem_max = t_sem_max
        self.average_semantic_diffusion = average_semantic_diffusion
        self.prior_correct = prior_correct

        self.ot_solver = ot_solver
        self.hierarchy_depth = hierarchy_depth
        self.max_Q = max_Q
        self.max_rank = max_rank
        self.entR = entR
        self.m = m
        self.distance = distance

        self.mu = mu
        self.t = t
        self.beta = beta
        self.embedder = embedder
        self.n_components = n_components

        # State storage
        self.kernel_a = self.kernel_b = None
        self.leaf_pca_a_ = self.leaf_pca_b_ = None
        self.Q_full_a_ = self.Q_full_b_ = None
        self.W_lab_a_ = self.W_lab_b_ = None  # Using W_lab
        self.idx_lab_a_ = self.idx_lab_b_ = None
        self.idx_unl_a_ = self.idx_unl_b_ = None
        self.T_sparse = self.Distances12 = self.W = self.embedding_ = None
        self.classes_ = self.n = self.n_a = self.n_b = None

    def _log(self, msg):
        if self.verbose: print(msg)

    @staticmethod
    def _assemble_sparse_query_map(Q_lab, Q_unl, idx_lab, idx_unl, n_total):
        Q_lab = Q_lab.tocoo()
        data_parts = [Q_lab.data]
        row_parts = [idx_lab[Q_lab.row]]
        col_parts = [Q_lab.col]
        if Q_unl is not None and Q_unl.shape[0] > 0:
            Q_unl = Q_unl.tocoo()
            data_parts.append(Q_unl.data)
            row_parts.append(idx_unl[Q_unl.row])
            col_parts.append(Q_unl.col)
        return sparse.coo_matrix((np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
                                 shape=(n_total, Q_lab.shape[1])).tocsr()

    
    
    def _svd_p_full(self, Q_full, W_lab, n_components=100, random_state=None):
        n, m = Q_full.shape[0], W_lab.shape[0]
    
        def matvec(v):
            return Q_full @ (W_lab.T @ v)
    
        def rmatvec(u):
            return W_lab @ (Q_full.T @ u)
    
        P_op = LinearOperator(
            shape=(n, m),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )
    
        k = min(n_components, n - 1, m - 1)
    
        U, S, Vt = svds(P_op, k=k, random_state=random_state)
        order = np.argsort(S)[::-1]
    
        U = U[:, order]
        S = S[order]
        Vt = Vt[order]
    
        # PCA-like coordinates of rows of P
        coords = U * S
    
        return coords, S, Vt

    def _build_graph_from_coords(self, coords):
        G = graphtools.Graph(coords, n_pca=None, knn=self.n_neighbors, decay=self.decay, distance=self.knn_dist,
                             thresh=1e-4, n_jobs=self.n_jobs, random_state=self.random_state, verbose=bool(self.verbose))
        return G.K

    def _compute_domain_geometry(self, x, y, domain_name="A"):
        y = np.asarray(y).ravel()
        unl_mask = LabelUtils.get_unlabeled_mask(y)
        idx_lab, idx_unl = np.flatnonzero(~unl_mask), np.flatnonzero(unl_mask)

        self._log(f"\n[Domain {domain_name}] Fitting forest on labeled points...")
        kernel = ForestKernel(**self.kernel_params)
        kernel.fit(x[idx_lab], y[idx_lab])

        # --- EXTRACTING W_lab instead of Q_lab ---
        self._log(f"[Domain {domain_name}] Extracting reference and query maps (W_lab, Q_lab), and assembling full unlabeled+labeled map Q_full...")
        Q_lab = kernel.get_train_query_map().tocsr()
        Q_unl = kernel.get_query_map(x[idx_unl]).tocsr() if len(idx_unl) > 0 else None
        
        W_lab = kernel.get_reference_map().tocsr()
        
        Q_full = self._assemble_sparse_query_map(Q_lab, Q_unl, idx_lab, idx_unl, x.shape[0])


        coords_full, _, _ = self._svd_p_full(
            Q_full,
            W_lab,
            n_components=self.n_pca,
            random_state=self.random_state,
        )
        
        prox = self._build_graph_from_coords(coords_full)

        return coords_full, Q_full, W_lab, kernel, idx_lab, idx_unl, prox
    


    
    def _compute_von_neumann_entropy(self, Q_lab, W_lab, t_max=100):
        n = Q_lab.shape[0]
        k = min(self.n_pca, n - 1, W_lab.shape[0] - 1)
        if k < 1:
            return np.zeros(t_max)
    
        def matvec(v):
            return Q_lab @ (W_lab.T @ v)
    
        def rmatvec(u):
            return W_lab @ (Q_lab.T @ u)
    
        P_op = LinearOperator(
            shape=(Q_lab.shape[0], W_lab.shape[0]),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )
    
        # singular values of P
        _, singular_values, _ = svds(P_op, k=k)
        singular_values = np.sort(singular_values)[::-1]
    
        entropy = []
        singular_values_t = singular_values.copy()
    
        for _ in range(t_max):
            prob = singular_values_t / np.sum(singular_values_t)
            prob = prob + np.finfo(float).eps
            entropy.append(-np.sum(prob * np.log(prob)))
            singular_values_t *= singular_values
    
        return np.asarray(entropy)
    


    def _diffuse_labels(self, Q_lab, W_lab, Y_lab, t_sem):
        """
        Diffuse labeled class probabilities through the labeled semantic operator.
        The labeled operator is never materialized. It is applied as
    
            P_lab Y = Q_lab @ (W_lab.T @ Y)
    
        where P_lab = Q_lab W_lab.T is row-stochastic by construction.
    
        If t_sem is None or 0, no diffusion is applied.
    
        If t_sem == "auto", the diffusion time is selected using the same
        Von Neumann entropy knee criterion used in PHATE, computed from the
        singular values of P_lab in a matrix-free way.
    
        If t_sem is an integer, exactly that many diffusion steps are used.

        If average_semantic_diffusion=True, the returned labels are averaged over all
        diffusion scales:
        
            (Y + P Y + ... + P^t Y) / (t + 1)
        
        
        Otherwise, the returned labels are the final t-step probabilities P^t Y.
        """
        if t_sem is None or t_sem == 0:
            return Y_lab

  
        if t_sem == "auto":
            entropy = self._compute_von_neumann_entropy(
                Q_lab=Q_lab,
                W_lab=W_lab,
                t_max=self.t_sem_max,
            )
            t_opt = int(vne.find_knee_point(entropy))
            self._log(f"Selected semantic diffusion t={t_opt} by PHATE VNE knee.")
        else:
            t_opt = int(t_sem)
            self._log(f"Using fixed semantic diffusion t={t_opt}.")
    
        Y = Y_lab.copy()
        Y_sum = Y.copy()
        
        for _ in range(t_opt):
            Y = Q_lab @ (W_lab.T @ Y)
            Y = np.asarray(Y, dtype=float)
        
            if self.average_semantic_diffusion:
                Y_sum += Y
        
        if self.average_semantic_diffusion:
            return Y_sum / (t_opt + 1)
        
        return Y
    



    def _get_semantic_vectors(self, Q_full, W_lab, y, labels, idx_lab, t_sem):
        y = np.asarray(y).ravel()
        n_classes = len(labels)
        lab2idx = {lab: k for k, lab in enumerate(labels)}
        y_lab = y[idx_lab]
        if y_lab.size == 0: return np.zeros((Q_full.shape[0], n_classes))

        y_lab_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        Y_lab = np.zeros((len(idx_lab), n_classes), dtype=np.float64)
        Y_lab[np.arange(len(idx_lab)), y_lab_idx] = 1.0

        self._log("Projecting onto semantic space...")
        Y_lab = self._diffuse_labels(
            Q_lab=Q_full[idx_lab],
            W_lab=W_lab,
            Y_lab=Y_lab,
            t_sem=t_sem,
        )
        
        S = W_lab.T @ Y_lab
        
        
        post = Q_full @ S
        post = np.asarray(post, dtype=float)

        row_sums = post.sum(axis=1)
        self._log(
            "Semantic row sums before prior correction and l2 normalization: "
            f"min={row_sums.min():.6f}, "
            f"mean={row_sums.mean():.6f}, "
            f"max={row_sums.max():.6f}"
        )

        if self.prior_correct:
            counts = np.bincount(y_lab_idx, minlength=n_classes).astype(float)
            prior = counts / max(counts.sum(), 1.0)
            post *= (1.0 / np.maximum(prior, 1e-12))[None, :]
            post_sums = post.sum(axis=1, keepdims=True)
            post_sums[post_sums <= 1e-12] = 1.0
            post /= post_sums



        if self.l2_normalize:
            post = preprocessing.normalize(post, norm="l2", axis=1)
        return post

    def _compute_dense_ot(self, post_a, post_b):
        self._log("Computing dense OT...")
    
        X = np.asarray(post_a, dtype=float).copy()
        Y = np.asarray(post_b, dtype=float).copy()
    
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        Y = np.nan_to_num(Y, nan=0.0, posinf=1e6, neginf=-1e6)
    
        eps = 1e-12
    
        if self.distance == "cosine":
            row_norms_x = np.linalg.norm(X, axis=1)
            row_norms_y = np.linalg.norm(Y, axis=1)
    
            zero_x = row_norms_x < eps
            zero_y = row_norms_y < eps
    
            if X.shape[1] == 0 or Y.shape[1] == 0:
                raise ValueError("Semantic vectors have zero columns.")
    
            X[zero_x, 0] = eps
            Y[zero_y, 0] = eps
    
        self.Distances12 = cdist(X, Y, self.distance)
        self.Distances12 = np.nan_to_num(
            self.Distances12,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
    
        N1, N2 = X.shape[0], Y.shape[0]
        m_eff = self.m
    
        if N1 == N2:
            if m_eff == 1:
                a = np.repeat(1.0, N1)
                b = np.repeat(1.0, N1)
                transport = "wot" if self.entR == 0 else "wotR"
            else:
                transport = "wotpartial" if self.entR == 0 else "wotpartialR"
                a = np.repeat(1 / N1, N1)
                b = np.repeat(1 / N1, N1)
                m_eff = np.floor(m_eff * N1) / N1
        else:
            if m_eff != 1:
                a = np.repeat(1 / N1, N1)
                b = np.repeat(1 / N2, N2)
                m_eff = np.floor(m_eff * N1) / N1
                transport = "wotpartialR" if self.entR > 0 else "wotpartial"
            else:
                transport = "wotR" if self.entR > 0 else "wot"
                a = np.repeat(1.0, N1).astype(float)
                b = np.repeat(N1 / N2, N2)
    
        C = self.Distances12[:N1, :N2]
    
        if transport == "wot":
            T = ot.emd(a, b, C)
        elif transport == "wotR":
            T = ot.bregman.sinkhorn_log(a, b, C, reg=self.entR)
        elif transport == "wotpartial":
            T = ot.partial.partial_wasserstein(a, b, C, m=m_eff, nb_dummies=100)
            T[T < 1e-10] = 0
        elif transport == "wotpartialR":
            T = ot.partial.entropic_partial_wasserstein(a, b, C, reg=self.entR, m=m_eff)
            T[T < 1e-10] = 0
        else:
            raise ValueError("Not implemented")
    
        T[T < 1e-5] = 0
        return T

    def _compute_coupling(self, post_a, post_b):
        if self.ot_solver == "hiref":
            return solve_surjection_hiref(post_a, post_b, hierarchy_depth=self.hierarchy_depth, max_Q=self.max_Q, 
                                          max_rank=self.max_rank, verbose=self.verbose, random_state=self.random_state)
        return self._compute_dense_ot(post_a, post_b)

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        if sparse.issparse(T):
            W_ab = prox_a.dot(T)
            W_ba = prox_b.dot(T.transpose())
        else:
            W_ab = sparse.csr_matrix(prox_a.dot(T))
            W_ba = sparse.csr_matrix(prox_b.dot(T.transpose()))
        
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)
            print_mat_stats("Cross-domain B→A (W21)", W_ba)
        
        return sparse.bmat([[(1 - self.mu) * prox_a, self.mu * W_ab], 
                            [self.mu * W_ba, (1 - self.mu) * prox_b]], format="csr")

    def fit(self, x_a, x_b, y_a, y_b):
        """Fits FoSTA alignment across two domains."""
        self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
        self.n = self.n_a + self.n_b
        labels = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)
        self.classes_ = labels

        (self.leaf_pca_a_, self.Q_full_a_, self.W_lab_a_, self.kernel_a, 
         self.idx_lab_a_, self.idx_unl_a_, prox_a) = self._compute_domain_geometry(x_a, y_a, "A")
        
        (self.leaf_pca_b_, self.Q_full_b_, self.W_lab_b_, self.kernel_b, 
         self.idx_lab_b_, self.idx_unl_b_, prox_b) = self._compute_domain_geometry(x_b, y_b, "B")

        post_a = self._get_semantic_vectors(
            self.Q_full_a_,
            self.W_lab_a_,
            y_a,
            labels,
            self.idx_lab_a_,
            t_sem=self.t_sem_a,
        )
        
        post_b = self._get_semantic_vectors(
            self.Q_full_b_,
            self.W_lab_b_,
            y_b,
            labels,
            self.idx_lab_b_,
            t_sem=self.t_sem_b,
        )

        self.T_sparse = self._compute_coupling(post_a, post_b)
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """Fits alignment and computes embedding."""
        self.fit(x_a, x_b, y_a, y_b)
        if self.embedder == "PHATE":
            embedder = PageRankPHATE(n_components=self.n_components, t=self.t, knn_dist="precomputed_affinity",
                                     kernel_symm="+", random_state=self.random_state, verbose=self.verbose,
                                     n_jobs=self.n_jobs, beta=self.beta)
            self.embedding_ = embedder.fit_transform(self.W)
        else:
            self.embedding_ = UMAP(n_components=self.n_components, metric="precomputed", 
                                   random_state=self.random_state).fit_transform(kernel2Dist(self.W.toarray()))
        return self.embedding_

    def get_embeddings(self):
        return self.embedding_