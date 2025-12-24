import numpy as np
from scipy import sparse
from sklearn import preprocessing
import math

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
from .hiref import HiRef_fast_exact as HiRef
# from .hiref import rank_annealing # Not needed, we force binary schedule

class RFMALI(object):
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 embedder='spectral',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.rfgap_params = {
            'random_state': random_state,
            'prediction_type': 'classification', 
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'max_normalize': True,
            'verbose': 0,
            'n_jobs': -1,
        }

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

        # optional debug attrs
        self._dpt_M_a = None
        self._dpt_M_b = None
        self._clusters_a = None
        self._clusters_b = None

    # ------------------------------------------------------------
    # Posterior builders
    # ------------------------------------------------------------
    def _get_semantic_vectors(
        self,
        W,                # (N, N) prox OR (N, M) landmark weights
        y,                # (N,) labels (may include -1/NaN)
        labels,           # canonical labels (same across domains)
        clusters=None,    # None OR (N,) landmark id per point in [0..M-1]
        eps=1e-12,
        prior_correct=True
    ):
        """
        Build C-dim semantic vectors (posteriors).
        """
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)
        N = W.shape[0]
        C = len(labels)
        lab2idx = {lab: k for k, lab in enumerate(labels)}

        # Case 1: Full proximity (N x N)
        if clusters is None:
            post = np.zeros((N, C), dtype=float)

            for lab, k in lab2idx.items():
                class_mask = (~mask_unl) & (y == lab)
                cnt = int(class_mask.sum())
                if cnt == 0:
                    continue

                if sparse.issparse(W):
                    vec = W[:, class_mask].sum(axis=1).A.ravel()
                else:
                    vec = W[:, class_mask].sum(axis=1)

                post[:, k] = vec / cnt

            # l2 normalize for cosine distance use in HiRef
            post = preprocessing.normalize(post, norm="l2", axis=1) / np.sqrt(2)
            return post

        # Case 2: Landmark DPT weights (N x M)
        clusters = np.asarray(clusters).ravel()
        if clusters.shape[0] != N:
            raise ValueError(f"clusters must have shape (N,), got {clusters.shape} with N={N}")

        M = W.shape[1]
        y_lab = y[~mask_unl]
        cl_lab = clusters[~mask_unl]

        if y_lab.size == 0:
            raise ValueError("No labeled samples in y. Cannot build semantic vectors.")

        try:
            y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(f"Found label {e} in y that is not in `labels`.")

        counts = np.zeros((M, C), dtype=float)
        np.add.at(counts, (cl_lab, y_idx), 1.0)

        row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        Q = counts / row_sums 

        post = W.dot(Q) if sparse.issparse(W) else (W @ Q)

        if prior_correct:
            class_prior = np.bincount(y_idx, minlength=C).astype(float)
            class_prior /= max(class_prior.sum(), 1.0)
            post /= np.maximum(class_prior[None, :], eps)

        post = preprocessing.normalize(post, norm="l2", axis=1)
        return post

    # ------------------------------------------------------------
    # DPT / Landmark diffusion machinery
    # ------------------------------------------------------------
    def _get_diffusion_operators(self, K, random_state=None, verbose=True, **graph_kwargs):
        n_landmark = 2000
        G = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark if n_landmark < K.shape[0] else None,
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

    def compute_dpt(self, P):
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

        print("Fitting RFGAP on Domain A...")
        self.rfgap_a = RFGAP(**self.rfgap_params)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities()

        print("Fitting RFGAP on Domain B...")
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()

        print("Building C-dim vectors...")
        if not self.dpt:
            post_a = self._get_semantic_vectors(prox_a, y_a, labels, clusters=None)
            post_b = self._get_semantic_vectors(prox_b, y_b, labels, clusters=None)
        else:
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(prox_a, random_state=self.random_state)
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(prox_b, random_state=self.random_state)
            M_a = self.compute_dpt(P_MM_a)
            M_b = self.compute_dpt(P_MM_b)
            self._dpt_M_a, self._dpt_M_b = M_a, M_b
            self._clusters_a, self._clusters_b = clusters_a, clusters_b
            
            trans_a = P_NM_a.dot(M_a)
            trans_b = P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b)

        # ------------------------------------------------------------
        # OT with Scaling Fix
        # ------------------------------------------------------------
        print("Computing Exact Monge Coupling...")
        n_a, dim_a = post_a.shape
        n_b, dim_b = post_b.shape
        max_n = max(n_a, n_b)
        pad_exp = math.ceil(math.log2(max_n))
        n_pad = 2 ** pad_exp
        
        # --- FIX 1: LOWER DUMMY VALUE ---
        # Real data is normalized (norm=1). Max sq-dist = 4.
        # Dummy value 3.0 gives sq-dist = 9.0 (safely larger, but numerically stable).
        DUMMY_VAL = 1.0 
        
        def _pad_matrix(M_in):
            curr_n, curr_c = M_in.shape
            M_padded = np.zeros((n_pad, curr_c + 1), dtype=M_in.dtype)
            M_padded[:curr_n, :curr_c] = M_in
            if curr_n < n_pad:
                M_padded[curr_n:, -1] = DUMMY_VAL
            return M_padded

        post_a_pad = _pad_matrix(post_a)
        post_b_pad = _pad_matrix(post_b)

        rank_schedule = [2] * pad_exp

        # --- FIX 2: TUNE HIREF ---
        # rescale_cost=True helps normalize the cost matrix internally to mean=1
        # This makes the choice of gamma much more robust.
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            post_a_pad, post_b_pad,
            rank_schedule=rank_schedule,
            base_rank=1,
            # iters_per_level=100,     # Increased from 60 for better convergence
            rescale_cost=True,       # CRITICAL: Auto-scales the cost matrix
            return_coupling=True
        )

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        data = np.asarray(data, dtype=np.float64)

        # 5. Filter / Slice back to original dimensions
        # Keep only interactions where row < n_a AND col < n_b
        valid_mask = (rows < n_a) & (cols < n_b)
        
        rows_clean = rows[valid_mask]
        cols_clean = cols[valid_mask]
        data_clean = data[valid_mask]

        self.T_sparse = sparse.coo_matrix(
            (data_clean, (rows_clean, cols_clean)),
            shape=(n_a, n_b)
        ).tocsr()

        # ---------- DIAGNOSTICS ----------
        #  would be useful here in a notebook, 
        # but text diagnostics suffice for the class.
        T = self.T_sparse
        row_nnz = np.diff(T.indptr)
        print("\nMONGE MAP DIAGNOSTICS")
        print("---------------------")
        print(f"Original sizes: {n_a} x {n_b}")
        print(f"Padded size used: {n_pad}")
        print(f"Total Matches (before filtering): {len(rows)}")
        print(f"Total Matches (after filtering): {len(rows_clean)}")
        
        # Check for violations
        print(f"Rows with != 1 match: {np.sum(row_nnz != 1)} (Expected: {max(0, n_a - n_b)} unassigned if n_a > n_b)")
        
        # Normalize: Binary permutation shouldn't need norm, but safeguards for float errors
        self.T_sparse.data[:] = 1.0 

        print("Building joint affinity matrix...")
        W_ab = (prox_a.dot(self.T_sparse) + self.T_sparse.dot(prox_b))
        W_ba = W_ab.T

        self.W = sparse.bmat(
            [
                [self.mu * prox_a, (1 - self.mu) * W_ab],
                [(1 - self.mu) * W_ba, self.mu * prox_b]
            ],
            format="csr"
        )

        print("Model fit complete.")
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        print("Computing joint embedding...")

        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                random_state=self.random_state,
                verbose=self.verbose,
                kernel_symm=None,
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