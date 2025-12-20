import numpy as np
from scipy import sparse
from sklearn import preprocessing

# Graph tools
import graphtools

# RF-GAP
from rfgap import RFGAP

# Embedders
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

from utils.utils import kernel2Dist

# OT machinery
from .hiref import HiRef_fast as HiRef
from .hiref import rank_annealing

import sys


class RFMALI(object):
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 embedder='phate',
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
            'prediction_type': 'classification',  # force classification mode, more robust to y dtype
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'max_normalize': True,  # ensures consistent scaling across datasets with unequal sizes
            'verbose': 0,
            'n_jobs': -1,
        }

        self.T_sparse = None
        self.W_combined = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None

        # optional debug attrs
        self._dpt_M_a = None
        self._dpt_M_b = None
        self._clusters_a = None
        self._clusters_b = None

    # ------------------------------------------------------------
    # Missing-label handling
    # ------------------------------------------------------------
    def _unlabeled_mask(self, y: np.ndarray) -> np.ndarray:
        """True for NaN/None OR sentinel -1. Works for int/float/object."""
        y = np.asarray(y).ravel()
        mask = np.zeros(y.shape[0], dtype=bool)

        # sentinel -1
        try:
            mask |= (y == -1)
        except Exception:
            pass

        # NaN/None
        if np.issubdtype(y.dtype, np.number):
            mask |= np.isnan(y)
        else:
            for i, v in enumerate(y):
                if v is None:
                    mask[i] = True
                else:
                    try:
                        if v != v:  # NaN check for object
                            mask[i] = True
                    except Exception:
                        pass
        return mask

    def _labeled_unique(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        mask_unl = self._unlabeled_mask(y)
        return np.unique(y[~mask_unl])

    def _assert_same_label_space(self, y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
        labels_a = self._labeled_unique(y_a)
        labels_b = self._labeled_unique(y_b)

        set_a, set_b = set(labels_a.tolist()), set(labels_b.tolist())
        if set_a != set_b:
            only_a = sorted(set_a - set_b)
            only_b = sorted(set_b - set_a)
            raise ValueError(
                "Domain label mismatch (shared semantic space violated).\n"
                f"  Labels only in domain A: {only_a}\n"
                f"  Labels only in domain B: {only_b}\n"
                f"  Labels in A: {sorted(set_a)}\n"
                f"  Labels in B: {sorted(set_b)}"
            )

        labels = np.array(sorted(set_a), dtype=labels_a.dtype)
        self.classes_ = labels
        return labels

    # ------------------------------------------------------------
    # Posterior builders
    # ------------------------------------------------------------
    def _get_semantic_vectors(
        self,
        W,                 # (N, N) prox OR (N, M) landmark weights
        y,                 # (N,) labels (may include -1/NaN)
        labels,            # canonical labels (same across domains)
        clusters=None,     # None OR (N,) landmark id per point in [0..M-1]
        eps=1e-12,
        prior_correct=True
    ):
        """
        Build C-dim semantic vectors (posteriors) from:
          - Full proximities W = prox (N x N) with clusters=None  -> classwise KDE on columns
          - Landmark weights W = trans (N x M) with clusters      -> landmark mixture + optional prior correction
    
        Unlabeled samples in y (-1/NaN/None) are ignored when estimating class statistics.
        """
        y = np.asarray(y).ravel()
        mask_unl = self._unlabeled_mask(y)
        N = W.shape[0]
        C = len(labels)
    
        # Map labels -> [0..C-1]
        lab2idx = {lab: k for k, lab in enumerate(labels)}
    
        # ------------------------------------------------------------
        # Case 1: Full proximity (N x N): classwise KDE sum over columns
        # ------------------------------------------------------------
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
    
            # normalize to form distribution (safe)
            post = preprocessing.normalize(post, norm="l1", axis=1)
    
            return post
    
        # ------------------------------------------------------------
        # Case 2: Landmark DPT weights (N x M): build landmark class mixture Q (M x C)
        # ------------------------------------------------------------
        clusters = np.asarray(clusters).ravel()
        if clusters.shape[0] != N:
            raise ValueError(f"clusters must have shape (N,), got {clusters.shape} with N={N}")
    
        M = W.shape[1]
        if clusters.min() < 0 or clusters.max() >= M:
            raise ValueError(f"clusters out of bounds: [{clusters.min()}, {clusters.max()}] vs M={M}")
    
        # use only labeled points to estimate Q and prior
        y_lab = y[~mask_unl]
        cl_lab = clusters[~mask_unl]
    
        # if *no* labeled data, you can't estimate semantics
        if y_lab.size == 0:
            raise ValueError("No labeled samples in y (after masking -1/NaN). Cannot build semantic vectors.")
    
        # label indices for labeled points
        try:
            y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Found label {e} in y that is not in `labels`. "
                "Make sure you call _assert_same_label_space() and pass the returned labels."
            )
    
        # landmark class counts: counts[m, c]
        counts = np.zeros((M, C), dtype=float)
        np.add.at(counts, (cl_lab, y_idx), 1.0)
    
        # Q[m,c] = p(c | landmark m)
        row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        Q = counts / row_sums  # (M, C)
    
        # point posteriors: post = W @ Q
        post = W.dot(Q) if sparse.issparse(W) else (W @ Q)  # (N, C)
    
        # prior correction
        if prior_correct:
            class_prior = np.bincount(y_idx, minlength=C).astype(float)
            class_prior /= max(class_prior.sum(), 1.0)
            post /= np.maximum(class_prior[None, :], eps)
    
        # normalize to form distribution
        post = preprocessing.normalize(post, norm="l1", axis=1)

        return post

    # ------------------------------------------------------------
    # DPT / Landmark diffusion machinery
    # ------------------------------------------------------------
    def _get_diffusion_operators(self, K, random_state=None, verbose=True, **graph_kwargs):
        """
        Returns P_NM (N x M), P_MM (M x M), clusters (N,)
        using graphtools LandmarkGraph when possible.
        """
        n_landmark=200
        G = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark if n_landmark < K.shape[0] else None,
            kernel_symm=None,  # already symmetric in our RFGAP use
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

    def compute_dpt(self, P):
        """
        DPT-like aggregated transition matrix: (I - (P - 1 phi^T))^{-1} - I
        with min-max normalization (as in your old code).
        """
        n = P.shape[0]
        I = np.eye(n)
        ones = np.ones(n, dtype=float)

        # stationary distribution (left eigenvector of P for eigenvalue 1)
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

        Mmat = preprocessing.MinMaxScaler().fit_transform(Mmat.T).T  # Min-Max row-wise to ensure positivity before l1 normalization in _get_semantic_vectors

        return Mmat

    # ------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------
    def fit(self, x_a, x_b, y_a, y_b):
        self.n = x_a.shape[0]
        y_a = np.asarray(y_a).ravel()
        y_b = np.asarray(y_b).ravel()

        # enforce shared label space (ignoring unlabeled)
        labels = self._assert_same_label_space(y_a, y_b)

        # IMPORTANT: RFGAP is classification here; ensure no NaNs are passed to sklearn
        y_a_fit = y_a.copy()
        y_b_fit = y_b.copy()
        y_a_fit[self._unlabeled_mask(y_a_fit)] = -1
        y_b_fit[self._unlabeled_mask(y_b_fit)] = -1

        print("Fitting RFGAP on Domain A...")
        self.rfgap_a = RFGAP(**self.rfgap_params)
        self.rfgap_a.fit(x_a, y_a_fit)
        prox_a = self.rfgap_a.get_proximities()

        print("Fitting RFGAP on Domain B...")
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b_fit)
        prox_b = self.rfgap_b.get_proximities()

        print("Building C-dim vectors...")
        if not self.dpt:
            # ---- Simply use RF-GAP affinities (no diffusion)----
            post_a = self._get_semantic_vectors(prox_a, y_a, labels, clusters=None)
            post_b = self._get_semantic_vectors(prox_b, y_b, labels, clusters=None)
        else:
            # ---- DPT behavior: LandmarkGraph -> DPT (multi-step aggregation) -> balanced posteriors ----
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(prox_a, random_state=self.random_state, verbose=True)
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(prox_b, random_state=self.random_state, verbose=True)

            M_a = self.compute_dpt(P_MM_a)
            M_b = self.compute_dpt(P_MM_b)

            self._dpt_M_a, self._dpt_M_b = M_a, M_b
            self._clusters_a, self._clusters_b = clusters_a, clusters_b

            trans_a = P_NM_a.dot(M_a)  # (N, M)
            trans_b = P_NM_b.dot(M_b)  # (N, M)

            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b)

        print("Computing Optimal Transport...")
        rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n)
        frontier = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule)

        row_idx = np.array([int(p[0][0]) for p in frontier], dtype=np.int64)
        col_idx = np.array([int(p[1][0]) for p in frontier], dtype=np.int64)
        data = np.ones_like(row_idx, dtype=float)
        self.T_sparse = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.n, self.n))

        print("Building joint affinity matrix...")
        W_ab = (prox_a.dot(self.T_sparse) + self.T_sparse.dot(prox_b)) / 2
        W_ba = W_ab.T

        self.W_combined = sparse.bmat(
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

        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=-1,
                beta=0.5,
            )
            self.embedding_ = phate_op.fit_transform(self.W_combined)
            return self.embedding_

        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.embedding_ = embedder.fit_transform(self.W_combined)
            return self.embedding_

        elif self.embedder == 'UMAP':
            DistM = kernel2Dist(self.W_combined.toarray())
            self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', random_state=self.random_state).fit_transform(DistM)
            return self.embedding_

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

    def get_embeddings(self):
        return self.embedding_