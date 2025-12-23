import numpy as np
from scipy import sparse
from sklearn import preprocessing
import pandas as pd

# Graph tools for DPT
import graphtools

# RF-GAP
from rfgap import RFGAP

# Embedders
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

# Utils
from utils.utils import kernel2Dist
from utils.labels import LabelUtils

# OT solver
from .hiref import HiRef_fast_sparse_out as HiRef
from .hiref import rank_annealing

import sys


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
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None

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
        mask_unl = LabelUtils.get_unlabeled_mask(y)
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

        labels = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)  # validate and get shared labels
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
        (rows, cols, data), _ = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule, return_coupling=True)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        data = np.asarray(data, dtype=np.float64)
        self.T_sparse = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(self.n, self.n)
        ).tocsr()

        # rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n)
        # T_dense = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule, return_coupling=True, dense_coupling=True)
        # T_dense = np.asarray(T_dense)

        # ---------- DENSE ----------
        # T = T_dense
        # n = T.shape[0]
        
        # row_nnz = np.count_nonzero(T, axis=1)
        # col_nnz = np.count_nonzero(T, axis=0)
        
        # row_sums = T.sum(axis=1)
        # col_sums = T.sum(axis=0)
        
        # print("\nDENSE COUPLING")
        # print("--------------")
        # print("Empty rows:", np.sum(row_nnz == 0), "/", n)
        # print("Empty cols:", np.sum(col_nnz == 0), "/", n)
        
        # print("Rows with >1 nonzero:", np.sum(row_nnz > 1), "/", n)
        # print("Cols with >1 nonzero:", np.sum(col_nnz > 1), "/", n)
        
        # vals = T[T != 0]
        # print("Unique nonzero values:", np.unique(vals))
        # print("All nonzero == 1:", np.all(vals == 1))
        
        # print("Row sums: min =", row_sums.min(), "max =", row_sums.max())
        # print("Col sums: min =", col_sums.min(), "max =", col_sums.max())
        # print("Total mass:", T.sum())


        # ---------- SPARSE ----------
        T = self.T_sparse.tocsr()
        n = T.shape[0]
        
        row_nnz = np.diff(T.indptr)
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        
        Tc = T.tocsc()
        col_nnz = np.diff(Tc.indptr)
        col_sums = np.asarray(Tc.sum(axis=0)).ravel()
        
        print("\nSPARSE COUPLING")
        print("---------------")
        print("Empty rows:", np.sum(row_nnz == 0), "/", n)
        print("Empty cols:", np.sum(col_nnz == 0), "/", n)
        
        print("Rows with >1 nonzero:", np.sum(row_nnz > 1), "/", n)
        print("Cols with >1 nonzero:", np.sum(col_nnz > 1), "/", n)
        
        print("Unique nonzero values:", np.unique(T.data))
        print("All nonzero == 1:", np.all(T.data == 1))
        
        print("Row sums: min =", row_sums.min(), "max =", row_sums.max())
        print("Col sums: min =", col_sums.min(), "max =", col_sums.max())
        print("Total mass:", T.sum())


        ## COMPARISON
        # diff = self.T_sparse - T_dense
        # print("max |T_sparse - T_dense| =", np.abs(diff).max())
        # print(
        #     "exact equality:",
        #     np.allclose(self.T_sparse.toarray(), T_dense, atol=0.0)
        # )


        # Row normalize
        self.T_sparse = preprocessing.normalize(self.T_sparse, norm="l1", axis=1)

        print("Building joint affinity matrix...")
        W_ab = (prox_a.dot(self.T_sparse) + self.T_sparse.dot(prox_b)) / 2
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
            self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', random_state=self.random_state).fit_transform(DistM)
            return self.embedding_
        
        elif self.embedder == 'barycentric':
            raise NotImplementedError("Barycentric embedding not implemented yet in RFMALI.")

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

    def get_embeddings(self):
        return self.embedding_