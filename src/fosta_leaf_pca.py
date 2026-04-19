import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import sparse


import graphtools
from forestkernel import ForestKernel

from src.phate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref


class FoSTA_Leaf:
    """
    FoSTA: Forest-guided Semantic Transport Alignment

    Per domain:
        raw data
          -> fit forest on labeled points only
          -> compute full leaf query-map embedding on all points
          -> sparse PCA in leaf space
          -> graphtools graph on Leaf-PCA coordinates

    Semantics / OT:
        Q_full @ Y_lab

    Final embedding:
        within-domain graph affinities + cross-domain propagation through HiRef coupling
          -> PHATE / Spectral / UMAP
    """

    def __init__(
        self,
        mu=0.5,
        kernel_method='gap',
        model_type='rf',
        n_estimators=1000,
        pca_dim=100,
        n_neighbors=5,
        decay=40,
        knn_dist='euclidean',
        t='auto',
        beta=0.7,
        prior_correct=True,
        semantic_norm='l2',
        embedder='PHATE',
        n_components=2,
        verbose=0,
        random_state=None,
        n_jobs=-1,
    ):
        self.mu = mu
        self.kernel_method = kernel_method
        self.model_type = model_type
        self.n_estimators = n_estimators

        self.pca_dim = pca_dim

        self.n_neighbors = n_neighbors
        self.decay = decay
        self.knn_dist = knn_dist

        self.prior_correct = prior_correct
        self.t = t
        self.beta = beta
        self.semantic_norm = semantic_norm
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.kernel_params = {
            'random_state': self.random_state,
            'prediction_type': 'classification',
            'n_estimators': self.n_estimators,
            'kernel_method': self.kernel_method,
            'model_type': self.model_type,
            'force_nonzero_diag': False,
        }

        self.kernel_a = None
        self.kernel_b = None

        self.leaf_pca_a_ = None
        self.leaf_pca_b_ = None

        self.Q_full_a_ = None
        self.Q_full_b_ = None
        self.idx_lab_a_ = None
        self.idx_lab_b_ = None
        self.idx_unl_a_ = None
        self.idx_unl_b_ = None

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

    # ------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _build_graph_from_coords(self, coords):
        G = graphtools.Graph(
            coords,
            n_pca=None,
            knn=self.n_neighbors,
            decay=self.decay,
            distance=self.knn_dist,
            thresh=1e-4,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=bool(self.verbose),
        )
        return G.K

    def _compute_domain_geometry(self, x, y, domain_name="A"):
        """
        Returns
        -------
        coords_full : (N, pca_dim) dense
        Q_full      : sparse csr_matrix, shape (N, L)
        kernel      : fitted ForestKernel
        idx_lab     : labeled indices
        idx_unl     : unlabeled indices
        prox        : within-domain graph affinity on Leaf-PCA coordinates
        """
        y = np.asarray(y).ravel()
        unl_mask = LabelUtils.get_unlabeled_mask(y)
        lab_mask = ~unl_mask

        idx_lab = np.flatnonzero(lab_mask)
        idx_unl = np.flatnonzero(unl_mask)

        x_lab = x[idx_lab]
        y_lab = y[idx_lab]

        self._log(f"\n[Domain {domain_name}] Fitting forest on labeled points...")
        self._log(f"[Domain {domain_name}] labeled: {len(idx_lab)} | unlabeled: {len(idx_unl)}")

        kernel = ForestKernel(**self.kernel_params)
        kernel.fit(x_lab, y_lab)

        self._log(f"[Domain {domain_name}] Computing full leaf query-map embedding...")
        Q_full = kernel.get_query_map(x).tocsr()

        self._log(f"[Domain {domain_name}] Running sparse PCA in leaf space...")
        pca = PCA(
            n_components=self.pca_dim,
            svd_solver="arpack",
            random_state=self.random_state,
        )
        coords_full = pca.fit_transform(Q_full)

        self._log(f"[Domain {domain_name}] Building graph on Leaf-PCA coordinates...")
        prox = self._build_graph_from_coords(coords_full)

        return coords_full, Q_full, kernel, idx_lab, idx_unl, prox

    # ------------------------------------------------------------
    # Semantic projection directly from full query map
    # ------------------------------------------------------------
    def _get_semantic_vectors_from_query_map(
        self,
        Q_full,
        y,
        labels,
        idx_lab,
        eps=1e-12,
    ):
        y = np.asarray(y).ravel()
        n_classes = len(labels)
        n_total = Q_full.shape[0]
        n_lab = len(idx_lab)

        lab2idx = {lab: k for k, lab in enumerate(labels)}
        y_lab = y[idx_lab]

        if y_lab.size == 0:
            return np.zeros((n_total, n_classes), dtype=float)

        try:
            y_lab_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(f"Found label {e} in y that is not in `labels`.")

        counts_c = np.bincount(y_lab_idx, minlength=n_classes).astype(float)
        n_lab_float = float(counts_c.sum())
        class_prior = counts_c / max(n_lab_float, 1.0)
        inv_prior = 1.0 / np.maximum(class_prior, eps)

        Y_lab = np.zeros((n_lab, n_classes), dtype=np.float64)
        Y_lab[np.arange(n_lab), y_lab_idx] = 1.0

        self._log("Projecting onto semantic space...")
        post = Q_full @ (Q_full[idx_lab].T @ Y_lab)

        if self.prior_correct:
            post *= inv_prior[None, :]

        if self.semantic_norm == 'l2':
            post = preprocessing.normalize(post, norm="l2", axis=1)
        elif self.semantic_norm == 'l1':
            post = preprocessing.normalize(post, norm="l1", axis=1)
        else:
            self._log(f"[WARN] Unknown normalization={self.semantic_norm}, skipping normalization.")

        return np.asarray(post, dtype=float)

    # ------------------------------------------------------------
    # Joint affinity
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
        W_ab = prox_a.dot(T)
        W_ba = prox_b.dot(T.transpose())

        W_joint = sparse.bmat(
            [
                [(1 - self.mu) * prox_a, self.mu * W_ab],
                [self.mu * W_ba, (1 - self.mu) * prox_b]
            ],
            format="csr"
        )

        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)
            print_mat_stats("Cross-domain B→A (W21)", W_ba)

        return W_joint

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

        self._log("Computing domain A geometry...")
        (
            self.leaf_pca_a_,
            self.Q_full_a_,
            self.kernel_a,
            self.idx_lab_a_,
            self.idx_unl_a_,
            prox_a,
        ) = self._compute_domain_geometry(x_a, y_a, domain_name="A")

        self._log("Computing domain B geometry...")
        (
            self.leaf_pca_b_,
            self.Q_full_b_,
            self.kernel_b,
            self.idx_lab_b_,
            self.idx_unl_b_,
            prox_b,
        ) = self._compute_domain_geometry(x_b, y_b, domain_name="B")

        self._log("Building semantic vectors for both domains...")
        post_a = self._get_semantic_vectors_from_query_map(
            self.Q_full_a_, y_a, labels, self.idx_lab_a_
        )
        post_b = self._get_semantic_vectors_from_query_map(
            self.Q_full_b_, y_b, labels, self.idx_lab_b_
        )

        self._log("Computing HiRef optimal transport...")
        self.T_sparse = solve_surjection_hiref(
            post_a,
            post_b,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        if self.verbose:
            print_mat_stats("Coupling Matrix", self.T_sparse.tocsr())
            print("=================================\n")

        self._log("Building joint affinity matrix...")
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

        self._log("Model fit complete.")
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)

        self._log("Computing joint embedding...")

        if self.embedder == 'PHATE':
            embedder = PageRankPHATE(
                n_components=self.n_components,
                t=self.t,
                knn_dist='precomputed_affinity',
                kernel_symm='+',
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                beta=self.beta,
            )
            self.embedding_ = embedder.fit_transform(self.W)

        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.embedding_ = embedder.fit_transform(self.W)

        elif self.embedder == 'UMAP':
            DistM = kernel2Dist(self.W.toarray())
            self.embedding_ = UMAP(
                n_components=self.n_components,
                metric='precomputed',
                random_state=self.random_state
            ).fit_transform(DistM)

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

        return self.embedding_

    def get_embeddings(self):
        return self.embedding_

