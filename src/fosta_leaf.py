import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs
from sklearn import preprocessing

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
          -> fit RF-GAP on labeled points only
          -> RF-GAP leaf maps (Q_train, W_train)
          -> asymmetric diffusion maps on labeled points
          -> Nyström extension to unlabeled points
          -> graphtools graph on full DM coordinates

    Semantics / OT:
        Q_full @ (W_train.T @ Y_lab)

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
        dm_dim=100,
        dm_t=1,
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

        self.dm_dim = dm_dim
        self.dm_t = dm_t

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

        self.dm_coords_a_ = None
        self.dm_coords_b_ = None

        self.Q_full_a_ = None
        self.Q_full_b_ = None
        self.W_train_a_ = None
        self.W_train_b_ = None
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

    @staticmethod
    def _make_rfgap_operator(Q, W):
        n = Q.shape[0]

        def matvec(v):
            v = np.asarray(v).reshape(-1)
            return Q @ (W.T @ v)

        return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)

    @staticmethod
    def _canonicalize_eigenvectors(evecs):
        evecs = evecs.copy()
        for j in range(evecs.shape[1]):
            idx = np.argmax(np.abs(evecs[:, j]))
            if np.real(evecs[idx, j]) < 0:
                evecs[:, j] *= -1
        return evecs

    # ------------------------------------------------------------
    # RF-GAP diffusion maps on labeled set
    # ------------------------------------------------------------
    def _compute_dm_from_leaf_maps(self, Q, W, imag_tol=1e-10):
        A = self._make_rfgap_operator(Q, W)

        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(Q.shape[0])

        evals, evecs = eigs(A, k=self.dm_dim + 1, which="LM", v0=v0)

        self._log("\n=== Raw RF-GAP eigenvalues ===")
        if self.verbose:
            for j, lam in enumerate(evals):
                print(f"  eig {j}: {lam}")

        idx0 = np.argmin(np.abs(evals - 1))
        self._log(f"Removing trivial mode closest to 1: eigenvalue = {evals[idx0]}")

        mask = np.ones(len(evals), dtype=bool)
        mask[idx0] = False
        evals = evals[mask]
        evecs = evecs[:, mask]

        order = np.argsort(-np.abs(evals))
        evals = evals[order][:self.dm_dim]
        evecs = evecs[:, order][:, :self.dm_dim]
        evecs = self._canonicalize_eigenvectors(evecs)

        self._log("=== Nontrivial RF-GAP eigenvalues kept ===")
        if self.verbose:
            for j, lam in enumerate(evals):
                is_complex = abs(lam.imag) > imag_tol
                print(f"  mode {j+1}: {lam} | complex={is_complex}")

        coords = evecs * (evals ** self.dm_t)

        max_imag = np.max(np.abs(coords.imag))
        self._log(f"Max imaginary part in labeled DM coords: {max_imag:.3e}")

        return coords, evals, evecs

    @staticmethod
    def _assemble_sparse_query_map(Q_train, Q_unl, idx_lab, idx_unl, n_total):
        Q_train = Q_train.tocoo()
        rows_lab = idx_lab[Q_train.row]

        data_parts = [Q_train.data]
        row_parts = [rows_lab]
        col_parts = [Q_train.col]

        if Q_unl is not None and Q_unl.shape[0] > 0:
            Q_unl = Q_unl.tocoo()
            rows_unl = idx_unl[Q_unl.row]
            data_parts.append(Q_unl.data)
            row_parts.append(rows_unl)
            col_parts.append(Q_unl.col)

        data = np.concatenate(data_parts) if len(data_parts) > 1 else data_parts[0]
        rows = np.concatenate(row_parts) if len(row_parts) > 1 else row_parts[0]
        cols = np.concatenate(col_parts) if len(col_parts) > 1 else col_parts[0]

        return sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_total, Q_train.shape[1]),
        ).tocsr()

    def _build_graph_from_dm(self, coords):
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

    def _compute_domain_geometry(self, x, y, domain_name="A", imag_tol=1e-10):
        """
        Returns
        -------
        coords_full : (N, dm_dim) dense
        Q_full      : sparse csr_matrix, shape (N, L)
        W_train     : sparse csr_matrix, shape (N_lab, L)
        kernel      : fitted ForestKernel
        idx_lab     : labeled indices
        idx_unl     : unlabeled indices
        prox        : within-domain graph affinity on DM coordinates
        """
        y = np.asarray(y).ravel()
        unl_mask = LabelUtils.get_unlabeled_mask(y)
        lab_mask = ~unl_mask

        idx_lab = np.flatnonzero(lab_mask)
        idx_unl = np.flatnonzero(unl_mask)

        x_lab = x[idx_lab]
        y_lab = y[idx_lab]

        self._log(f"\n[Domain {domain_name}] Fitting RF-GAP on labeled points...")
        self._log(f"[Domain {domain_name}] labeled: {len(idx_lab)} | unlabeled: {len(idx_unl)}")

        kernel = ForestKernel(**self.kernel_params)
        kernel.fit(x_lab, y_lab)

        self._log(f"[Domain {domain_name}] Extracting train leaf maps...")
        Q_train = kernel.get_train_query_map().tocsr()
        W_train = kernel.get_reference_map().tocsr()

        self._log(f"[Domain {domain_name}] Computing RF-GAP diffusion coordinates on labeled points...")
        coords_lab, evals, evecs = self._compute_dm_from_leaf_maps(
            Q_train, W_train, imag_tol=imag_tol
        )

        coords_full = np.zeros((x.shape[0], self.dm_dim), dtype=np.float64)
        coords_full[idx_lab] = coords_lab.real

        if len(idx_unl) > 0:
            self._log(f"[Domain {domain_name}] Extending unlabeled points with Nyström...")
            x_unl = x[idx_unl]
            Q_unl = kernel.get_query_map(x_unl).tocsr()

            P_cross = Q_unl @ W_train.T
            coords_unl = (P_cross @ evecs) * (evals ** (self.dm_t - 1))

            max_imag_unl = np.max(np.abs(coords_unl.imag))
            self._log(f"[Domain {domain_name}] Max imaginary part in unlabeled Nyström coords: {max_imag_unl:.3e}")

            coords_full[idx_unl] = coords_unl.real
        else:
            Q_unl = None

        self._log(f"[Domain {domain_name}] Assembling full sparse query map...")
        Q_full = self._assemble_sparse_query_map(
            Q_train=Q_train,
            Q_unl=Q_unl,
            idx_lab=idx_lab,
            idx_unl=idx_unl,
            n_total=x.shape[0],
        )

        self._log(f"[Domain {domain_name}] Building graph on DM coordinates...")
        prox = self._build_graph_from_dm(coords_full)

        return coords_full, Q_full, W_train, kernel, idx_lab, idx_unl, prox

    # ------------------------------------------------------------
    # Semantic projection directly from leaf maps
    # ------------------------------------------------------------
    def _get_semantic_vectors_from_leaf_maps(
        self,
        Q_full,
        W_train,
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
        S = W_train.T @ Y_lab
        post = Q_full @ S

        if self.prior_correct:
            post *= inv_prior[None, :]

        if self.semantic_norm == 'l2':
            post = preprocessing.normalize(post, norm="l2", axis=1)
        elif self.semantic_norm == 'l1':
            post = preprocessing.normalize(post, norm="l1", axis=1)
        else:
            self._log(f"[WARN] Unknown normalization={self.semantic_norm}, skipping normalization.")

        return post

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
            self.dm_coords_a_,
            self.Q_full_a_,
            self.W_train_a_,
            self.kernel_a,
            self.idx_lab_a_,
            self.idx_unl_a_,
            prox_a,
        ) = self._compute_domain_geometry(x_a, y_a, domain_name="A")

        self._log("Computing domain B geometry...")
        (
            self.dm_coords_b_,
            self.Q_full_b_,
            self.W_train_b_,
            self.kernel_b,
            self.idx_lab_b_,
            self.idx_unl_b_,
            prox_b,
        ) = self._compute_domain_geometry(x_b, y_b, domain_name="B")

        self._log("Building semantic vectors for both domains...")
        post_a = self._get_semantic_vectors_from_leaf_maps(
            self.Q_full_a_, self.W_train_a_, y_a, labels, self.idx_lab_a_
        )
        post_b = self._get_semantic_vectors_from_leaf_maps(
            self.Q_full_b_, self.W_train_b_, y_b, labels, self.idx_lab_b_
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