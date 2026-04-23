import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from scipy.spatial.distance import cdist
import ot

import graphtools
from forestkernel import ForestKernel

from src.phate import PageRankPHATE
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref


class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment

    Per domain:
        raw data
          -> fit symmetric forest kernel on labeled points only
          -> build full leaf map from:
                Q_lab = get_train_query_map() on labeled train points
                Q_unl = get_query_map() on unlabeled points
          -> TruncatedSVD in leaf space using Q_full
          -> graphtools graph on reduced leaf coordinates

    Semantics / OT:
        Q_full @ (Q_lab.T @ Y_lab)

    Final embedding:
        within-domain graph affinities + cross-domain propagation through coupling
          -> PHATE / UMAP
    """

    def __init__(
        self,
        mu=0.5,
        kernel_method="original",  # oob or original
        model_type="rf",
        n_estimators=500,
        bootstrap=True,
        n_svd=100,
        n_neighbors=5,  # try maybe 10, 30 
        decay=40,
        knn_dist="euclidean",
        t="auto",  # t=2 or 'auto'
        beta=0.9,
        prior_correct=True,
        semantic_norm="l2",
        embedder="PHATE",
        n_components=2,
        ot_solver="hiref",
        hierarchy_depth = 6,
        max_Q = int(2**10),
        max_rank = 16,
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
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.kernel_params = {
            "random_state": random_state,
            "prediction_type": "classification",
            "n_estimators": self.n_estimators,
            "kernel_method": self.kernel_method,
            "model_type": self.model_type,
            "bootstrap": self.bootstrap,
        }

        # Leaf preprocessing parameters
        self.n_svd = n_svd

        # Graph construction parameters
        self.n_neighbors = n_neighbors
        self.decay = decay
        self.knn_dist = knn_dist

        self.prior_correct = prior_correct
        self.semantic_norm = semantic_norm

        # OT parameters
        self.ot_solver = ot_solver
        # HiRef
        self.hierarchy_depth = hierarchy_depth
        self.max_Q = max_Q
        self.max_rank = max_rank
        # Dense OT
        self.entR = entR
        self.m = m
        self.distance = distance

        # Affinity balancing parameter
        self.mu = mu

        # Final embedding parameters
        self.t = t
        self.beta = beta
        self.embedder = embedder
        self.n_components = n_components


        self.kernel_a = None
        self.kernel_b = None

        self.leaf_svd_a_ = None
        self.leaf_svd_b_ = None

        self.Q_full_a_ = None
        self.Q_full_b_ = None
        self.Q_lab_a_ = None
        self.Q_lab_b_ = None

        self.idx_lab_a_ = None
        self.idx_lab_b_ = None
        self.idx_unl_a_ = None
        self.idx_unl_b_ = None

        self.T_sparse = None
        self.Distances12 = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _assemble_sparse_query_map(Q_lab, Q_unl, idx_lab, idx_unl, n_total):
        Q_lab = Q_lab.tocoo()
        row_lab = idx_lab[Q_lab.row]

        data_parts = [Q_lab.data]
        row_parts = [row_lab]
        col_parts = [Q_lab.col]

        if Q_unl is not None and Q_unl.shape[0] > 0:
            Q_unl = Q_unl.tocoo()
            row_unl = idx_unl[Q_unl.row]

            data_parts.append(Q_unl.data)
            row_parts.append(row_unl)
            col_parts.append(Q_unl.col)

        data = np.concatenate(data_parts) if len(data_parts) > 1 else data_parts[0]
        rows = np.concatenate(row_parts) if len(row_parts) > 1 else row_parts[0]
        cols = np.concatenate(col_parts) if len(col_parts) > 1 else col_parts[0]

        return sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_total, Q_lab.shape[1]),
        ).tocsr()

    def _reduce_leaf_coords(self, Q_full, domain_name="A"):
        self._log(f"[Domain {domain_name}] Reducing leaf coordinates with TruncatedSVD...")

        reducer = TruncatedSVD(
            n_components=min(self.n_estimators, self.n_svd),
            algorithm="arpack",
            random_state=self.random_state,
        )
        coords_full = reducer.fit_transform(Q_full)

        return coords_full

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

        self._log(f"[Domain {domain_name}] Extracting labeled query map...")
        Q_lab = kernel.get_train_query_map().tocsr()

        if len(idx_unl) > 0:
            self._log(f"[Domain {domain_name}] Computing unlabeled query map...")
            x_unl = x[idx_unl]
            Q_unl = kernel.get_query_map(x_unl).tocsr()
        else:
            Q_unl = None

        self._log(f"[Domain {domain_name}] Assembling full query map...")
        Q_full = self._assemble_sparse_query_map(
            Q_lab=Q_lab,
            Q_unl=Q_unl,
            idx_lab=idx_lab,
            idx_unl=idx_unl,
            n_total=x.shape[0],
        )

        coords_full = self._reduce_leaf_coords(Q_full, domain_name=domain_name)

        self._log(f"[Domain {domain_name}] Building graph on reduced leaf coordinates...")
        prox = self._build_graph_from_coords(coords_full)

        return coords_full, Q_full, Q_lab, kernel, idx_lab, idx_unl, prox

    def _get_semantic_vectors(
        self,
        Q_full,
        Q_lab,
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
        S = Q_lab.T @ Y_lab
        post = Q_full @ S
        post = np.asarray(post, dtype=float)

        if self.prior_correct:
            post *= inv_prior[None, :]

        if self.semantic_norm == "l2":
            post = preprocessing.normalize(post, norm="l2", axis=1)
        elif self.semantic_norm == "l1":
            post = preprocessing.normalize(post, norm="l1", axis=1)
        else:
            self._log(f"[WARN] Unknown normalization={self.semantic_norm}, skipping normalization.")

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
            self._log("Computing HiRef optimal transport...")
            return solve_surjection_hiref(
                post_a,
                post_b,
                hierarchy_depth=self.hierarchy_depth,
                max_Q=self.max_Q,
                max_rank=self.max_rank,
                verbose=self.verbose,
                random_state=self.random_state,
            )

        if self.ot_solver == "dense":
            return self._compute_dense_ot(post_a, post_b)

        raise ValueError(f"Unknown ot_solver={self.ot_solver}")

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        if sparse.issparse(T):
            W_ab = prox_a.dot(T)
            W_ba = prox_b.dot(T.transpose())
        else:
            W_ab = sparse.csr_matrix(prox_a.dot(T))
            W_ba = sparse.csr_matrix(prox_b.dot(T.transpose()))

        W_joint = sparse.bmat(
            [
                [(1 - self.mu) * prox_a, self.mu * W_ab],
                [self.mu * W_ba, (1 - self.mu) * prox_b],
            ],
            format="csr",
        )

        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)
            print_mat_stats("Cross-domain B→A (W21)", W_ba)

        return W_joint

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
            self.leaf_svd_a_,
            self.Q_full_a_,
            self.Q_lab_a_,
            self.kernel_a,
            self.idx_lab_a_,
            self.idx_unl_a_,
            prox_a,
        ) = self._compute_domain_geometry(x_a, y_a, domain_name="A")

        self._log("Computing domain B geometry...")
        (
            self.leaf_svd_b_,
            self.Q_full_b_,
            self.Q_lab_b_,
            self.kernel_b,
            self.idx_lab_b_,
            self.idx_unl_b_,
            prox_b,
        ) = self._compute_domain_geometry(x_b, y_b, domain_name="B")

        self._log("Building semantic vectors for both domains...")
        post_a = self._get_semantic_vectors(
            self.Q_full_a_,
            self.Q_lab_a_,
            y_a,
            labels,
            self.idx_lab_a_,
        )
        post_b = self._get_semantic_vectors(
            self.Q_full_b_,
            self.Q_lab_b_,
            y_b,
            labels,
            self.idx_lab_b_,
        )

        self.T_sparse = self._compute_coupling(post_a, post_b)

        if self.verbose:
            if sparse.issparse(self.T_sparse):
                print_mat_stats("Coupling Matrix", self.T_sparse.tocsr())
            else:
                print_mat_stats("Coupling Matrix", sparse.csr_matrix(self.T_sparse))
            print("=================================\n")

        self._log("Building joint affinity matrix...")
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

        self._log("Model fit complete.")
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)

        self._log("Computing joint embedding...")

        if self.embedder == "PHATE":
            embedder = PageRankPHATE(
                n_components=self.n_components,
                t=self.t,
                knn_dist="precomputed_affinity",
                kernel_symm="+",
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                beta=self.beta,
            )
            self.embedding_ = embedder.fit_transform(self.W)

        elif self.embedder == "UMAP":
            DistM = kernel2Dist(self.W.toarray())
            self.embedding_ = UMAP(
                n_components=self.n_components,
                metric="precomputed",
                random_state=self.random_state,
            ).fit_transform(DistM)

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

        return self.embedding_

    def get_embeddings(self):
        return self.embedding_