import numpy as np
from sklearn import preprocessing
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, svds
import ot

import graphtools
from .rfgap.forestkernel import ForestKernel

from src.phate import PageRankPHATE
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref

import sys


class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment
    Exact match to reference class logic, using W_lab and Row-Normalization.
    """

    def __init__(
        self,
        mu=1,
        old_version=True,
        kernel_method="gap",
        force_nonzero_diag=True,
        force_symmetric=True,
        normalize_diagonal=True,
        model_type="rf",
        n_estimators=1000,
        n_pca=100,
        n_neighbors=5,
        decay=40,
        knn_dist="euclidean",
        t="auto",
        beta=0.7,
        prior_correct=True,
        l2_normalize=True,
        embedder="PHATE",
        n_components=2,
        ot_solver="hiref",
        
        entR=0,
        m=1,
        distance="cosine",
        verbose=1,
        random_state=None,
        n_jobs=-1,
    ):
        self.old_version = old_version

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
        self.force_nonzero_diag = force_nonzero_diag
        self.force_symmetric = force_symmetric
        self.normalize_diagonal = normalize_diagonal
        self.kernel_params = {
            "random_state": random_state,
            "prediction_type": "classification",
            "oob_score": True,
            "n_estimators": self.n_estimators,
            "kernel_method": self.kernel_method,
            "force_nonzero_diag": self.force_nonzero_diag,
            "model_type": self.model_type,
            "bootstrap": True,
        }

        self.n_pca = n_pca
        self.n_neighbors = n_neighbors
        self.decay = decay
        self.knn_dist = knn_dist
        self.l2_normalize = l2_normalize
        self.prior_correct = prior_correct

        self.ot_solver = ot_solver

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
        self.prox_a = self.prox_b = None
        self.leaf_pca_a_ = self.leaf_pca_b_ = None
        self.Q_full_a_ = self.Q_full_b_ = None
        self.W_lab_a_ = self.W_lab_b_ = None  # Using W_lab
        self.idx_lab_a_ = self.idx_lab_b_ = None
        self.idx_unl_a_ = self.idx_unl_b_ = None
        self.T_sparse = self.Distances12 = self.W = self.embedding_ = None
        self.classes_ = self.n = self.n_a = self.n_b = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

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

    def _build_graph_from_coords(self, coords, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        G = graphtools.Graph(
            coords,
            n_pca=None,
            knn=n_neighbors,
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
        idx_lab = np.flatnonzero(~unl_mask)
        idx_unl = np.flatnonzero(unl_mask)

        self._log(f"\n[Domain {domain_name}] Fitting forest on labeled points...")
        kernel = ForestKernel(**self.kernel_params)

        # ------------------------------------------------------------
        # Always fit labeled-only model for semantic pipeline
        # ------------------------------------------------------------
        kernel.fit(x, y, idx_unlabeled=idx_unl)

        Q_full = kernel.get_train_query_map().tocsr()
        W_full = kernel.get_reference_map().tocsr()
        W_lab = W_full[idx_lab].tocsr()

        # get oob classification error for sanity check
        oob_error = kernel.forest_.oob_score_
        self._log(f"[Domain {domain_name}] OOB classification acc.: {oob_error:.4f}")

        # ------------------------------------------------------------
        # Geometry branch
        # ------------------------------------------------------------
        if self.old_version:
            # ===== OLD FoSTA =====
            self._log(f"[Domain {domain_name}] Using OLD FoSTA kernel (full NxN)...")

            prox = kernel.get_kernel(
                normalize_diagonal=self.normalize_diagonal,
                force_symmetric=self.force_symmetric,
            ).tocsr()
            prox.data = np.maximum(prox.data, 0)

            print(
                f"Shapes: Q_full={Q_full.shape}, "
                f"W_lab={W_lab.shape}, "
                f"W_full={W_full.shape}, "
                f"prox={prox.shape}"
            )

            coords_full = None  # no PCA coords

        else:
            # ===== NEW METHOD =====
            self._log(f"[Domain {domain_name}] Matrix-free PCA on P = Q_full W_lab^T...")

            coords_full, S, _ = self._svd_p_full(
                Q_full,
                W_lab,
                n_components=self.n_pca,
                random_state=self.random_state,
            )

            if self.n_neighbors == "auto":
                n_neighbors = int(np.ceil(np.sqrt(Q_full.shape[0])))
                n_neighbors = max(5, min(n_neighbors, 30, Q_full.shape[0] - 1))
                self._log(f"Auto-setting n_neighbors={n_neighbors} based on dataset size.")
            else:
                n_neighbors = self.n_neighbors

            prox = self._build_graph_from_coords(
                coords_full,
                n_neighbors=n_neighbors,
            )

        return coords_full, Q_full, W_lab, kernel, idx_lab, idx_unl, prox

    def _get_semantic_vectors(self, Q_full, W_lab, y, labels, idx_lab):
        y = np.asarray(y).ravel()
        n_classes = len(labels)

        lab2idx = {lab: k for k, lab in enumerate(labels)}
        y_lab = y[idx_lab]

        if y_lab.size == 0:
            return np.zeros((Q_full.shape[0], n_classes))

        y_lab_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)

        Y_lab = np.zeros((len(idx_lab), n_classes), dtype=np.float64)
        Y_lab[np.arange(len(idx_lab)), y_lab_idx] = 1.0

        self._log("Projecting onto semantic space...")

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

        if self.l2_normalize:
            post = preprocessing.normalize(post, norm="l2", axis=1)

        return post
    
    def _get_semantic_vectors_from_prox(self, prox, y, labels):
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)
    
        N, K = prox.shape
        n_classes = len(labels)
    
        lab2idx = {lab: k for k, lab in enumerate(labels)}
        y_lab = y[~mask_unl]
    
        if y_lab.size == 0:
            return np.zeros((N, n_classes), dtype=float)
    
        y_lab_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
    
        Y = np.zeros((K, n_classes), dtype=np.float64)
        idx_lab = np.flatnonzero(~mask_unl)
        Y[idx_lab, y_lab_idx] = 1.0
    
        post = prox.dot(Y) if sparse.issparse(prox) else np.dot(prox, Y)
    
        if self.prior_correct:
            counts = np.bincount(y_lab_idx, minlength=n_classes).astype(float)
            prior = counts / max(counts.sum(), 1.0)
            post *= (1.0 / np.maximum(prior, 1e-12))[None, :]
    
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
        print("post_a checksum:", np.max(post_a), np.mean(post_a), np.sum(post_a))
        print("post_b checksum:", np.max(post_b), np.mean(post_b), np.sum(post_b))
        if self.ot_solver == "hiref":
            return solve_surjection_hiref(
                post_a,
                post_b,
                verbose=self.verbose,
                random_state=self.random_state,
            )

        if self.ot_solver == "dense":
            return self._compute_dense_ot(post_a, post_b)

        raise ValueError(f"Unknown ot_solver={self.ot_solver}")

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix just like in old FoSTA.
        """
        if sparse.issparse(T):
            W_ab = self.mu * (prox_a.dot(T) + T.dot(prox_b)) / 2
        else:
            T = np.asarray(T)
            W_ab = self.mu * (
                prox_a.dot(T) + sparse.csr_matrix(T.dot(prox_b))
            ) / 2
    
        W_ab = sparse.csr_matrix(W_ab)
        W_ba = W_ab.transpose()
    
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)
            print_mat_stats("Cross-domain B→A (W21)", W_ba)
    
        return sparse.bmat(
            [
                [prox_a, W_ab],
                [W_ba, prox_b],
            ],
            format="csr",
        )

    def fit(self, x_a, x_b, y_a, y_b):
        """Fits FoSTA alignment across two domains."""
        self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
        self.n = self.n_a + self.n_b

        labels = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)
        self.classes_ = labels

        (
            self.leaf_pca_a_,
            self.Q_full_a_,
            self.W_lab_a_,
            self.kernel_a,
            self.idx_lab_a_,
            self.idx_unl_a_,
            self.prox_a,
        ) = self._compute_domain_geometry(x_a, y_a, "A")

        (
            self.leaf_pca_b_,
            self.Q_full_b_,
            self.W_lab_b_,
            self.kernel_b,
            self.idx_lab_b_,
            self.idx_unl_b_,
            self.prox_b,
        ) = self._compute_domain_geometry(x_b, y_b, "B")

        if self.old_version:
            post_a = self._get_semantic_vectors_from_prox(
                self.prox_a,
                y_a,
                labels,
            )
        
            post_b = self._get_semantic_vectors_from_prox(
                self.prox_b,
                y_b,
                labels,
            )
        else:
            post_a = self._get_semantic_vectors(
                self.Q_full_a_,
                self.W_lab_a_,
                y_a,
                labels,
                self.idx_lab_a_,
            )
        
            post_b = self._get_semantic_vectors(
                self.Q_full_b_,
                self.W_lab_b_,
                y_b,
                labels,
                self.idx_lab_b_,
            )

        self.post_a = post_a
        self.post_b = post_b

        self.T_sparse = self._compute_coupling(post_a, post_b)
        self.W = self._build_balanced_affinity(self.prox_a, self.prox_b, self.T_sparse)

        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """Fits alignment and computes embedding."""
        self.fit(x_a, x_b, y_a, y_b)

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

        else:
            self.embedding_ = UMAP(
                n_components=self.n_components,
                metric="precomputed",
                random_state=self.random_state,
            ).fit_transform(kernel2Dist(self.W.toarray()))

        return self.embedding_

    def get_embeddings(self):
        return self.embedding_