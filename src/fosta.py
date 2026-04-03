import time
import numpy as np
from scipy import sparse
from sklearn import preprocessing

# Graph tools for DPT
import graphtools

# RF-GAP
from forestkernel import ForestKernel

# Embedders
from src.phate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

# Utils
from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

# OT solver
from .hiref.adaptive_HiRef import solve_surjection_hiref

import sys


class FoSTA(object):
    '''FoSTA: Forest-guided Semantic Transport Alignment'''
    def __init__(self,
                 mu=0.5,  # cross-domain block strength (0.5 = equal weight, 0.0 = ignore cross-domain affinities, 1.0 = rely solely on cross-domain affinities)
                 dpt=False,
                 kernel_method='gap',
                 model_type='rf',
                 n_landmark=2000,

                 euclidean_mode=False,  # MALI-style approach without forest-guided proximities (only for ablation, not recommended)
                 n_pca=100,  # only for euclidean mode
                 n_neighbors=5,  # only for euclidean mode
                 decay=40,  # only for euclidean mode
                 knn_dist='euclidean',  # only for euclidean mode

                 t='auto',
                 beta=0.7,
                 n_estimators=1000,
                 prior_correct=True,
                 semantic_norm='l2',  # normalization method for semantic vectors (supports 'l1' and 'l2')
                 embedder='PHATE',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.kernel_method = kernel_method
        self.model_type = model_type
        self.n_landmark = n_landmark  # number of landmarks for DPT
        self.euclidean_mode = euclidean_mode
        self.n_pca = n_pca
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
        self.n_estimators = n_estimators

        self.kernel_params = {
            'random_state': self.random_state,
            'prediction_type': 'classification',  # force classification mode
            'n_estimators': self.n_estimators,
            'kernel_method': self.kernel_method,
            'model_type': self.model_type,
            'force_nonzero_diag': True,
            'force_symmetric': False,  # Better transfer without forcing symmetry in RFGAP
            'normalize_diagonal': True,
            'allow_semi_supervised': True,  # Allow unlabeled data to influence kernels
        }

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

        self.runtime_summary_ = None

    # ------------------------------------------------------------
    # Posterior builders
    # ------------------------------------------------------------
    def _get_semantic_vectors(
        self,
        W,                     # (N, K) Adjacency. K=N (Full, sparse) or K=M (Landmarks, dense)
        y,                     # (N,) Labels
        labels,                # List of unique canonical labels
        clusters=None,         # (N,) Landmark IDs (optional, only for landmark path)
        eps=1e-12,
        prior_correct=True
    ):
        """
        Builds C-dim semantic vectors (posteriors) via unified matrix diffusion.

        Logic:
        1. Construct Signal Basis Y_encoded (K x C):
            - If Full: One-hot encoding of labeled points.
            - If Landmarks: Probability distribution P(class | landmark).
        2. Diffuse: Post = W @ Y_encoded
        3. Correct: Divide by class priors to handle imbalance.
        4. Transform: Apply Metric scaling (Cosine/Hellinger).
        """
        # --- Setup Data & Labels ---
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)

        N, K = W.shape
        C = len(labels)

        # Map labels -> [0..C-1]
        lab2idx = {lab: k for k, lab in enumerate(labels)}

        # Filter strictly to labeled data
        y_lab = y[~mask_unl]
        if y_lab.size == 0:
            return np.zeros((N, C), dtype=float)

        try:
            y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(f"Found label {e} in y that is not in `labels`.")

        # --- Compute Priors (for correction) ---
        # We compute this regardless of path to support prior_correct
        counts_c = np.bincount(y_idx, minlength=C).astype(float)
        n_lab = float(counts_c.sum())

        # p_c = count / total
        class_prior = counts_c / max(n_lab, 1.0)
        # 1/p_c (used to normalize "Total Affinity" to "Average Affinity")
        inv_prior = 1.0 / np.maximum(class_prior, eps)

        # --- Construct Signal Basis Y_encoded (K x C) ---
        # This matrix represents the initial class signal on the graph nodes (points or landmarks)
        Y_encoded = np.zeros((K, C), dtype=np.float64)

        if clusters is None:
            # Case A: Full Graph (Basis = Points, K=N)
            # Create strict One-Hot encoding for labeled points
            # Y[i, c] = 1.0 if point i has label c, else 0
            labeled_indices = np.flatnonzero(~mask_unl)
            Y_encoded[labeled_indices, y_idx] = 1.0

        else:
            # Case B: Landmarks (Basis = Landmarks, K=M)
            # Aggregate counts: "Landmark k contains 5 Class A and 10 Class B"
            clusters = np.asarray(clusters).ravel()
            landmark_ids = clusters[~mask_unl]  # Map labeled points to their landmarks

            # Fast unbuffered add
            np.add.at(Y_encoded, (landmark_ids, y_idx), 1.0)

            # Row-Normalize (Counts -> Probs P(c|m))
            row_sums = Y_encoded.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Prevent div/0 for empty landmarks
            Y_encoded /= row_sums

        # --- projection onto semantic space ---
        # W is sparse (N, N) or dense (N, K), Y is dense (K, C) -> Result is Dense (N, C)
        # This calculates the raw sum of affinities to class signals
        post = W.dot(Y_encoded)

        # Remove self-similarity contribution (labeled points only)
        # Efficient leave-self-out: subtract W[i,i] from the column corresponding to y_i
        if clusters is None:
            # diag of W as (N,)
            d = W.diagonal()
            # subtract only for labeled points
            labeled_indices = np.flatnonzero(~mask_unl)
            # y_idx is only labels for labeled points; map those onto full-length array
            y_full_idx = np.empty(N, dtype=int)
            y_full_idx[~mask_unl] = y_idx
            post[labeled_indices, y_full_idx[~mask_unl]] -= d[labeled_indices]

        # --- Prior Correction ---
        if prior_correct:
            # Broadcast multiplication: post[:, c] *= (1 / p_c)
            # Converts "Total Affinity" -> "Density-Independent Affinity"
            post *= inv_prior[None, :]

        # --- Metric Transformation ---
        # Prepares vectors so standard Euclidean distance downstream matches desired metric
        if self.semantic_norm == 'l2':  # to simulate Cosine distance, same as MALI, and better fits Hiref_fast assumptions
            post = preprocessing.normalize(post, norm="l2", axis=1)

        elif self.semantic_norm == 'l1':
            post = preprocessing.normalize(post, norm="l1", axis=1)
        else:
            print(f"[WARN] Unknown normalization={self.semantic_norm}, skipping metric transform.")

        return post

    # ------------------------------------------------------------
    # DPT / Landmark diffusion machinery
    # ------------------------------------------------------------
    def _get_diffusion_operators(self, K, random_state=None, verbose=True, **graph_kwargs):
        """
        Returns P_NM (N x M), P_MM (M x M), clusters (N,)
        """
        n_landmark = self.n_landmark
        G = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark if n_landmark is not None and n_landmark < K.shape[0] else None,
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

    def _compute_dpt(self, P):
        """
        DPT-like aggregated transition matrix: (I - (P - 1 phi^T))^{-1} - I
        """
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
    # Balanced Affinity Construction
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix just like in MALI, with max-normalized T as input (OT coupling matrix)
        """
        W_ab = prox_a.dot(T)  # (n_a x n_b)
        W_ba = prox_b.dot(T.transpose())  # (n_b x n_a)
        W_sym = sparse.bmat(
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

        return W_sym

    def _print_runtime_summary(self):
        if not self.verbose or self.runtime_summary_ is None:
            return

        times = self.runtime_summary_
        total = times["total"]
        print("\nRUNTIME SUMMARY")
        print("---------------")
        print(f"Total: {total:.3f}s")
        for key in ["geometry", "semantics", "transport", "joint_affinity", "embedding"]:
            val = times[key]
            pct = 100.0 * val / total if total > 0 else 0.0
            print(f"{key:15s} {val:8.3f}s   ({pct:5.1f}%)")

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

        t_geometry = time.perf_counter()
        if not self.euclidean_mode:
            print("Fitting RFGAP on Domain A...") if self.verbose > 0 else None
            self.kernel_a = ForestKernel(**self.kernel_params)
            self.kernel_a.fit(x_a, y_a)
            prox_a = self.kernel_a.get_kernel()

            print("Fitting RFGAP on Domain B...") if self.verbose > 0 else None
            self.kernel_b = ForestKernel(**self.kernel_params)
            self.kernel_b.fit(x_b, y_b)
            prox_b = self.kernel_b.get_kernel()

        else:
            n_pca_a = min(self.n_pca, x_a.shape[1]) if self.n_pca is not None else None
            if n_pca_a is not None and n_pca_a < 100:
                n_pca_a = None
            prox_a = graphtools.Graph(
                x_a,
                n_pca=n_pca_a,
                knn=self.n_neighbors,
                decay=self.decay,
                distance=self.knn_dist,
                thresh=1e-4,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=False
            ).K

            n_pca_b = min(self.n_pca, x_b.shape[1]) if self.n_pca is not None else None
            if n_pca_b is not None and n_pca_b < 100:
                n_pca_b = None
            prox_b = graphtools.Graph(
                x_b,
                n_pca=n_pca_b,
                knn=self.n_neighbors,
                decay=self.decay,
                distance=self.knn_dist,
                thresh=1e-4,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=False
            ).K
        t_geometry = time.perf_counter() - t_geometry

        print("Building C-dim vectors...") if self.verbose > 0 else None
        t_semantics = time.perf_counter()
        if not self.dpt:
            post_a = self._get_semantic_vectors(
                prox_a, y_a, labels, clusters=None, prior_correct=self.prior_correct
            )
            post_b = self._get_semantic_vectors(
                prox_b, y_b, labels, clusters=None, prior_correct=self.prior_correct
            )
        else:
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(
                prox_a, random_state=self.random_state, verbose=True
            )
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(
                prox_b, random_state=self.random_state, verbose=True
            )
            M_a = self._compute_dpt(P_MM_a)
            M_b = self._compute_dpt(P_MM_b)
            trans_a = P_NM_a.dot(M_a)
            trans_b = P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(
                trans_a, y_a, labels, clusters=clusters_a, prior_correct=self.prior_correct
            )
            post_b = self._get_semantic_vectors(
                trans_b, y_b, labels, clusters=clusters_b, prior_correct=self.prior_correct
            )
        t_semantics = time.perf_counter() - t_semantics

        print("Computing Optimal Transport...") if self.verbose > 0 else None
        t_transport = time.perf_counter()
        self.T_sparse = solve_surjection_hiref(
            post_a, post_b, verbose=self.verbose, random_state=self.random_state
        )
        t_transport = time.perf_counter() - t_transport

        # ---------- DIAGNOSTICS ----------
        T = self.T_sparse.tocsr()
        if self.verbose:
            print_mat_stats("Coupling Matrix", T)
            print("=================================\n")
            print("Building joint affinity matrix...")

        t_joint = time.perf_counter()
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)
        t_joint = time.perf_counter() - t_joint

        self.runtime_summary_ = {
            "geometry": float(t_geometry),
            "semantics": float(t_semantics),
            "transport": float(t_transport),
            "joint_affinity": float(t_joint),
            "embedding": 0.0,
            "total": float(t_geometry + t_semantics + t_transport + t_joint),
        }

        print("Model fit complete.") if self.verbose > 0 else None
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        t_total = time.perf_counter()
        self.fit(x_a, x_b, y_a, y_b)
        print("Computing joint embedding...") if self.verbose > 0 else None

        t_embed = time.perf_counter()
        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t=self.t,
                knn_dist='precomputed_affinity',
                kernel_symm='+',
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                beta=self.beta,
            )
            self.embedding_ = phate_op.fit_transform(self.W)

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
        t_embed = time.perf_counter() - t_embed

        total = time.perf_counter() - t_total
        self.runtime_summary_["embedding"] = float(t_embed)
        self.runtime_summary_["total"] = float(total)
        self._print_runtime_summary()

        return self.embedding_

    def get_embeddings(self):
        return self.embedding_