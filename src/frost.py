import numpy as np
from scipy import sparse
from sklearn import preprocessing

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
from .hiref import HiRef_fast as HiRef
from .hiref import rank_annealing

import sys

class FROST(object):
    '''Fast Random Forest-guided Optimal Semantic Transport for Manifold Alignment (FROST)'''
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 n_landmark=2000,
                 prior_correct=True,
                 dist='cosine',  # distance metric for HiRef (supports 'hellinger' and 'cosine')
                 embedder='spectral',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.n_landmark = n_landmark  # number of landmarks for DPT
        self.prior_correct = prior_correct
        self.dist = dist
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.rfgap_params = {
            'random_state': random_state,
            'prediction_type': 'classification',  # force classification mode
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'max_normalize': True,
            'class_weight': 'balanced',  # handle class imbalance in RF
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
        W,                 # (N, N) prox OR (N, M) landmark weights
        y,                 # (N,) labels (may include -1/NaN)
        labels,            # canonical labels (same across domains)
        clusters=None,     # None OR (N,) landmark id per point in [0..M-1]
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

        # Map labels -> [0..C-1]
        lab2idx = {lab: k for k, lab in enumerate(labels)}

        # --------------------------------------------------------
        # Compute Raw Posteriors
        # --------------------------------------------------------
        
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

                if prior_correct:
                    post[:, k] = vec / cnt
                else:
                    post[:, k] = vec

        # Case 2: Landmark DPT weights (N x M)
        else:
            clusters = np.asarray(clusters).ravel()
            if clusters.shape[0] != N:
                raise ValueError(f"clusters must have shape (N,), got {clusters.shape}")

            M = W.shape[1]
            # use only labeled points to estimate Q and prior
            y_lab = y[~mask_unl]
            cl_lab = clusters[~mask_unl]

            if y_lab.size == 0:
                raise ValueError("No labeled samples in y. Cannot build semantic vectors.")

            try:
                y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
            except KeyError as e:
                raise ValueError(f"Found label {e} in y that is not in `labels`.")

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

        # --------------------------------------------------------
        # Apply Metric Transformation
        # --------------------------------------------------------
        if self.dist == 'cosine':
            # L2 normalize and scale to prepare for cosine distance in HiRef
            post = preprocessing.normalize(post, norm="l2", axis=1) / np.sqrt(2)
            
        elif self.dist == 'hellinger':
            # L1 normalize and sqrt transform for Squared Hellinger in HiRef
            post = preprocessing.normalize(post, norm="l1", axis=1)
            post = np.sqrt(post) / np.sqrt(2)
        
        else:
            raise ValueError(f"Unknown dist={self.dist}, must be 'cosine' or 'hellinger'.")

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

    def compute_dpt(self, P):
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
    # Helper: Compressed OT Logic
    # ------------------------------------------------------------
    def _solve_compressed_ot(self, post_fixed, post_to_compress):
        """
        Solves OT where `post_to_compress` is clustered to match the size of `post_fixed`.
        Returns a sparse coupling matrix (n_fixed x n_to_compress).

        This implements the logic: T_full = T_coarse @ M_membership
        """
        n_fixed = post_fixed.shape[0]
        n_compress = post_to_compress.shape[0]
        k = n_fixed  # Compress target to match source size

        # Compress
        #    Cluster the larger domain into 'k' centroids to match the size of the smaller domain.
        km = MiniBatchKMeans(n_clusters=k, random_state=self.random_state)
        z = km.fit_predict(post_to_compress)         # (n_compress,) assignments
        post_centroids = km.cluster_centers_         # (k, C) semantic centers

        # HiRef (Fixed <-> Centroids)
        #    Calculate OT between the fixed domain and the centroids.
        #    T_coarse[i, j] = 1 if point i is assigned to centroid j.
        rank_schedule = rank_annealing.optimal_rank_schedule(n=n_fixed)
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            post_fixed, post_centroids,
            rank_schedule=rank_schedule,
            return_coupling=True
        )
        
        T_coarse = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_fixed, k)
        ).tocsr()

        # Build Membership (Centroids -> Original Points)
        #    We construct M_membership to satisfy the requirement:
        #    "Assign to every point in the cluster with value 1/cluster_size"
        #
        #    If cluster 'j' has N_j points:
        #      M_membership[j, p] = 1 / N_j   for all points p in cluster j
        #      M_membership[j, p] = 0         otherwise
        
        counts = np.bincount(z, minlength=k).astype(np.float64)
        
        # Safe inverse:
        # - If cluster is empty (counts=0), inv_counts becomes 0.
        # - This ensures empty clusters result in empty rows (unassigned).
        inv_counts = np.zeros_like(counts)
        inv_counts[counts > 0] = 1.0 / counts[counts > 0]
        
        M_membership = sparse.coo_matrix(
            (inv_counts[z], (z, np.arange(n_compress, dtype=np.int64))),
            shape=(k, n_compress)
        ).tocsr()

        # Lift back to full resolution
        #    T_full = T_coarse @ M_membership
        #
        #    Logic:
        #    - If T_coarse connects point x to centroid j (weight 1),
        #    - And M_membership connects centroid j to points {p1..pN} (weight 1/N),
        #    - The dot product connects point x to ALL points {p1..pN} with weight 1/N.
        #
        #    Edge cases handled automatically by sparse algebra:
        #    - If x is assigned to multiple centroids, contributions are summed.
        #    - If a centroid is empty (inv_count=0), x gets 0 connection to it (no mass created).
        #    - If x is unassigned (T_coarse row is 0), the result row is 0.
        T_full = (T_coarse @ M_membership).tocsr()

        # Diagnostics for empty clusters
        n_empty = int(np.sum(counts == 0))
        if n_empty > 0:
            print(f"[WARN] MiniBatchKMeans produced {n_empty}/{k} empty clusters during compression.")

        return T_full
    
    
    # ------------------------------------------------------------
    # Balanced Affinity Construction
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix just like in MALI, with T the OT coupling, max normalized.
        """
        W_ab = (prox_a.dot(T) + T.dot(prox_b))   # (n_a x n_b)
        W_ba = W_ab.transpose()
        W_sym = sparse.bmat(
            [
                [self.mu * prox_a, (1-self.mu) * W_ab],
                [(1-self.mu) * W_ba,   self.mu * prox_b]
            ],
            format="csr"
        )
        return W_sym

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

        print("Fitting RFGAP on Domain B...") if self.verbose > 0 else None
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()

        print("Building C-dim vectors...") if self.verbose > 0 else None
        if not self.dpt:
            post_a = self._get_semantic_vectors(prox_a, y_a, labels, clusters=None, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(prox_b, y_b, labels, clusters=None, prior_correct=self.prior_correct)
        else:
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(
                prox_a, random_state=self.random_state, verbose=True
            )
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(
                prox_b, random_state=self.random_state, verbose=True
            )
            M_a = self.compute_dpt(P_MM_a)
            M_b = self.compute_dpt(P_MM_b)
            self._dpt_M_a, self._dpt_M_b = M_a, M_b
            self._clusters_a, self._clusters_b = clusters_a, clusters_b
            
            trans_a = P_NM_a.dot(M_a)
            trans_b = P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b, prior_correct=self.prior_correct)

        print("Computing Optimal Transport...") if self.verbose > 0 else None
        n_a, n_b = self.n_a, self.n_b

        if n_a == n_b:
            # --- Balanced Case (equivelent to m=1 in MALI) ---
            rank_schedule = rank_annealing.optimal_rank_schedule(n=n_a)
            (rows, cols, data), _ = HiRef.hiref_lr_fast(
                post_a, post_b,
                rank_schedule=rank_schedule,
                return_coupling=True
            )
            self.T_sparse = sparse.coo_matrix(
                (data, (rows, cols)), shape=(n_a, n_b)
            ).tocsr()

        # Imbalanced cases, correspond to n_a =! n_b with m=1 in MALI
        elif n_a < n_b:
            # --- Compress B -> A ---
            # Returns T (n_a x n_b) directly
            self.T_sparse = self._solve_compressed_ot(post_fixed=post_a, post_to_compress=post_b)

        else: # n_b < n_a
            # --- Compress A -> B ---
            # Returns T (n_b x n_a). We need A->B, so we transpose.
            T_ba = self._solve_compressed_ot(post_fixed=post_b, post_to_compress=post_a)
            self.T_sparse = T_ba.T.tocsr()

        # ---------- DIAGNOSTICS ----------
        T = self.T_sparse.tocsr()
        if self.verbose:
            print("\nSPARSE COUPLING")
            print("---------------")
            print(f"Empty rows: {np.sum(np.diff(T.indptr) == 0)} / {T.shape[0]}")
            print(f"Empty cols: {np.sum(np.diff(T.tocsc().indptr) == 0)} / {T.shape[1]}")
            print("Total mass:", T.sum())
            # print row/col sums
            row_sums = np.asarray(T.sum(axis=1)).ravel()
            col_sums = np.asarray(T.sum(axis=0)).ravel()
            print(f"Row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")
            print(f"Col sums: min={col_sums.min():.4f}, max={col_sums.max():.4f}, mean={col_sums.mean():.4f}")
            print("Building joint affinity matrix...")

        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

        print("Model fit complete.") if self.verbose > 0 else None
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        print("Computing joint embedding...")

        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                kernel_symm=None,  # already use pre-symmetrized affinity
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