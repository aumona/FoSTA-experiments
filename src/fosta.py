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

# Utils
from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

# OT solver
from .hiref.adaptive_HiRef import solve_compressed_hiref, solve_dummy_hiref

import sys

class FoSTA(object):
    '''FoSTA: Forest-guided Semantic Transport Alignment'''
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 n_landmark=2000,
                 t='auto',
                 beta=0.7,
                 prior_correct=True,
                 semantic_norm='l2',  # normalization method for semantic vectors (supports 'l1' and 'l2')
                 embedder='spectral',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.n_landmark = n_landmark  # number of landmarks for DPT
        self.prior_correct = prior_correct
        self.t = t
        self.beta = beta
        self.semantic_norm = semantic_norm
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
            # 'class_weight': 'balanced',  # handle class imbalance in RF
            'verbose': 0,
            'n_jobs': n_jobs,
        }

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None
        self.n_a = None
        self.n_b = None

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
            landmark_ids = clusters[~mask_unl] # Map labeled points to their landmarks
            
            # Fast unbuffered add
            np.add.at(Y_encoded, (landmark_ids, y_idx), 1.0)
            
            # Row-Normalize (Counts -> Probs P(c|m))
            row_sums = Y_encoded.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0 # Prevent div/0 for empty landmarks
            Y_encoded /= row_sums

        # --- Diffusion (The Projection) ---
        # W is sparse (N, N) or dense (N, K), Y is dense (K, C) -> Result is Dense (N, C)
        # This calculates the raw sum of affinities to class signals
        post = W.dot(Y_encoded) if sparse.issparse(W) else np.dot(W, Y_encoded)

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
        W_ab = (prox_a.dot(T) + T.dot(prox_b)) / 2   # (n_a x n_b)
        W_ba = W_ab.transpose()
        W_sym = sparse.bmat(
            [
            [self.mu * prox_a, (1-self.mu) * W_ab],
            [(1-self.mu) * W_ba,   self.mu * prox_b]
            ],
            format="csr"
        )
        
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)

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

        print("Fitting RFGAP on Domain A...") if self.verbose > 0 else None
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
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(prox_a, random_state=self.random_state, verbose=True)
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(prox_b, random_state=self.random_state, verbose=True)
            M_a = self._compute_dpt(P_MM_a)
            M_b = self._compute_dpt(P_MM_b)
            trans_a = P_NM_a.dot(M_a)
            trans_b = P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b, prior_correct=self.prior_correct)

        print("Computing Optimal Transport...") if self.verbose > 0 else None
        self.T_sparse = solve_dummy_hiref(post_a, post_b, verbose=self.verbose, dummy_mode='uniform')


        # ---------- DIAGNOSTICS ----------
        T = self.T_sparse.tocsr()
        if self.verbose:
            print_mat_stats("Coupling Matrix", T)
            print("=================================\n")
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
                t=self.t,
                knn_dist='precomputed_affinity',
                kernel_symm='+',  # already use pre-symmetrized affinity, but set to '+' to be safe
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                beta=self.beta,
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