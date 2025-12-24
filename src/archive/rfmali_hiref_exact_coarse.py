import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import math

# Graph tools
import graphtools
from rfgap import RFGAP

# Embedders
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

# Utils
from utils.utils import kernel2Dist
from utils.labels import LabelUtils
from .hiref import HiRef_fast_exact as HiRef

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
    # Coarse / Quantized OT Logic
    # ------------------------------------------------------------
    def _solve_quantized_ot(self, post_a, post_b):
        """
        Solves OT between A and B using a 'Coarse-to-Fine' strategy to handle 
        class imbalance. It quantizes (clusters) the larger domain to match the 
        size of the smaller domain, solves a square 1-to-1 matching, and then 
        broadcasts the solution back.
        """
        n_a, dim = post_a.shape
        n_b, _ = post_b.shape
        
        # Determine strategy based on sizes
        # If sizes are roughly equal (within 10%), just use standard matching
        if abs(n_a - n_b) / max(n_a, n_b) < 0.1:
            mode = "Equal"
            n_target = min(n_a, n_b)
            # Truncate slightly if not perfectly equal to make square (or handle rectangular)
            # For simplicity here we assume standard HiRef handles rectangular 
            # or we just let it run. But strict quantization logic:
            source_vecs = post_a
            target_vecs = post_b
            labels_large = None
            
        elif n_a > n_b:
            mode = "A_to_B" # Cluster A to size of B
            n_target = n_b
            large_post = post_a
            small_post = post_b
        else: # n_b > n_a
            mode = "B_to_A" # Cluster B to size of A
            n_target = n_a
            large_post = post_b
            small_post = post_a

        # 1. Clustering Step (Quantization)
        # ---------------------------------------------------
        if mode != "Equal":
            if self.verbose >= 0:
                print(f"Imbalance detected ({n_a} vs {n_b}). Quantizing larger domain via KMeans...")
            
            # Use MiniBatchKMeans for efficiency
            k = min(n_target, large_post.shape[0])
            
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(256 * k, 10000),
                random_state=self.random_state,
                n_init=3
            )
            kmeans.fit(large_post)
            
            # The "Source" for OT is now the centroids
            # Re-normalize centroids so cosine distance logic holds
            source_vecs = preprocessing.normalize(kmeans.cluster_centers_, norm='l2')
            target_vecs = small_post
            labels_large = kmeans.labels_
        else:
            source_vecs = post_a
            target_vecs = post_b

        # 2. Solve Square OT
        # ---------------------------------------------------
        # Effective N for OT
        n_ot = min(source_vecs.shape[0], target_vecs.shape[0])
        
        # Pad to power of 2 for HiRef
        pad_exp = math.ceil(math.log2(n_ot))
        n_pad = 2 ** pad_exp
        DUMMY_VAL = 1.0 
        
        def _pad(M):
            curr_n, curr_c = M.shape
            M_pad = np.zeros((n_pad, curr_c + 1), dtype=M.dtype)
            M_pad[:curr_n, :curr_c] = M
            if curr_n < n_pad:
                M_pad[curr_n:, -1] = DUMMY_VAL
            return M_pad

        src_pad = _pad(source_vecs[:n_ot]) # Ensure we take matching sizes if "Equal" mode slightly off
        tgt_pad = _pad(target_vecs[:n_ot])
        
        # Binary rank schedule for HiRef (fastest)
        rank_schedule = [2] * pad_exp
        
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            src_pad, tgt_pad,
            rank_schedule=rank_schedule,
            base_rank=1,
            rescale_cost=True,
            return_coupling=True
        )
        
        # Filter padding
        valid_mask = (rows < n_ot) & (cols < n_ot)
        rows_clean = rows[valid_mask]
        cols_clean = cols[valid_mask]
        data_clean = data[valid_mask]
        
        # Square Coupling Matrix (approx 1-to-1)
        T_coarse = sparse.coo_matrix(
            (data_clean, (rows_clean, cols_clean)), 
            shape=(n_ot, n_ot)
        ).tocsr()

        # 3. Broadcast / De-Quantize
        # ---------------------------------------------------
        if mode == "Equal":
            # Just return direct match (handling slight size mismatch if any by padding zeros)
            # If strictly equal, T_coarse is fine. 
            if n_a != n_b:
                # Resize result to full shape if slight mismatch existed
                T_final = sparse.coo_matrix((n_a, n_b), dtype=float)
                # (Simple truncation logic for "Equal" case, usually n_a==n_b here)
                return T_coarse 
            return T_coarse
            
        elif mode == "A_to_B":
            # A was clustered. T_coarse is (n_centroids_A x n_B).
            # We want T_fine (n_A x n_B).
            # Broadcast: Every point in A inherits the connection of its centroid.
            # Efficient implementation: Row slicing
            T_fine = T_coarse[labels_large, :]
            return T_fine
            
        elif mode == "B_to_A":
            # B was clustered. T_coarse is (n_A x n_centroids_B).
            # We want T_fine (n_A x n_B).
            # Broadcast: Every point in B inherits the connection of its centroid.
            # Efficient implementation: T_fine = T_coarse * OneHot(labels_B)^T
            
            row_idx = labels_large # Cluster ID for each point in B
            col_idx = np.arange(len(labels_large)) # Point ID in B
            vals = np.ones(len(labels_large))
            
            # Expander maps (n_centroids_B -> n_B)
            Expander = sparse.coo_matrix(
                (vals, (row_idx, col_idx)),
                shape=(n_ot, len(labels_large))
            ).tocsr()
            
            T_fine = T_coarse.dot(Expander)
            return T_fine

    # ------------------------------------------------------------
    # Balanced Affinity Construction
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
            """
            Constructs a joint affinity matrix using independent pre-normalization.
            
            Robustness:
            - Works if N_a > N_b (T rows sum to 1, T.T rows sum to ClusterSize)
            - Works if N_b > N_a (T rows sum to ClusterSize, T.T rows sum to 1)
            - Works if N_a ~ N_b
        
            More efficient symmetrization:
            - Avoids building full P then doing (P + P.T)/2.
            - Builds symmetric blocks directly:
                W_sym = [[sym(A), sym_off(B,C)], [sym_off(C,B), sym(D)]]
            """
        
            T_ab = preprocessing.normalize(T, norm="l1", axis=1)  # T_ab: Probability of jumping A -> B (Row-stochastic)
            T_ba = preprocessing.normalize(T.transpose(), norm="l1", axis=1)  # T_ba: Probability of jumping B -> A (Row-stochastic)
        
            # Cross blocks (sparse)
            W_ab = (prox_a.dot(T_ab) + T_ab.dot(prox_b)) / 2   # (n_a x n_b)
            W_ba = (prox_b.dot(T_ba) + T_ba.dot(prox_a)) / 2   # (n_b x n_a)
        
            # Symmetrize off-diagonal blocks without building full W
            UR = (W_ab + W_ba.transpose()) * 0.5
            LL = UR.transpose()
        
            W_sym = sparse.bmat(
                [
                    [self.mu * prox_a, (1-self.mu) * UR],
                    [(1-self.mu) * LL,   self.mu * prox_b]
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
        # OT with Quantization Strategy (Imbalance Fix)
        # ------------------------------------------------------------
        print("Computing Monge Coupling (using Quantized OT if needed)...")
        
        self.T_sparse = self._solve_quantized_ot(post_a, post_b)

        # Enforce binary weights for graph clarity before smoothing
        # self.T_sparse.data[:] = 1.0 

        # ------------------------------------------------------------
        # Joint Graph Construction
        # ------------------------------------------------------------
        print("Building balanced joint affinity matrix...")
        
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

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
                kernel_symm=None,
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
        
        elif self.embedder == 'barycentric':
            raise NotImplementedError("Barycentric embedding not implemented in this version.")

        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

    def get_embeddings(self):
        return self.embedding_