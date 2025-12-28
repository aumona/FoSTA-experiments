import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# Graph tools
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
from ..hiref import HiRef_fast as HiRef
from ..hiref import rank_annealing

class FROST(object):
    '''Fast Random Forest-guided Optimal Semantic Transport for Manifold Alignment (FROST)'''
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 n_landmark=2000,
                 prior_correct=True,
                 semantic_norm='l1',
                 embedder='spectral',
                 n_components=2,
                 verbose=0,
                 random_state=None,
                 sugar_k=5,         # Neighbors for SUGAR density estimation
                 n_jobs=-1):
        
        self.mu = mu
        self.dpt = dpt
        self.n_landmark = n_landmark
        self.prior_correct = prior_correct
        self.semantic_norm = semantic_norm
        self.embedder = embedder
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.sugar_k = sugar_k

        self.rfgap_params = {
            'random_state': random_state,
            'prediction_type': 'classification',
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'max_normalize': True,
            'class_weight': 'balanced',
            'verbose': 0,
            'n_jobs': n_jobs,
        }

        self.T_sparse = None
        self.W = None
        self.embedding_ = None
        self.classes_ = None
        
        # Track sizes for slicing
        self.n_a_orig = None
        self.n_b_orig = None
        self.n_a_aug = None

    # ------------------------------------------------------------
    # SUGAR (Subsampling Using Geodesic AReas) - Oversampling
    # ------------------------------------------------------------
    def _sugar_augment(self, X, y, target_n):
        """
        Performs geometry-preserving oversampling (SUGAR).
        """
        current_n = X.shape[0]
        n_needed = target_n - current_n
        
        if n_needed <= 0:
            return X, y

        if self.verbose:
            print(f"SUGAR: Augmenting {current_n} -> {target_n} samples ({n_needed} new).")

        # Standardize for NN search (to handle feature scaling issues)
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Estimate local density via k-NN
        nbrs = NearestNeighbors(n_neighbors=self.sugar_k + 1, n_jobs=self.n_jobs).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Radius of the k-th neighbor (proxy for inverse density)
        # Larger radius = sparser region = higher probability to sample
        radii = distances[:, -1] 
        probs = radii / np.sum(radii)
        
        # 2. Generate Points
        rng = np.random.RandomState(self.random_state)
        chosen_indices = rng.choice(np.arange(current_n), size=n_needed, replace=True, p=probs)
        
        X_synth = []
        y_synth = []
        
        for idx in chosen_indices:
            x_anchor = X[idx]
            y_anchor = y[idx]
            
            # Find a random neighbor of the SAME CLASS
            nn_indices = indices[idx, 1:] # exclude self
            same_class_nn = [n_idx for n_idx in nn_indices if y[n_idx] == y_anchor]
            
            if len(same_class_nn) > 0:
                target_idx = rng.choice(same_class_nn)
                x_target = X[target_idx]
                
                # Interpolate (Convex Combination)
                alpha = rng.uniform(0.1, 0.9)
                x_gen = x_anchor + alpha * (x_target - x_anchor)
            else:
                # Fallback: duplicate with tiny jitter
                x_gen = x_anchor + rng.normal(0, 0.01 * np.std(X, axis=0))

            X_synth.append(x_gen)
            y_synth.append(y_anchor)
            
        X_augmented = np.vstack([X, np.array(X_synth)])
        y_augmented = np.concatenate([y, np.array(y_synth)])
        
        return X_augmented, y_augmented

    # ------------------------------------------------------------
    # Posterior builders
    # ------------------------------------------------------------        
    def _get_semantic_vectors(self, W, y, labels, clusters=None, eps=1e-12, prior_correct=True):
        y = np.asarray(y).ravel()
        mask_unl = LabelUtils.get_unlabeled_mask(y)
        N, K = W.shape
        C = len(labels)
        lab2idx = {lab: k for k, lab in enumerate(labels)}
        
        y_lab = y[~mask_unl]
        if y_lab.size == 0: return np.zeros((N, C), dtype=float)

        try:
            y_idx = np.array([lab2idx[v] for v in y_lab], dtype=int)
        except KeyError as e:
            raise ValueError(f"Found label {e} in y that is not in `labels`.")

        counts_c = np.bincount(y_idx, minlength=C).astype(float)
        inv_prior = 1.0 / np.maximum(counts_c / max(counts_c.sum(), 1.0), eps)

        Y_encoded = np.zeros((K, C), dtype=np.float64)

        if clusters is None:
            Y_encoded[np.flatnonzero(~mask_unl), y_idx] = 1.0
        else:
            clusters = np.asarray(clusters).ravel()
            landmark_ids = clusters[~mask_unl]
            np.add.at(Y_encoded, (landmark_ids, y_idx), 1.0)
            row_sums = Y_encoded.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            Y_encoded /= row_sums

        post = W.dot(Y_encoded) if sparse.issparse(W) else np.dot(W, Y_encoded)

        if prior_correct:
            post *= inv_prior[None, :]

        if self.semantic_norm == 'l2':
            post = preprocessing.normalize(post, norm="l2", axis=1) / np.sqrt(2)
        elif self.semantic_norm == 'l1':
            post = preprocessing.normalize(post, norm="l1", axis=1)

        return post

    # ------------------------------------------------------------
    # DPT / Landmark machinery
    # ------------------------------------------------------------
    def _get_diffusion_operators(self, K, random_state=None, verbose=True, **graph_kwargs):
        n_landmark = self.n_landmark
        G = graphtools.Graph(
            K, precomputed="affinity",
            n_landmark=n_landmark if n_landmark is not None and n_landmark < K.shape[0] else None,
            kernel_symm=None, random_state=random_state, verbose=verbose, **graph_kwargs,
        )
        if hasattr(G, "landmark_op") and hasattr(G, "transitions"):
            P_MM = np.asarray(G.landmark_op)
            P_NM = G.transitions.toarray()
            clusters = np.asarray(G.clusters).ravel()
            _, clusters = np.unique(clusters, return_inverse=True)
            M = P_NM.shape[1]
            if P_MM.shape[0] > M: P_MM = P_MM[:M, :M]
            return P_NM, P_MM, clusters

        P_MM = G.P.toarray()
        P_NM = np.eye(P_MM.shape[0])
        clusters = np.arange(P_MM.shape[0])
        return P_NM, P_MM, clusters

    def compute_dpt(self, P):
        n = P.shape[0]
        wL, lv = np.linalg.eig(P.T)
        phi0 = np.maximum(lv[:, np.argmin(np.abs(wL - 1.0))].real, 0)
        phi0 /= phi0.sum()
        Mmat = np.linalg.solve(np.eye(n) - (P - np.outer(np.ones(n), phi0)), np.eye(n)) - np.eye(n)
        return preprocessing.MinMaxScaler().fit_transform(Mmat.T).T

    # ------------------------------------------------------------
    # Balanced Affinity Construction
    # ------------------------------------------------------------
    def _build_balanced_affinity(self, prox_a, prox_b, T):
        W_ab = (prox_a.dot(T) + T.dot(prox_b)) 
        W_ba = W_ab.transpose()
        W_sym = sparse.bmat([
            [self.mu * prox_a, (1-self.mu) * W_ab],
            [(1-self.mu) * W_ba, self.mu * prox_b]
        ], format="csr")
        return W_sym

    # ------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------
    def fit(self, x_a, x_b, y_a, y_b):
        # 1. Track Original Sizes
        self.n_a_orig = x_a.shape[0]
        self.n_b_orig = x_b.shape[0]
        
        # 2. Detect Imbalance and Apply SUGAR (Pre-processing)
        # Note: We overwrite x_a/x_b with augmented versions for training
        if self.n_a_orig < self.n_b_orig:
            x_a, y_a = self._sugar_augment(x_a, y_a, target_n=self.n_b_orig)
        elif self.n_b_orig < self.n_a_orig:
            x_b, y_b = self._sugar_augment(x_b, y_b, target_n=self.n_a_orig)
            
        # Track augmented sizes for slicing later
        self.n_a_aug = x_a.shape[0]
        self.n_b_aug = x_b.shape[0]
        self.n = self.n_a_aug + self.n_b_aug
        
        if self.verbose:
            print(f"Data balanced. A: {self.n_a_orig}->{self.n_a_aug}, B: {self.n_b_orig}->{self.n_b_aug}")

        y_a = np.asarray(y_a).ravel()
        y_b = np.asarray(y_b).ravel()
        labels = LabelUtils.validate_shared_labels(y_a, y_b, strict=True)
        self.classes_ = labels

        # 3. Fit RFGAP on Augmented Data
        if self.verbose: print("Fitting RFGAP on Domain A...")
        self.rfgap_a = RFGAP(**self.rfgap_params)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities()

        if self.verbose: print("Fitting RFGAP on Domain B...")
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()

        # 4. Build Semantic Vectors
        if self.verbose: print("Building C-dim vectors...")
        if not self.dpt:
            post_a = self._get_semantic_vectors(prox_a, y_a, labels, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(prox_b, y_b, labels, prior_correct=self.prior_correct)
        else:
            P_NM_a, P_MM_a, clusters_a = self._get_diffusion_operators(prox_a, random_state=self.random_state)
            P_NM_b, P_MM_b, clusters_b = self._get_diffusion_operators(prox_b, random_state=self.random_state)
            M_a, M_b = self.compute_dpt(P_MM_a), self.compute_dpt(P_MM_b)
            
            trans_a, trans_b = P_NM_a.dot(M_a), P_NM_b.dot(M_b)
            post_a = self._get_semantic_vectors(trans_a, y_a, labels, clusters=clusters_a, prior_correct=self.prior_correct)
            post_b = self._get_semantic_vectors(trans_b, y_b, labels, clusters=clusters_b, prior_correct=self.prior_correct)

        # 5. Balanced Optimal Transport
        if self.verbose: print("Computing Balanced Optimal Transport...")
        rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n_a_aug)
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            post_a, post_b,
            rank_schedule=rank_schedule,
            return_coupling=True
        )
        self.T_sparse = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.n_a_aug, self.n_b_aug)
        ).tocsr()

        # 6. Joint Graph
        self.W = self._build_balanced_affinity(prox_a, prox_b, self.T_sparse)

        if self.verbose: print("Model fit complete.")
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """
        Fits the model and returns the embedding ONLY for the original points.
        Synthetic points are used for structural alignment but are sliced out of the result.
        """
        # Fits on augmented data
        self.fit(x_a, x_b, y_a, y_b)
        
        if self.verbose: print("Computing joint embedding...")
        
        # Compute embedding on the FULL augmented graph (N_aug_a + N_aug_b)
        if self.embedder == 'PHATE':
            full_embedding = PageRankPHATE(
                n_components=self.n_components, t='auto', knn_dist='precomputed_affinity',
                random_state=self.random_state, n_jobs=-1, verbose=self.verbose
            ).fit_transform(self.W)
        elif self.embedder == 'spectral':
            full_embedding = SpectralEmbedding(
                n_components=self.n_components, affinity='precomputed',
                random_state=self.random_state, n_jobs=self.n_jobs
            ).fit_transform(self.W)
        elif self.embedder == 'UMAP':
            full_embedding = UMAP(
                n_components=self.n_components, metric='precomputed',
                random_state=self.random_state
            ).fit_transform(kernel2Dist(self.W.toarray()))
        
        self.embedding_ = full_embedding # Store full embedding for debug
        
        # Slice out synthetic points
        # Structure of W and Embedding is: [Domain A (Augmented) | Domain B (Augmented)]
        # Domain A Original: indices 0 to n_a_orig
        # Domain B Original: indices n_a_aug to (n_a_aug + n_b_orig)
        
        emb_a_original = full_embedding[:self.n_a_orig]
        
        start_b = self.n_a_aug
        end_b = start_b + self.n_b_orig
        emb_b_original = full_embedding[start_b:end_b]
        
        return np.vstack([emb_a_original, emb_b_original])

    def get_embeddings(self):
        """Returns the stored embedding (NOTE: This is the FULL augmented embedding)"""
        return self.embedding_