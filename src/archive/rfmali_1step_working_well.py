import sys
import numpy as np
from rfgap import RFGAP
import scipy
from sklearn import preprocessing
import graphtools
from scipy import sparse
from scipy.spatial.distance import cdist

# embedders
from sklearn.manifold import SpectralEmbedding
from rfphate import PageRankPHATE
from umap import UMAP

# utils
from utils.utils import kernel2Dist

#OT solvers
from ..hiref import HiRef_fast as HiRef
from ..hiref import rank_annealing


class RFMALI(object):
    def __init__(self,
                 mu=0.5,
                 dpt=False,
                 embedder='phate',
                 n_components=2,
                 knn_dist='precomputed_affinity',   ### CHANGED: match class 2 API
                 verbose=0,
                 random_state=None,
                 n_jobs=-1):
        self.mu = mu
        self.dpt = dpt
        self.n_components = n_components
        self.knn_dist = knn_dist              ### CHANGED
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.embedder = embedder

        self.rfgap_params = {
            'random_state': random_state,
            'prediction_type': 'classification',
            'prox_method': 'rfgap',
            'model_type': 'rf',
            'oob_score': False,
            'non_zero_diagonal': True,
            'force_symmetric': True,
            'verbose': 0,
            'n_jobs': -1,
        }

        self.T_sparse = None
        self.W_combined = None
        self.embedding_ = None
        self.classes_ = None
        self.n = None    

    
    def _unlabeled_mask(self, y: np.ndarray) -> np.ndarray:
        """
        Robust unlabeled mask: True for NaN/None OR for sentinel -1.
        Works for int/float/object arrays.
        """
        y = np.asarray(y).ravel()
    
        # Start with all-False
        mask = np.zeros(y.shape[0], dtype=bool)
    
        # Sentinel -1 (works for ints, floats, and many object arrays)
        try:
            mask |= (y == -1)
        except Exception:
            pass
    
        # NaN / None (robust)
        # Numeric path
        if np.issubdtype(y.dtype, np.number):
            mask |= np.isnan(y)
        else:
            # Object path (None, np.nan, etc.)
            for i, v in enumerate(y):
                if v is None:
                    mask[i] = True
                else:
                    try:
                        # np.nan != np.nan is True
                        if v != v:
                            mask[i] = True
                    except Exception:
                        pass
    
        return mask

    def _get_rfgap_posteriors(self, prox, y, labels):
        n_samples = prox.shape[0]
        n_classes = len(labels)
        label_to_col = {lab: i for i, lab in enumerate(labels)}
    
        y = np.asarray(y).ravel()
        mask_unl = self._unlabeled_mask(y)
    
        posteriors = np.zeros((n_samples, n_classes), dtype=float)
    
        for lab in labels:
            col_idx = label_to_col[lab]
    
            class_mask = (~mask_unl) & (y == lab)
            count = int(class_mask.sum())
            if count == 0:
                continue
    
            if sparse.issparse(prox):
                vec = prox[:, class_mask].sum(axis=1).A.ravel()
            else:
                vec = prox[:, class_mask].sum(axis=1)
    
            posteriors[:, col_idx] = vec / count
    
        posteriors = preprocessing.normalize(posteriors, norm='l1', axis=1)
        # posteriors = np.sqrt(posteriors) / np.sqrt(2)
        return posteriors

    def fit(self, x_a, y_a, x_b, y_b):
        """
        Fit the RFMALI model.
        """
        self.n = x_a.shape[0]

        # --- IMPORTANT CHANGE ---
        # Do NOT force dtype=float. 
        # If y_a is int (fully labeled), keep it int so RFGAP runs in Classification mode.
        # If y_a has NaNs, it is already float/object, so we leave it as is.
        y_a = np.array(y_a).ravel()
        y_b = np.array(y_b).ravel()

        print("Fitting RFGAP on Domain A...")
        self.rfgap_a = RFGAP(**self.rfgap_params)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities()
        
        print("Fitting RFGAP on Domain B...")
        self.rfgap_b = RFGAP(**self.rfgap_params)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities()
        
        print("Building Posteriors...")
        
        # Determine labels safely
        # 1. Filter out NaNs/-1 (if any)
        # 2. Use np.unique to sort and find distinct classes/same ordering across
        # Determine labels in BOTH domains (ignore unlabeled)
        mask_unl_a = self._unlabeled_mask(y_a)
        mask_unl_b = self._unlabeled_mask(y_b)
        labels_a = np.unique(y_a[~mask_unl_a])
        labels_b = np.unique(y_b[~mask_unl_b])
        # Enforce shared semantic space: identical label sets
        set_a, set_b = set(labels_a.tolist()), set(labels_b.tolist())
        if set_a != set_b:
            only_a = sorted(set_a - set_b)
            only_b = sorted(set_b - set_a)
            raise ValueError(
                "Domain label mismatch (shared semantic space violated).\n"
                f"  Labels only in domain A: {only_a}\n"
                f"  Labels only in domain B: {only_b}\n"
                f"  Labels in A: {sorted(set_a)}\n"
                f"  Labels in B: {sorted(set_b)}\n"
                "Tip: check for missing classes after masking unlabeled (-1/NaN) in either domain."
            )
        # Use a single canonical ordering (sorted) for consistent posterior columns
        labels = np.array(sorted(set_a), dtype=labels_a.dtype)
        # (optional) store for downstream sanity checks
        self.classes_ = labels

        post_a = self._get_rfgap_posteriors(prox_a, y_a, labels)
        post_b = self._get_rfgap_posteriors(prox_b, y_b, labels)

        # Compute Optimal Transport with HiRef
        print("Computing Optimal Transport...")
        rank_schedule = rank_annealing.optimal_rank_schedule(n=self.n)
        frontier = HiRef.hiref_lr_fast(post_a, post_b, rank_schedule=rank_schedule)
        print(frontier)

        row_idx = np.array([int(p[0][0]) for p in frontier], dtype=np.int64)
        col_idx = np.array([int(p[1][0]) for p in frontier], dtype=np.int64)
        data = np.ones_like(row_idx, dtype=float)
        self.T_sparse = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.n, self.n))

        # Fusion
        print("Building joint affinity matrix...")

        # Off-diagonal blocks
        W_ab = (prox_a.dot(self.T_sparse) + self.T_sparse.dot(prox_b)) / 2
        W_ba = W_ab.T

        self.W_combined = sparse.bmat(
            [
                [self.mu * prox_a, (1-self.mu) * W_ab],
                [(1-self.mu) * W_ba, self.mu * prox_b]
            ],
            format="csr"
        )
        
        print("Model fit complete.")
        return self

    def fit_transform(self, x_a, y_a, x_b, y_b):
        self.fit(x_a, y_a, x_b, y_b)
    
        if self.embedder == 'phate':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                knn_dist=self.knn_dist,
                random_state=self.random_state,
                verbose=self.verbose,
                n_jobs=-1,
                beta=0.5,
            )
            self.embedding_ = phate_op.fit_transform(self.W_combined)
            return self.embedding_
    
        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.embedding_ = embedder.fit_transform(self.W_combined)
            return self.embedding_
    
        elif self.embedder == 'umap':
            # DistM = kernel2Dist(self.W_combined.toarray())
            # self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed').fit_transform(DistM)
            self.embedding_ = UMAP(n_components=self.n_components).fit_transform(self.W_combined)
            return self.embedding_
    
        else:
            raise ValueError(f"Unknown embedder={self.embedder}")
    
    def get_embeddings(self):
        return self.embedding_