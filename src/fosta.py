import numpy as np
from sklearn import preprocessing
from scipy import sparse

from .rfgap.forestkernel import ForestKernel
from src.phate import PageRankPHATE
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref

# for dense OT solver (MALI-style)
from scipy.spatial.distance import cdist
import ot


class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment.
    Old-version path only.
    """

    def __init__(
        self,
        mu=1,

        kernel_method="gap",
        force_nonzero_diag=True,
        force_symmetric=True,
        normalize_diagonal=True,
        model_type="rf",
        n_estimators=1000,
        class_weight=None,
        bootstrap=True,

        t="auto",
        beta=0.7,
        prior_correct=True,
        l2_normalize=True,
        embedder="PHATE",
        n_components=2,
        verbose=1,
        random_state=None,
        n_jobs=-1,

        # labeled shared samples        -> always coupled
        # labeled domain-specific       -> not coupled
        # unlabeled predicted shared    -> coupled
        # unlabeled predicted nonshared -> not coupled
        unlabeled_coupling="predict_shared",  # include, exclude, or predict_shared --> should we include unlabeled points in the coupling computation, and if so, should we predict which ones are shared based on the forest predictions?

        ot_solver="hiref",
        entR=0,
        m=1,
        distance="cosine",
    ):
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.kernel_method = kernel_method
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.bootstrap = True if (kernel_method in ["gap", "oob"] or unlabeled_coupling == "predict_shared") else bootstrap
        self.force_nonzero_diag = force_nonzero_diag
        self.force_symmetric = force_symmetric
        self.normalize_diagonal = normalize_diagonal

        self.kernel_params = {
            "random_state": random_state,
            "prediction_type": "classification",
            "oob_score": True,
            "n_estimators": self.n_estimators,
            "class_weight": self.class_weight,
            "kernel_method": self.kernel_method,
            "force_nonzero_diag": self.force_nonzero_diag,
            "model_type": self.model_type,
            "bootstrap": self.bootstrap,
        }

        self.prior_correct = prior_correct
        self.l2_normalize = l2_normalize

        self.mu = mu
        self.t = t
        self.beta = beta
        self.embedder = embedder
        self.n_components = n_components

        self.unlabeled_coupling = unlabeled_coupling

        self.ot_solver = ot_solver
        self.entR = entR
        self.m = m
        self.distance = distance
        self.Distances12 = None

        # State storage
        self.kernel_a = self.kernel_b = None
        self.prox_a = self.prox_b = None
        self.post_a = self.post_b = None
        self.T_sparse = self.W = self.embedding_ = None
        self.classes_ = self.n = self.n_a = self.n_b = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _compute_domain_geometry(self, x, y, domain_name="A"):
        y = np.asarray(y).ravel()
        unl_mask = LabelUtils.get_unlabeled_mask(y)
        idx_unl = np.flatnonzero(unl_mask)

        self._log(f"\n[Domain {domain_name}] Fitting forest on labeled points...")
        kernel = ForestKernel(**self.kernel_params)

        # ------------------------------------------------------------
        # Fit transductive forest kernel.
        # ------------------------------------------------------------
        kernel.fit(x, y, idx_unlabeled=idx_unl)

        # get oob classification error for sanity check
        if hasattr(kernel.forest_, "oob_score_"):
            self._log(f"[Domain {domain_name}] OOB classification acc.: {kernel.forest_.oob_score_:.4f}")

        self._log(f"[Domain {domain_name}] Computing FoSTA kernel (full NxN)...")

        prox = kernel.get_kernel(
            normalize_diagonal=self.normalize_diagonal,
            force_symmetric=self.force_symmetric,
        ).tocsr()
        prox.data = np.maximum(prox.data, 0)

        self._log(f"[Domain {domain_name}] Shapes: prox={prox.shape} | NNZ density={prox.nnz / (prox.shape[0] * prox.shape[1]):.6f}")

        return kernel, prox

    def _get_semantic_vectors_from_prox(self, prox, y, labels, domain_name="A"):
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

        post = prox.dot(Y)
        post = np.asarray(post, dtype=float)

        row_sums = post.sum(axis=1)
        self._log(
            f"[Domain {domain_name}] Semantic row sums before prior correction and l2 normalization: "
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
    

    def _compute_dense_ot(self, post_a, post_b):
        self._log("Computing dense OT...")
    
        X = np.asarray(post_a, dtype=float).copy()
        Y = np.asarray(post_b, dtype=float).copy()
    
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        Y = np.nan_to_num(Y, nan=0.0, posinf=1e6, neginf=-1e6)
    
        eps = 1e-12
        if self.distance == "cosine":
            if X.shape[1] == 0 or Y.shape[1] == 0:
                raise ValueError("Semantic vectors have zero columns.")
    
            zero_x = np.linalg.norm(X, axis=1) < eps
            zero_y = np.linalg.norm(Y, axis=1) < eps
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
                b = np.repeat(1.0, N2)
                transport = "wot" if self.entR == 0 else "wotR"
            else:
                a = np.repeat(1.0 / N1, N1)
                b = np.repeat(1.0 / N2, N2)
                m_eff = np.floor(m_eff * N1) / N1
                transport = "wotpartial" if self.entR == 0 else "wotpartialR"
        else:
            if m_eff == 1:
                a = np.repeat(1.0, N1)
                b = np.repeat(N1 / N2, N2)
                transport = "wot" if self.entR == 0 else "wotR"
                self._log("Dense OT: unbalanced full transport.")
            else:
                a = np.repeat(1.0 / N1, N1)
                b = np.repeat(1.0 / N2, N2)
                m_eff = np.floor(m_eff * N1) / N1
                transport = "wotpartial" if self.entR == 0 else "wotpartialR"
    
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
            raise ValueError("Not implemented.")
    
        T[T < 1e-5] = 0
        return sparse.csr_matrix(T)
    
    
    def _compute_coupling(self, post_a, post_b):
        if self.ot_solver == "hiref":
            return solve_surjection_hiref(
                post_a.astype(np.float32),
                post_b.astype(np.float32),
                verbose=self.verbose,
                random_state=self.random_state,
            )
    
        if self.ot_solver == "dense":
            return self._compute_dense_ot(post_a, post_b)
    
        raise ValueError(f"Unknown ot_solver={self.ot_solver!r}. Use 'hiref' or 'dense'.")

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix using raw surjective T,
        followed by optional nonzero-edge mean cross-block scaling.
        """
        W_ab = prox_a.dot(T)
        W_ba = prox_b.dot(T.transpose())
    
        if self.verbose:
            print("\nJOINT AFFINITY BLOCK STATISTICS")
            print("------------------------------")
            print_mat_stats("Within-domain A (W1)", prox_a)
            print_mat_stats("Within-domain B (W2)", prox_b)
            print_mat_stats("Cross-domain A→B (W12)", W_ab)
            print_mat_stats("Cross-domain B→A (W21)", W_ba)
    
        return sparse.bmat(
            [
                [prox_a, self.mu * W_ab],
                [self.mu * W_ba, prox_b],
            ],
            format="csr",
        )

    def fit(self, x_a, x_b, y_a, y_b):
        """Fits FoSTA alignment across two domains."""
        self.n_a, self.n_b = x_a.shape[0], x_b.shape[0]
        self.n = self.n_a + self.n_b
    
        labels_a = LabelUtils.get_valid_classes(y_a)
        labels_b = LabelUtils.get_valid_classes(y_b)
    
        shared_labels = np.intersect1d(labels_a, labels_b)
        all_labels = np.union1d(labels_a, labels_b)
    
        if shared_labels.size == 0:
            raise ValueError("FoSTA requires at least one shared labeled class across domains.")
    
        self.classes_ = all_labels
        self.shared_classes_ = shared_labels
    
        self.kernel_a, self.prox_a = self._compute_domain_geometry(x_a, y_a, "A")
        self.kernel_b, self.prox_b = self._compute_domain_geometry(x_b, y_b, "B")
    
        self.post_a = self._get_semantic_vectors_from_prox(
            self.prox_a,
            y_a,
            all_labels,
            domain_name="A",
        )
    
        self.post_b = self._get_semantic_vectors_from_prox(
            self.prox_b,
            y_b,
            all_labels,
            domain_name="B",
        )
    
        y_a_arr = np.asarray(y_a).ravel()
        y_b_arr = np.asarray(y_b).ravel()
    
        mask_unl_a = LabelUtils.get_unlabeled_mask(y_a_arr)
        mask_unl_b = LabelUtils.get_unlabeled_mask(y_b_arr)
    
        if self.unlabeled_coupling == "exclude":
            idx_shared_a = np.flatnonzero((~mask_unl_a) & np.isin(y_a_arr, shared_labels))
            idx_shared_b = np.flatnonzero((~mask_unl_b) & np.isin(y_b_arr, shared_labels))
        
        elif self.unlabeled_coupling == "include":
            idx_shared_a = np.flatnonzero(mask_unl_a | np.isin(y_a_arr, shared_labels))
            idx_shared_b = np.flatnonzero(mask_unl_b | np.isin(y_b_arr, shared_labels))
        
        elif self.unlabeled_coupling == "predict_shared":
            pred_a = self.kernel_a.forest_.predict(x_a)
            pred_b = self.kernel_b.forest_.predict(x_b)
        
            idx_shared_a = np.flatnonzero(
                ((~mask_unl_a) & np.isin(y_a_arr, shared_labels))
                | (mask_unl_a & np.isin(pred_a, shared_labels))
            )
        
            idx_shared_b = np.flatnonzero(
                ((~mask_unl_b) & np.isin(y_b_arr, shared_labels))
                | (mask_unl_b & np.isin(pred_b, shared_labels))
            )
        
        else:
            raise ValueError(
                "unlabeled_coupling must be one of "
                "{'exclude', 'include', 'predict_shared'}."
            )
    
        if idx_shared_a.size == 0 or idx_shared_b.size == 0:
            raise ValueError(
                "FoSTA requires at least one labeled sample from the shared label set "
                "in each domain."
            )
    
        shared_mask = np.isin(all_labels, shared_labels)
    
        self._log(
            f"[FoSTA] Coupling selected points: "
            f"A={idx_shared_a.size}/{self.n_a}, B={idx_shared_b.size}/{self.n_b}, "
            f"mode={self.unlabeled_coupling}"
        )
    
        T_sub = self._compute_coupling(
            self.post_a[idx_shared_a][:, shared_mask],
            self.post_b[idx_shared_b][:, shared_mask],
        ).tocsr()
    
        rows, cols = T_sub.nonzero()
        vals = np.asarray(T_sub[rows, cols]).ravel()
    
        self.T_sparse = sparse.coo_matrix(
            (
                vals,
                (idx_shared_a[rows], idx_shared_b[cols]),
            ),
            shape=(self.n_a, self.n_b),
        ).tocsr()
    
        self.W = self._build_balanced_affinity(
            self.prox_a,
            self.prox_b,
            self.T_sparse,
        )
    
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