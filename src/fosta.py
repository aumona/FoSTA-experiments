import numpy as np
from sklearn import preprocessing
from scipy import sparse

from .rfgap.forestkernel import ForestKernel
from src.phate import PageRankPHATE
from umap import UMAP

from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils

from .hiref.adaptive_HiRef import solve_surjection_hiref


class FoSTA:
    """
    FoSTA: Forest-guided Semantic Transport Alignment.
    Old-version path only.
    """

    def __init__(
        self,
        mu='auto',
        kernel_method="gap",
        force_nonzero_diag=True,
        force_symmetric=True,
        normalize_diagonal=True,
        model_type="rf",
        n_estimators=1000,
        t="auto",
        beta=0.7,
        prior_correct=True,
        l2_normalize=True,
        embedder="PHATE",
        n_components=2,
        verbose=1,
        random_state=None,
        n_jobs=-1,
    ):
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.kernel_method = kernel_method
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

        self.prior_correct = prior_correct
        self.l2_normalize = l2_normalize

        self.mu = mu
        self.t = t
        self.beta = beta
        self.embedder = embedder
        self.n_components = n_components

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

        # ===== OLD FoSTA =====
        self._log(f"[Domain {domain_name}] Computing FoSTA kernel (full NxN)...")

        prox = kernel.get_kernel(
            normalize_diagonal=self.normalize_diagonal,
            force_symmetric=self.force_symmetric,
        ).tocsr()
        prox.data = np.maximum(prox.data, 0)

        self._log(f"[Domain {domain_name}] Shapes: prox={prox.shape}")

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

    def _compute_coupling(self, post_a, post_b):
        return solve_surjection_hiref(
            post_a,
            post_b,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_balanced_affinity(self, prox_a, prox_b, T):
        """
        Constructs a joint affinity matrix using raw surjective T,
        followed by optional nonzero-edge mean cross-block scaling.
        """
        W_ab_raw = (prox_a.dot(T) + T.dot(prox_b)) / 2
    
        if self.mu == "auto":
            intra_mean = 0.5 * (prox_a.data.mean() + prox_b.data.mean())
            cross_mean = W_ab_raw.data.mean() if W_ab_raw.nnz > 0 else 1.0
            mu_eff = intra_mean / max(cross_mean, 1e-12)
        else:
            mu_eff = float(self.mu)
    
        W_ab = mu_eff * W_ab_raw
        W_ba = W_ab.transpose()
    
        self._log(f"[FoSTA] effective mu={mu_eff:.4f}")
    
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

        self.kernel_a, self.prox_a = self._compute_domain_geometry(x_a, y_a, "A")
        self.kernel_b, self.prox_b = self._compute_domain_geometry(x_b, y_b, "B")

        self.post_a = self._get_semantic_vectors_from_prox(
            self.prox_a,
            y_a,
            labels,
            domain_name="A",
        )

        self.post_b = self._get_semantic_vectors_from_prox(
            self.prox_b,
            y_b,
            labels,
            domain_name="B",
        )

        self.T_sparse = self._compute_coupling(self.post_a, self.post_b)
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