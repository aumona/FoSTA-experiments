import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from umap import UMAP

from src.Pamona.Pamona import Pamona as Pamona_original


class Pamona(Pamona_original):
    """
    Robust wrapper around the original Pamona implementation with:

    - Input sanitization (NaN/inf handling, constant feature checks)
    - Optional semi-supervised alignment via a label prior matrix
    - Numerical safeguards for GW and projection steps
    - Optional low-dimensional embedding (UMAP / PCA)

    This wrapper is designed for general tabular benchmarks where:
    - Data may be noisy or partially corrupted
    - Graph-based geodesic distances may be unstable
    - Semi-supervision is optional and controlled via `gamma`

    Parameters
    ----------
    gamma : float, default=0.5
        Strength of the label prior. 
        - gamma = 0.0 → fully unsupervised
        - gamma > 0 → encourages matching samples with same labels

    n_components : int, default=2
        Output dimension for the final embedding (if embedder is used).

    embedder : {"UMAP", "PCA", None}, default="UMAP"
        Post-alignment dimensionality reduction applied to the joint embedding.

    random_state : int or None
        Random seed used for the embedder.

    **kwargs :
        Passed directly to the original Pamona class.
    """

    def __init__(
        self,
        gamma=0.5,
        n_components=2,
        embedder="UMAP",
        random_state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Semi-supervision ---
        self.gamma = gamma

        # --- Output embedding ---
        self.n_components = n_components
        self.embedder = embedder
        self.manual_seed = random_state

        # --- Cached inputs ---
        self.x_a_ = None
        self.x_b_ = None
        self.y_a_ = None
        self.y_b_ = None

        # --- Outputs ---
        self.integrated_data = None
        self.embedding_ = None
        self.T = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_safe_array(x, name="array"):
        """
        Convert input to a safe dense float array.

        - Leaves sparse inputs untouched
        - Replaces NaN / ±inf with finite values
        - Ensures fully finite output

        This prevents failures in:
        - kNN graph construction
        - geodesic distance computation
        - GW optimization
        """
        if issparse(x):
            return x

        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if not np.isfinite(x).all():
            raise ValueError(f"{name} still contains non-finite values after sanitization.")

        return x

    @staticmethod
    def _to_safe_labels(y):
        """
        Ensure labels are a 1D numpy array.
        """
        return np.asarray(y).ravel()

    @staticmethod
    def _check_finite_dense(x, name="array"):
        """
        Validate that a dense array contains only finite values.
        """
        x = np.asarray(x)
        if not np.isfinite(x).all():
            n_nan = np.isnan(x).sum()
            n_inf = np.isinf(x).sum()
            raise ValueError(f"{name} contains invalid values: NaN={n_nan}, inf={n_inf}")

    @staticmethod
    def _sanitize_embedding(x, name="embedding", tol=1000):
        """
        Clean embedding output:

        - Removes small imaginary parts from eig/SVD computations
        - Clips NaN / inf
        - Ensures real-valued finite output

        This is especially important after:
        - eigen decompositions (project_func)
        - SVD in alignment
        """
        x = np.asarray(x)

        if np.iscomplexobj(x):
            x = np.real_if_close(x, tol=tol)
            if np.iscomplexobj(x):
                max_imag = np.max(np.abs(np.imag(x)))
                if max_imag < 1e-8:
                    x = np.real(x)
                else:
                    raise ValueError(
                        f"{name} has non-negligible imaginary part "
                        f"(max |Im| = {max_imag:.3e})"
                    )

        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if not np.isfinite(x).all():
            raise ValueError(f"{name} contains NaN/inf after sanitization.")

        return x

    @staticmethod
    def _ensure_not_constant(x, name="array", eps=1e-12):
        """
        Detect degenerate inputs where all features are constant.

        Such inputs break:
        - distance computations
        - graph construction
        - eigen decompositions
        """
        if issparse(x):
            return

        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            return

        col_std = np.std(x, axis=0)
        if np.all(col_std < eps):
            raise ValueError(f"{name} is numerically constant across all features.")

    def _build_prior_matrix(self, y_a, y_b):
        """
        Construct the semi-supervised prior matrix M.

        M[i, j] = 1           → no penalty
        M[i, j] = 1 - gamma   → encourage matching labels

        Only valid (non-missing) labels are used.
        Missing labels (-1 or NaN) are ignored.

        This matrix modulates the GW cost:
        - lower values encourage alignment
        - acts as a soft constraint
        """
        n1, n2 = len(y_a), len(y_b)
        M_prior = np.ones((n1, n2), dtype=float)

        valid_a = (y_a != -1)
        valid_b = (y_b != -1)

        if np.issubdtype(y_a.dtype, np.floating):
            valid_a &= ~np.isnan(y_a)
        if np.issubdtype(y_b.dtype, np.floating):
            valid_b &= ~np.isnan(y_b)

        matches = (y_a[:, None] == y_b[None, :])
        matches &= valid_a[:, None]
        matches &= valid_b[None, :]

        M_prior[matches] = 1.0 - self.gamma
        self._check_finite_dense(M_prior, name="M_prior")
        return M_prior

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def fit(self, x_a, x_b, y_a, y_b):
        """
        Prepare sanitized inputs and optional prior matrix.

        This stage does NOT run Pamona yet.
        It only:
        - validates inputs
        - stores cleaned data
        - builds the semi-supervised prior (if enabled)
        """
        x_a = self._to_safe_array(x_a, name="x_a")
        x_b = self._to_safe_array(x_b, name="x_b")

        self._ensure_not_constant(x_a, name="x_a")
        self._ensure_not_constant(x_b, name="x_b")

        y_a = self._to_safe_labels(y_a) if y_a is not None else None
        y_b = self._to_safe_labels(y_b) if y_b is not None else None

        self.x_a_ = x_a
        self.x_b_ = x_b
        self.y_a_ = y_a
        self.y_b_ = y_b

        if self.gamma > 0 and y_a is not None and y_b is not None:
            self.M = self._build_prior_matrix(y_a, y_b)
        else:
            self.M = None

        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """
        Run full Pamona pipeline:

        1. Fit (sanitize + prior)
        2. Run GW-based alignment
        3. Project to common space
        4. Optionally reduce dimension (UMAP / PCA)

        Returns
        -------
        embedding : ndarray of shape (n_samples_total, n_components)
        """
        self.fit(x_a, x_b, y_a, y_b)

        try:
            integrated_data, T = self.run_Pamona([self.x_a_, self.x_b_])

        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Pamona failed during internal linear algebra. "
                "This usually indicates a numerically degenerate matrix."
            ) from e

        except FloatingPointError as e:
            raise ValueError(
                "Pamona failed due to a floating-point numerical issue."
            ) from e

        except AttributeError as e:
            raise ValueError(
                "Pamona failed due to an internal state error. "
                "This often happens when alignment succeeds but projection fails."
            ) from e

        if integrated_data is None:
            raise ValueError("Pamona returned None for integrated_data.")

        if not isinstance(integrated_data, (list, tuple)):
            raise ValueError("Pamona returned an invalid integrated_data object.")

        if len(integrated_data) != 2:
            raise ValueError("Pamona returned integrated_data with unexpected length.")

        a_int = self._sanitize_embedding(integrated_data[0], name="integrated_data[0]")
        b_int = self._sanitize_embedding(integrated_data[1], name="integrated_data[1]")

        self.integrated_data = [a_int, b_int]
        self.T = T

        joint = np.vstack(self.integrated_data)
        self._ensure_not_constant(joint, name="joint integrated data")

        # --- Optional embedding ---
        if self.embedder == "UMAP":
            self.embedding_ = UMAP(
                n_components=self.n_components,
                n_neighbors=20,
                min_dist=0.7,
                random_state=self.manual_seed,
            ).fit_transform(joint)

            self.embedding_ = self._sanitize_embedding(self.embedding_, name="UMAP embedding")
            return self.embedding_

        if self.embedder == "PCA":
            self.embedding_ = PCA(n_components=self.n_components).fit_transform(joint)
            self.embedding_ = self._sanitize_embedding(self.embedding_, name="PCA embedding")
            return self.embedding_

        # --- No embedding ---
        self.embedding_ = joint
        return self.embedding_