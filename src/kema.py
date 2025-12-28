import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator, lobpcg
from sklearn.base import BaseEstimator, TransformerMixin
import graphtools
import gc

class KEMA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, mu=0.5, reg=1e-3,
                 n_neighbors=10, n_pca=None, decay=40, knn_dist='euclidean',
                 n_jobs=1, verbose=True, max_iter=100, tol=1e-5,
                 kernel='rbf', unlabeled_value=0, random_state=None):

        self.n_components = n_components
        self.mu = mu
        self.reg = reg
        self.n_neighbors = n_neighbors
        self.n_pca = n_pca
        self.decay = decay
        self.knn_dist = knn_dist
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol

        self.kernel = kernel
        self.unlabeled_value = unlabeled_value
        self.random_state = random_state

    def fit(self, X1, X2, y1, y2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        y1, y2 = np.asarray(y1).ravel(), np.asarray(y2).ravel()
        self.n1, self.n2 = X1.shape[0], X2.shape[0]
        n_total = self.n1 + self.n2

        rng = np.random.default_rng(self.random_state)

        # ---------------------------------------------------------
        # Build manifold operator M (always), and optionally K
        # ---------------------------------------------------------
        if self.verbose:
            print("Building Graphtools Kernels... (The skeleton of your data)")

        G1 = graphtools.Graph(X1, n_pca=self.n_pca, knn=self.n_neighbors,
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False)
        G2 = graphtools.Graph(X2, n_pca=self.n_pca, knn=self.n_neighbors,
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False)

        # Use graphtools kernel as the manifold operator (sparse)
        M = sparse.block_diag((G1.K, G2.K), format='csr')

        del G1, G2
        gc.collect()

        # degrees / total weight
        D_M_vec = np.asarray(M.sum(axis=1)).ravel().astype(np.float64, copy=False)
        Sw = float(D_M_vec.sum())

        # In rbf mode, K := M (your original meaning)
        if self.kernel == 'rbf':
            K = M
            D_K_vec = D_M_vec
        elif self.kernel == 'lin':
            # no K projection in linear mode
            K = None
            D_K_vec = None
            d1, d2 = X1.shape[1], X2.shape[1]
            self.d1_ = d1
            self.d2_ = d2
            d_total = d1 + d2
        else:
            raise ValueError("kernel must be 'rbf' or 'lin'")

        # ---------------------------------------------------------
        # Semantic Logic
        # ---------------------------------------------------------
        if self.verbose:
            print("Preparing Sparse Logic... (Mapping the labels)")

        y_all = np.concatenate([y1, y2])
        is_labeled = (y_all != self.unlabeled_value)
        labeled_indices = np.where(is_labeled)[0]
        num_labeled = int(len(labeled_indices))
        classes = np.unique(y_all[is_labeled]) if num_labeled > 0 else np.array([])

        data, row_ind, col_ind = [], [], []
        class_map = {c: i for i, c in enumerate(classes)}
        for idx in labeled_indices:
            row_ind.append(idx)
            col_ind.append(class_map[y_all[idx]])
            data.append(1.0)

        H = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_total, len(classes)), dtype=np.float64)

        V_lab = sparse.csr_matrix(
            (np.ones(num_labeled, dtype=np.float64),
             (labeled_indices, np.zeros(num_labeled, dtype=int))),
            shape=(n_total, 1),
            dtype=np.float64
        )

        class_counts = np.asarray(H.sum(axis=0)).ravel().astype(np.float64, copy=False)
        Sws = float(np.sum(class_counts ** 2) + n_total)
        Swd = float((num_labeled ** 2) - np.sum(class_counts ** 2) + n_total)

        s_s = Sw / Sws if Sws > 0 else 0.0
        s_d = Sw / Swd if Swd > 0 else 0.0

        # CRITICAL: ensure h_counts is a 1D ndarray, not a matrix
        h_counts = np.asarray(H @ class_counts).ravel().astype(np.float64, copy=False)

        Ds_vec = (s_s * (h_counts + 1.0)).astype(np.float64, copy=False)

        # Avoid todense() returning numpy.matrix — force ndarray
        v_lab_flat = np.asarray(V_lab.sum(axis=1)).ravel().astype(np.float64, copy=False)
        Dd_vec = (s_d * ((v_lab_flat * num_labeled) - h_counts + 1.0)).astype(np.float64, copy=False)

        # ---------------------------------------------------------
        # Define inner operators (in sample space)
        # ---------------------------------------------------------
        def A_inner(u_col):
            # u_col: (N,1)
            Lu = (D_M_vec[:, None] * u_col) - (M @ u_col)

            H_u = H.T @ u_col
            HH_u = H @ H_u
            Lsu = (Ds_vec[:, None] * u_col) - s_s * HH_u - s_s * u_col

            return (1.0 - self.mu) * Lu + self.mu * Lsu

        def B_inner(u_col):
            term1 = Dd_vec[:, None] * u_col
            term2 = s_d * (V_lab @ (V_lab.T @ u_col))
            term3 = s_d * (H @ (H.T @ u_col))
            term4 = s_d * u_col
            return term1 - term2 + term3 - term4

        # ---------------------------------------------------------
        # Matrix-free generalized eigensystem
        # ---------------------------------------------------------
        if self.kernel == 'rbf':
            # original dual form: (K * A * K) v
            def matvec_KAK(v):
                v_col = np.asarray(v, dtype=np.float64).reshape(-1, 1)
                u = K @ v_col
                z = A_inner(u)
                out = (K @ z) + (self.reg * v_col)
                return out.ravel()

            def matvec_KBK(v):
                v_col = np.asarray(v, dtype=np.float64).reshape(-1, 1)
                u = K @ v_col
                z = B_inner(u)
                out = (K @ z) + (self.reg * v_col)
                return out.ravel()

            LO_A = LinearOperator((n_total, n_total), matvec=matvec_KAK, dtype=np.float64)
            LO_B = LinearOperator((n_total, n_total), matvec=matvec_KBK, dtype=np.float64)

            if self.verbose:
                print("Solving Matrix-Free Eigensystem... (rbf kernel mode)")

            X_init = rng.random((n_total, self.n_components))
            vals, vecs = lobpcg(LO_A, X_init, B=LO_B, tol=self.tol,
                                maxiter=self.max_iter, largest=False)

            self.alphas_ = vecs

            if self.verbose:
                print("Projecting Embedding... (rbf kernel mode)")
            self.embedding_ = np.vstack([
                K[:self.n1, :self.n1] @ self.alphas_[:self.n1],
                K[self.n1:, self.n1:] @ self.alphas_[self.n1:]
            ])
            return self

        # ---- linear/primal: (Z^T A Z) w, (Z^T B Z) w
        def Z_mv(w):
            w = np.asarray(w, dtype=np.float64).ravel()
            w1 = w[:self.d1_]
            w2 = w[self.d1_:]
            out = np.zeros((n_total,), dtype=np.float64)
            out[:self.n1] = X1 @ w1
            out[self.n1:] = X2 @ w2
            return out.reshape(-1, 1)

        def Zt_mv(u_col):
            u = np.asarray(u_col, dtype=np.float64).ravel()
            out = np.zeros((d_total,), dtype=np.float64)
            out[:self.d1_] = X1.T @ u[:self.n1]
            out[self.d1_:] = X2.T @ u[self.n1:]
            return out

        def matvec_XAX(w):
            w = np.asarray(w, dtype=np.float64).ravel()
            u = Z_mv(w)                 # (N,1)
            z = A_inner(u)              # (N,1)
            out = Zt_mv(z)              # (D,)
            return out + self.reg * w

        def matvec_XBX(w):
            w = np.asarray(w, dtype=np.float64).ravel()
            u = Z_mv(w)
            z = B_inner(u)
            out = Zt_mv(z)
            return out + self.reg * w

        LO_A = LinearOperator((d_total, d_total), matvec=matvec_XAX, dtype=np.float64)
        LO_B = LinearOperator((d_total, d_total), matvec=matvec_XBX, dtype=np.float64)

        if self.verbose:
            print("Solving Matrix-Free Eigensystem... (linear SSMA mode)")

        W_init = rng.random((d_total, self.n_components))
        vals, vecs = lobpcg(LO_A, W_init, B=LO_B, tol=self.tol,
                            maxiter=self.max_iter, largest=False)

        self.projectors_ = vecs  # (D, p)

        if self.verbose:
            print("Projecting Embedding... (linear mode)")
        P1 = self.projectors_[:self.d1_, :]
        P2 = self.projectors_[self.d1_:, :]
        self.embedding_ = np.vstack([X1 @ P1, X2 @ P2])

        return self

    def fit_transform(self, X1, X2, y1, y2):
        self.fit(X1, X2, y1, y2)
        return self.embedding_