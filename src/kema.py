import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator, lobpcg
from sklearn.base import BaseEstimator, TransformerMixin
import graphtools
import gc

class KEMA(BaseEstimator, TransformerMixin):
    """
    KEMA: Kernel Manifold Alignment (The "Memory-Light" Edition)
    
    Traditional KEMA is a memory hog. It wants to build and invert massive N x N 
    matrices that crash your RAM the moment you have more than a few thousand points.
    
    This implementation is different. It's 'Matrix-Free.' 
    Instead of building the whole mountain, we define the physics of how to walk 
    on it. By using Scipy's LinearOperators, we solve the alignment problem purely 
    through matrix-vector multiplications. 
    
    Memory footprint? O(N). 
    Solver? LOBPCG (iterative and smart).
    Vibe? Scalable and efficient.
    """

    def __init__(self, n_components=2, mu=0.5, reg=1e-3, 
                 n_neighbors=10, n_pca=None, decay=40, knn_dist='euclidean',
                 n_jobs=1, verbose=True, max_iter=100, tol=1e-5):
        
        self.n_components = n_components
        self.mu = mu # The slider between "Respect the Shape" and "Align the Labels"
        self.reg = reg # Numerical 'glue' to keep the solver stable
        self.n_neighbors = n_neighbors
        self.n_pca = n_pca
        self.decay = decay
        self.knn_dist = knn_dist
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X1, X2, y1, y2):
        """
        Learns the shared latent space between Domain 1 and Domain 2.
        
        We're essentially trying to find a projection that keeps the local 
        geometry of each domain intact while forcing points with the same 
        labels to 'handshake' in the middle.
        """
        X1, X2 = np.asarray(X1), np.asarray(X2)
        y1, y2 = np.asarray(y1).flatten(), np.asarray(y2).flatten()
        self.n1, self.n2 = X1.shape[0], X2.shape[0]
        n_total = self.n1 + self.n2
        
        # --- 1. The Kernel Bridge ---
        # We build sparse kernels using graphtools. This captures the 'flow' 
        # of the data manifolds. We only keep the sparse matrix (K) to save RAM.
        if self.verbose: print("Building Graphtools Kernels... (The skeleton of your data)")
        
        G1 = graphtools.Graph(X1, n_pca=self.n_pca, knn=self.n_neighbors, 
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False)
        G2 = graphtools.Graph(X2, n_pca=self.n_pca, knn=self.n_neighbors, 
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False)
        
        # We combine them into one big diagonal block matrix. 
        # Points in Domain 1 don't 'see' Domain 2 yet—that's what the labels are for.
        K = sparse.block_diag((G1.K, G2.K), format='csr')
        
        # Housekeeping: Graph objects are heavy, so we toss them early.
        del G1, G2
        gc.collect()

        # --- 2. The Semantic Logic ---
        # H is our 'Who's Who' matrix. It tells us which points belong to which class.
        # V_lab is just a simple flag for points that actually have a label.
        if self.verbose: print("Preparing Sparse Logic... (Mapping the labels)")

        y_all = np.concatenate([y1, y2])
        is_labeled = (y_all != 0) # Assumes 0 is our 'unlabeled' placeholder
        labeled_indices = np.where(is_labeled)[0]
        num_labeled = len(labeled_indices)
        classes = np.unique(y_all[is_labeled])
        
        # Build H (N x Classes) as a sparse matrix. Even with 1M points, 
        # this is tiny because most rows are empty.
        data, row_ind, col_ind = [], [], []
        class_map = {c: i for i, c in enumerate(classes)}
        for idx in labeled_indices:
            row_ind.append(idx)
            col_ind.append(class_map[y_all[idx]])
            data.append(1.0)
        
        H = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_total, len(classes)))
        
        V_lab = sparse.csr_matrix((np.ones(num_labeled), 
                                  (labeled_indices, np.zeros(num_labeled))), 
                                  shape=(n_total, 1))

        # --- 3. Precomputing Manifold 'Weights' ---
        # We need the row sums of our matrices to build Laplacians.
        # Doing this once saves us from repeating it a million times in the solver.
        D_K_vec = np.array(K.sum(axis=1)).flatten() 
        Sw = D_K_vec.sum()
        
        class_counts = np.array(H.sum(axis=0)).flatten()
        Sws = np.sum(class_counts**2) + n_total
        Swd = (num_labeled**2) - np.sum(class_counts**2) + n_total
        
        s_s = Sw / Sws if Sws > 0 else 0.0 # Scaling for 'Similarity'
        s_d = Sw / Swd if Swd > 0 else 0.0 # Scaling for 'Dissimilarity'
        
        # Pre-calculating these diagonals makes our matvec functions very lean.
        h_counts = H @ class_counts 
        Ds_vec = s_s * (h_counts + 1.0)
        
        v_lab_flat = np.array(V_lab.todense()).flatten()
        Dd_vec = s_d * ((v_lab_flat * num_labeled) - h_counts + 1.0)

        # --- 4. The Magic: Matrix-Free Operators ---
        # Here's the trick: We define the RESULT of a matrix-vector product
        # without ever forming the matrix itself. This is the O(N) secret sauce.

        def matvec_KAK(v):
            """ Computes (K * A_inner * K + reg*I) * v """
            # Projection 1: Drop it into kernel space
            u = K @ v 
            
            # The 'Inner' geometry: Local shape (L) + Semantic Pull (Ls)
            Lu = (D_K_vec[:, None] * u) - (K @ u)
            
            H_u = H.T @ u
            HH_u = H @ H_u
            Lsu = (Ds_vec[:, None] * u) - s_s * HH_u - s_s * u
            
            z = (1 - self.mu) * Lu + self.mu * Lsu
            
            # Projection 2: Back to alpha space + small regularization stabilizer
            return (K @ z) + (self.reg * v)

        def matvec_KBK(v):
            """ Computes (K * B_inner * K + reg*I) * v """
            u = K @ v
            
            # The 'Push' logic (Ld): Keeps different classes from overlapping
            term1 = Dd_vec[:, None] * u
            term2 = s_d * (V_lab @ (V_lab.T @ u))
            term3 = s_d * (H @ (H.T @ u))
            term4 = s_d * u
            
            z = term1 - term2 + term3 - term4
            
            return (K @ z) + (self.reg * v)

        # Wrap these functions as Scipy LinearOperators
        LO_A = LinearOperator((n_total, n_total), matvec=matvec_KAK, dtype=np.float64)
        LO_B = LinearOperator((n_total, n_total), matvec=matvec_KBK, dtype=np.float64)

        # --- 5. The Solver (LOBPCG) ---
        # We ask LOBPCG to find the directions that minimize the energy of our system.
        if self.verbose: print(f"Solving Matrix-Free Eigensystem... (Finding the 'sweet spot')")
        
        X_init = np.random.rand(n_total, self.n_components)
        
        # Solving the generalized eigenvalue problem: LO_A x = lambda LO_B x
        vals, vecs = lobpcg(LO_A, X_init, B=LO_B, tol=self.tol, maxiter=self.max_iter, largest=False)
        
        self.alphas_ = vecs
        
        # --- 6. Final Projection ---
        # We use the learned 'alphas' to project our kernels into the shared space.
        if self.verbose: print("Projecting Embedding... (Alignment complete!)")
        self.embedding_ = np.vstack([
            K[:self.n1, :self.n1] @ self.alphas_[:self.n1],
            K[self.n1:, self.n1:] @ self.alphas_[self.n1:]
        ])
        
        return self

    def fit_transform(self, X1, X2, y1, y2):
        """ Train the model and return the aligned data in one go. """
        self.fit(X1, X2, y1, y2)
        return self.embedding_