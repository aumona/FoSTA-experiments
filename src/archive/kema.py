import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from utils.labels import LabelUtils

class KEMA(BaseEstimator):
    """
    Python port of Devis Tuia's 'KMA.m'.
    Strictly reproduces the heuristics for matrix scaling and vector inversion.
    
    Includes support for Linear Manifold Alignment (Wang & Mahadevan, IJCAI 2011)
    via kernel='linear'.
    """
    def __init__(self, n_components=2, kernel='rbf', gamma=None, mu=0.5, 
                 n_neighbors=9, reg=1e-3):  # Default from original KEMA.m
        """
        Args:
            n_components (int): Dimensionality of the latent space.
            kernel (str): 'rbf', 'linear', 'poly', etc. 
                          If 'linear', solves the Primal problem (Wang 2011).
            gamma (float): Kernel coefficient for rbf/poly/sigmoid.
            mu (float): Trade-off parameter (0.0 to 1.0) between Geometry and Class Structure.
                        - mu -> 0.0: Prioritizes Manifold Geometry. The algorithm focuses on 
                          preserving the local neighborhood structure of the data (unsupervised).
                        - mu -> 1.0: Prioritizes Class Discrimination. The algorithm focuses on 
                          pulling samples of the same class together (supervised, LDA-like).
                        - mu = 0.5: Balanced approach.
            n_neighbors (int): Number of neighbors for the geometry graph (kNN).
            reg (float): Regularization added to the diagonal for numerical stability.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.mu = mu
        self.n_neighbors = n_neighbors
        self.reg = reg
       
    def fit(self, X1, X2, y1, y2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        y1, y2 = np.asarray(y1), np.asarray(y2)
        n1, n2 = X1.shape[0], X2.shape[0]
        n = n1 + n2
        
        # Determine mode: Linear (Primal) or Kernel (Dual)
        # Wang et al. 2011 uses the Primal form for linear alignment
        self.is_primal_ = (self.kernel == 'linear')
       
        # Kernels (Block Diagonal) OR Features (Block Diagonal)
        if self.is_primal_:
            # Primal: Work directly with Features X (Size D x D optimization)
            d1, d2 = X1.shape[1], X2.shape[1]
            self.d1_ = d1
            # Z = Block Diag(X1, X2) -- effectively handled by matrix mults below to save RAM
            # We don't compute kernels K here.
        else:
            # Dual: Work with Kernels K (Size N x N optimization)
            K1 = pairwise_kernels(X1, metric=self.kernel, gamma=self.gamma)
            K2 = pairwise_kernels(X2, metric=self.kernel, gamma=self.gamma)
            K = linalg.block_diag(K1, K2)
       
        # Geometry Graphs (G) ---
        W1 = kneighbors_graph(X1, self.n_neighbors, mode='connectivity', include_self=False).toarray()
        W2 = kneighbors_graph(X2, self.n_neighbors, mode='connectivity', include_self=False).toarray()
        W1 = np.maximum(W1, W1.T)
        W2 = np.maximum(W2, W2.T)
        W = linalg.block_diag(W1, W2)
       
        # Label Graphs (Ws, Wd) & SCALING
        y_all = np.concatenate([y1, y2])
        
        # --- Uniformized Label Handling ---
        mask_unl = LabelUtils.get_unlabeled_mask(y_all)
        is_labeled = ~mask_unl
        
        label_matrix = np.zeros((n, n))
        labeled_idx = np.where(is_labeled)[0]
       
        if len(labeled_idx) > 0:
            y_L = y_all[labeled_idx]
            Eq = (y_L[:, None] == y_L[None, :]).astype(float)
            idx_grid = np.ix_(labeled_idx, labeled_idx)
            label_matrix[idx_grid] = Eq
           
        Ws = label_matrix.copy()
        Wd = 1.0 - label_matrix
       
        # Mask unlabeled rows in Wd
        mask_mat = np.zeros((n,n))
        mask_mat[np.ix_(labeled_idx, labeled_idx)] = 1.0
        Wd = Wd * mask_mat
       
        # Tuia Heuristic: Add Eye & Scale
        Ws += np.eye(n)
        Wd += np.eye(n)
       
        Sw = W.sum()
        if Ws.sum() > 0: Ws = Ws / Ws.sum() * Sw
        if Wd.sum() > 0: Wd = Wd / Wd.sum() * Sw
       
        # Laplacians
        D = np.diag(W.sum(axis=1));   L = D - W
        Ds = np.diag(Ws.sum(axis=1)); Ls = Ds - Ws
        Dd = np.diag(Wd.sum(axis=1)); Ld = Dd - Wd
       
        # Solvers ---
        # A (Pull) vs B (Push)
        A_inner = ((1 - self.mu) * L + self.mu * Ls)
        B_inner = Ld
       
        if self.is_primal_:
            # Wang 2011: Solve Z' L Z v = lambda Z' D Z v
            # Z = [X1 0; 0 X2]
            # Calculating Z.T @ Matrix @ Z via blocks:
            # M = [M11 M12; M21 M22]
            # Z.T M Z = [X1.T M11 X1,  X1.T M12 X2; ...]
            
            # Slice Laplacians for block multiplication
            L_A11, L_A12 = A_inner[:n1, :n1], A_inner[:n1, n1:]
            L_A21, L_A22 = A_inner[n1:, :n1], A_inner[n1:, n1:]
            
            L_B11, L_B12 = B_inner[:n1, :n1], B_inner[:n1, n1:]
            L_B21, L_B22 = B_inner[n1:, :n1], B_inner[n1:, n1:]

            # Construct Primal Matrices (Size D_total x D_total)
            KAK_11 = X1.T @ L_A11 @ X1
            KAK_12 = X1.T @ L_A12 @ X2
            KAK_21 = X2.T @ L_A21 @ X1
            KAK_22 = X2.T @ L_A22 @ X2
            
            KBK_11 = X1.T @ L_B11 @ X1
            KBK_12 = X1.T @ L_B12 @ X2
            KBK_21 = X2.T @ L_B21 @ X1
            KBK_22 = X2.T @ L_B22 @ X2

            KAK = np.block([[KAK_11, KAK_12], [KAK_21, KAK_22]])
            KBK = np.block([[KBK_11, KBK_12], [KBK_21, KBK_22]])
            
            reg_dim = KAK.shape[0] # D1 + D2
        else:
            # KEMA (Dual): Solve K L K alpha = lambda K D K alpha
            KAK = K @ A_inner @ K
            KBK = K @ B_inner @ K
            reg_dim = n # N1 + N2
       
        KAK += self.reg * np.eye(reg_dim)
       
        # Symmetrize
        KAK = (KAK + KAK.T) / 2
        KBK = (KBK + KBK.T) / 2

        # regularize firmly once
        KAK += (self.reg + 1e-6) * np.eye(reg_dim) 
       
        # Solve for LARGEST eigenvalues of (Push, Pull)
        vals, vecs = linalg.eigh(KBK, KAK)

        # Ensure symmetry perfectly before solver (numerical noise can break eigh)
        KAK = (KAK + KAK.T) / 2
        KBK = (KBK + KBK.T) / 2
       
        vals, vecs = linalg.eigh(KBK, KAK)
           
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        
        # Store results
        if self.is_primal_:
            self.projectors_ = vecs[:, :self.n_components] # Size (D1+D2, p)
        else:
            self.alphas_ = vecs[:, :self.n_components] # Size (N, p)
            
        self.X1_fit_ = X1
        self.X2_fit_ = X2
        self.n1_ = n1
        
        # Uniformized class extraction
        self.classes_ = LabelUtils.get_valid_classes(y_all)
        
        self.y1_ = y1
        self.y2_ = y2
       
        # --- 6. Vector Inversion Check ---
        # Pass K matrices only if Dual, otherwise pass None
        K1_arg = K1 if not self.is_primal_ else None
        K2_arg = K2 if not self.is_primal_ else None
        self._check_vector_inversion(K1_arg, K2_arg)
       
        return self

    def _check_vector_inversion(self, K1, K2):
        # Calculate Embeddings (E) based on mode
        if self.is_primal_:
            # Linear/Primal: E = X @ P
            P1 = self.projectors_[:self.d1_]
            P2 = self.projectors_[self.d1_:]
            E1 = self.X1_fit_ @ P1
            E2 = self.X2_fit_ @ P2
        else:
            # Kernel/Dual: E = K @ alpha
            alpha1 = self.alphas_[:self.n1_]
            alpha2 = self.alphas_[self.n1_:]
            E1 = K1 @ alpha1
            E2 = K2 @ alpha2

        # Normalize for comparison
        E1_z = (E1 - E1.mean(0)) / (E1.std(0) + 1e-9)
        E2_z = (E2 - E2.mean(0)) / (E2.std(0) + 1e-9)
       
        centroids_src = {}
        # Get mask for source labels to safely skip unlabeled
        mask_unl1 = LabelUtils.get_unlabeled_mask(self.y1_)

        for c in self.classes_:
            mask = (self.y1_ == c) & (~mask_unl1)
            if mask.any():
                centroids_src[c] = E1_z[mask].mean(0)

        flip_vector = np.ones(self.n_components)
        
        # Get mask for target labels
        mask_unl2 = LabelUtils.get_unlabeled_mask(self.y2_)
       
        for j in range(self.n_components):
            err_normal = 0.0
            err_flip = 0.0
           
            for c in self.classes_:
                if c not in centroids_src: continue
                
                mask2 = (self.y2_ == c) & (~mask_unl2)
                if not mask2.any(): continue
              
                cent_target = E2_z[mask2].mean(0)
              
                err_normal += (centroids_src[c][j] - cent_target[j])**2
                err_flip += (centroids_src[c][j] - (-cent_target[j]))**2
           
            if err_flip < err_normal:
                flip_vector[j] = -1.0
        
        # Apply flip to the learned weights
        if self.is_primal_:
            self.projectors_[self.d1_:, :] *= flip_vector
        else:
            self.alphas_[self.n1_:, :] *= flip_vector

    def transform(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        
        if self.is_primal_:
            # Linear Primal Projection: X @ P
            P1 = self.projectors_[:self.d1_]
            P2 = self.projectors_[self.d1_:]
            return np.vstack([X1 @ P1, X2 @ P2])
        else:
            # Kernel Dual Projection: K_new @ alpha
            K1_new = pairwise_kernels(X1, self.X1_fit_, metric=self.kernel, gamma=self.gamma)
            K2_new = pairwise_kernels(X2, self.X2_fit_, metric=self.kernel, gamma=self.gamma)
           
            alpha1 = self.alphas_[:self.n1_]
            alpha2 = self.alphas_[self.n1_:]
           
            return np.vstack([K1_new @ alpha1, K2_new @ alpha2])

    def fit_transform(self, X1, X2, y1, y2):
        self.fit(X1, X2, y1, y2)
        
        if self.is_primal_:
            # Directly project the training data
            return self.transform(X1, X2)
        else:
            # Recompute kernels for training data (Dual)
            K1 = pairwise_kernels(self.X1_fit_, metric=self.kernel, gamma=self.gamma)
            K2 = pairwise_kernels(self.X2_fit_, metric=self.kernel, gamma=self.gamma)
           
            alpha1 = self.alphas_[:self.n1_]
            alpha2 = self.alphas_[self.n1_:]
           
            return np.vstack([K1 @ alpha1, K2 @ alpha2])