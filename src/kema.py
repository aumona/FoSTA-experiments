import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph

class KEMA(BaseEstimator):
    """
    Python port of Devis Tuia's 'KMA.m'.
    Strictly reproduces the heuristics for matrix scaling and vector inversion.
    """
    def __init__(self, n_components=2, kernel='rbf', gamma=None, mu=0.5, 
                 n_neighbors=9, reg=1e-3):
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
        
        # Kernels (Block Diagonal)
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
        is_labeled = (y_all != -1)
        if np.issubdtype(y_all.dtype, np.floating):
             is_labeled &= ~np.isnan(y_all)
        
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
        
        KAK = K @ A_inner @ K
        KBK = K @ B_inner @ K
        
        KAK += self.reg * np.eye(n)
        
        # Symmetrize
        KAK = (KAK + KAK.T) / 2
        KBK = (KBK + KBK.T) / 2

        # regularize firmly once
        KAK += (self.reg + 1e-6) * np.eye(n) 
        
        # Solve for LARGEST eigenvalues of (Push, Pull)
        vals, vecs = linalg.eigh(KBK, KAK)

        # Ensure symmetry perfectly before solver (numerical noise can break eigh)
        KAK = (KAK + KAK.T) / 2
        KBK = (KBK + KBK.T) / 2
        
        vals, vecs = linalg.eigh(KBK, KAK)
            
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        
        self.alphas_ = vecs[:, :self.n_components]
        self.X1_fit_ = X1
        self.X2_fit_ = X2
        self.n1_ = n1
        self.classes_ = np.unique(y_all[is_labeled])
        self.y1_ = y1
        self.y2_ = y2
        
        # --- 6. Vector Inversion Check ---
        self._check_vector_inversion(K1, K2)
        
        return self

    def _check_vector_inversion(self, K1, K2):
        alpha1 = self.alphas_[:self.n1_]
        E1 = K1 @ alpha1
        E1_z = (E1 - E1.mean(0)) / (E1.std(0) + 1e-9)

        alpha2 = self.alphas_[self.n1_:]
        E2 = K2 @ alpha2
        E2_z = (E2 - E2.mean(0)) / (E2.std(0) + 1e-9)
        
        centroids_src = {}
        for c in self.classes_:
            mask = (self.y1_ == c)
            if mask.any():
                centroids_src[c] = E1_z[mask].mean(0)

        flip_vector = np.ones(self.n_components)
        
        for j in range(self.n_components):
            err_normal = 0.0
            err_flip = 0.0
            
            for c in self.classes_:
                if c not in centroids_src: continue
                mask2 = (self.y2_ == c)
                if not mask2.any(): continue
                
                cent_target = E2_z[mask2].mean(0)
                
                err_normal += (centroids_src[c][j] - cent_target[j])**2
                err_flip += (centroids_src[c][j] - (-cent_target[j]))**2
            
            if err_flip < err_normal:
                flip_vector[j] = -1.0
                
        self.alphas_[self.n1_:, :] *= flip_vector

    def transform(self, X1, X2):
        K1_new = pairwise_kernels(X1, self.X1_fit_, metric=self.kernel, gamma=self.gamma)
        K2_new = pairwise_kernels(X2, self.X2_fit_, metric=self.kernel, gamma=self.gamma)
        
        alpha1 = self.alphas_[:self.n1_]
        alpha2 = self.alphas_[self.n1_:]
        
        return np.vstack([K1_new @ alpha1, K2_new @ alpha2])

    def fit_transform(self, X1, X2, y1, y2):
        self.fit(X1, X2, y1, y2)
        # Recompute kernels for training data
        K1 = pairwise_kernels(self.X1_fit_, metric=self.kernel, gamma=self.gamma)
        K2 = pairwise_kernels(self.X2_fit_, metric=self.kernel, gamma=self.gamma)
        
        alpha1 = self.alphas_[:self.n1_]
        alpha2 = self.alphas_[self.n1_:]
        
        return np.vstack([K1 @ alpha1, K2 @ alpha2])