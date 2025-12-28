import sys
import numpy as np
from src.Pamona.Pamona_without_projection import Pamona as Pamona_original
from umap import UMAP
from utils.labels import LabelUtils
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize
from utils.utils import kernel2Dist
import graphtools
import scipy.sparse as sparse

class JPamona(Pamona_original):
    def __init__(self, mu=0.5, gamma=0.5, n_components=2, embedder='spectral',
                 random_state=None,
                 n_neighbors=10, n_pca=None, decay=40, knn_dist='euclidean',
                 virtual_cells=0,
                 **kwargs):
        """
        Wrapper for Pamona with Semi-Supervised support.

        Args:
            mu (float): Weighting factor for combining intra- and inter-dataset
            gamma (float): Weight of the label prior (0.0 to 0.99).
                           0.0 = Unsupervised (Geometry only).
                           0.9 = Strong supervision (Forces same-label alignment).
            n_components (int): Number of dimensions for the final embedding (default 2).
            random_state (int or None): Random seed for reproducibility.
            **kwargs: Arguments passed to the original Pamona class
        """
        # Initialize the parent Pamona class
        super().__init__(**kwargs)
        self.mu = mu
        self.gamma = gamma
        self.embedder = embedder
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_pca = n_pca
        self.decay = decay
        self.knn_dist = knn_dist
        self.manual_seed = random_state
        self.integrated_data = None
        self.virtual_cells = virtual_cells

    def fit(self, x_a, x_b, y_a, y_b):
        """
        Computes the alignment matrix T and integrated data.
        """
        # Ensure inputs are numpy arrays
        x_a = np.array(x_a)
        x_b = np.array(x_b)
        self.n_a = x_a.shape[0]
        self.n_b = x_b.shape[0]
        
        # --- Semi-Supervised Logic: Construct Matrix M ---
        # We inject prior knowledge if gamma > 0 and labels are provided.
        if self.gamma > 0 and y_a is not None and y_b is not None:
            # Initialize M as ones (neutral cost)
            n1, n2 = x_a.shape[0], x_b.shape[0]
            M_prior = np.ones((n1, n2))

            # --- Uniformized Label Handling ---
            # Use shared helper to get masks for missing values (NaN, None, -1)
            mask_unl_a = LabelUtils.get_unlabeled_mask(y_a)
            mask_unl_b = LabelUtils.get_unlabeled_mask(y_b)
            
            # Valid masks are simply the inverse
            valid_a = ~mask_unl_a
            valid_b = ~mask_unl_b

            # Flatten to ensure broadcasting works
            y_a = np.asarray(y_a).ravel()
            y_b = np.asarray(y_b).ravel()

            # Identify matches: Where labels are equal AND both are valid
            # Broadcasting: (N, 1) == (1, M) -> (N, M) matrix
            matches = (y_a[:, None] == y_b[None, :])
            
            # Apply validity mask
            matches &= valid_a[:, None]
            matches &= valid_b[None, :]

            # Apply prior: Reduce cost for matching labels
            # M = 1 - gamma (Matches become 'cheaper' to transport)
            M_prior[matches] = 1.0 - self.gamma
            
            # Set the M attribute in the parent class
            self.M = M_prior
        else:
            # Ensure no prior is set if gamma is 0
            self.M = None
        
        self.T = self.run_Pamona([x_a, x_b])[0]
        self.T_norm = self.T / (self.T.max())


        # Build clean kNN graphs for the diagonal blocks
        # Note: 'connectivity' mode gives 0/1. If you want distances, use mode='distance'
        prox_a = graphtools.Graph(x_a, n_pca=self.n_pca, knn=self.n_neighbors, 
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False).K.toarray()
        prox_b = graphtools.Graph(x_b, n_pca=self.n_pca, knn=self.n_neighbors, 
                              decay=self.decay, distance=self.knn_dist,
                              n_jobs=1, verbose=False).K.toarray()

        W_ab = (np.dot(prox_a, self.T_norm) + np.dot(self.T_norm, prox_b))

        W_ba = W_ab.T

        # Assemble the Block Matrix
        # [ mu * Wa        (1-mu) * Wab ]
        # [ (1-mu) * Wba    mu * Wb     ]
        self.W = np.block([
            [self.mu * prox_a, (1 - self.mu) * W_ab],
            [(1 - self.mu) * W_ba, self.mu * prox_b]
        ])
        

        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
    
        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                random_state=self.manual_seed,
                n_jobs=-1,
                beta=0.5,
            )
            self.embedding_ = phate_op.fit_transform(self.W)
            return self.embedding_
    
        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.manual_seed,
                n_jobs=-1,
            )
            self.embedding_ = embedder.fit_transform(self.W)
            return self.embedding_
    
        elif self.embedder == 'UMAP':
            DistM = kernel2Dist(self.W)
            self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', random_state=self.manual_seed).fit_transform(DistM)
            return self.embedding_
        
        elif self.embedder == "barycentric":            
            if self.n_a > self.n_b: 
                # Case: A is the Anchor (Larger). Project B onto A.
                # Operation: B_proj = T.T * A
                # We need rows of T.T (which correspond to points in B) to sum to 1.
                
                # Normalize the rows (axis=1) of the transposed matrix
                T_transpose_norm = normalize(self.T.transpose(), norm='l1', axis=1)
                
                b_onto_a = np.dot(T_transpose_norm, x_a)
                self.embedding_ = np.concatenate((x_a, b_onto_a))
                
            else:
                # Case: B is the Anchor (Larger). Project A onto B.
                # Operation: A_proj = T * B
                # We need rows of T (which correspond to points in A) to sum to 1.
                
                # Normalize the rows (axis=1) of T
                T_norm = normalize(self.T, norm='l1', axis=1)
                
                a_onto_b = np.dot(T_norm, x_b)
                self.embedding_ = np.concatenate((a_onto_b, x_b))
                
            return self.embedding_
    
        else:
            raise ValueError(f"Unknown embedder={self.embedder}")
    
    def get_embeddings(self):
        return self.embedding_