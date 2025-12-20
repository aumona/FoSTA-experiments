import numpy as np
from src.Pamona.Pamona import Pamona as Pamona_original
from umap import UMAP

class Pamona(Pamona_original):
    def __init__(self, gamma=0.5, n_components=2, random_state=None, **kwargs):
        """
        Wrapper for Pamona with Semi-Supervised support.

        Args:
            gamma (float): Weight of the label prior (0.0 to 0.99).
                           0.0 = Unsupervised (Geometry only).
                           0.9 = Strong supervision (Forces same-label alignment).
            n_components (int): Number of dimensions for the final embedding (default 2).
            random_state (int or None): Random seed for reproducibility.
            **kwargs: Arguments passed to the original Pamona class 
                      (e.g., n_shared, Lambda, output_dim).
        """
        # Initialize the parent Pamona class
        super().__init__(**kwargs)
        self.gamma = gamma
        self.output_dim = n_components
        self.manual_seed = random_state
        self.integrated_data = None

    def fit(self, x_a, x_b, y_a, y_b):
        """
        Computes the alignment matrix T and integrated data.
        """
        # Ensure inputs are numpy arrays
        x_a = np.array(x_a)
        x_b = np.array(x_b)
        
        # --- Semi-Supervised Logic: Construct Matrix M ---
        # We inject prior knowledge if gamma > 0 and labels are provided.
        if self.gamma > 0 and y_a is not None and y_b is not None:
            # Initialize M as ones (neutral cost)
            n1, n2 = x_a.shape[0], x_b.shape[0]
            M_prior = np.ones((n1, n2))

            # Handle missing labels (assuming -1 or NaN are unlabeled)
            y_a = np.array(y_a)
            y_b = np.array(y_b)
            
            # Mask for valid labels
            valid_a = (y_a != -1)
            valid_b = (y_b != -1)
            # If labels are floats (NaN check)
            if np.issubdtype(y_a.dtype, np.floating):
                valid_a &= ~np.isnan(y_a)
                valid_b &= ~np.isnan(y_b)

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


        
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        """
        Runs fit and returns the stacked 2D embeddings.
        """
        self.fit(x_a, x_b, y_a, y_b)
        # run_Pamona returns: (integrated_data_list, T_matrix)
        self.integrated_data, self.T = self.run_Pamona([x_a, x_b])
        
        # integrated_data is a list [aligned_xA, aligned_xB] project using Joint Spectral Embedding
        # intra = standard kNN weights, inter = coupling T
        # We stack them to return a single matrix (n_samples_total, self.output_dim)
        # Final Projection to 2D using UMAP for visualization (default params from Pamona) 
        # embedding_all = UMAP(n_components=2,
        #                      n_neighbors=20,
        #                      min_dist=0.7,
        #                      random_state=self.manual_seed).fit_transform(np.vstack(self.integrated_data))

        # Alternatively, return the integrated data without UMAP projection, which is the original formulation abd better suits our comparison.
        embedding_all = np.vstack(self.integrated_data)
        
        return embedding_all