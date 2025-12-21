import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def sugar_generator(rfgap_estimator, affinity_matrix, n_new_samples=None):
    """
    Implements the SUGAR geometry-based data generation.
    
    Parameters
    ----------
    rfgap_estimator : RFGAP object
        A fitted RFGAP estimator.
    affinity_matrix : sparse matrix (N, N)
        The RFGAP affinity/proximity matrix (symmetric).
        High value = close proximity.
    n_new_samples : int, optional
        Number of samples to generate. If None, matches X.shape[0].
    noise_scale : float
        Magnitude of Gaussian noise to add before projecting back.
    k_neighbors : int
        Number of neighbors for local bandwidth estimation (if needed).
        
    Returns
    -------
    X_new : array (N_new, D)
        The generated synthetic points aligned to the manifold.
    """