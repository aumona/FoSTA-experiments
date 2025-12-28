from src.hiref import HiRef_fast as HiRef
from src.hiref import rank_annealing
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# ------------------------------------------------------------
# Universal OT Solver (The Update)
# ------------------------------------------------------------
def solve_compressed_hiref(post_a, post_b, verbose=0, random_state=None):
    """
    Universal Adaptive Transport Solver.
    Automatically handles Balanced (N_a == N_b) and Imbalanced (N_a != N_b) cases.
    
    This method implements a locally adaptive version of the "Mass Rebalancing" 
    strategy described in the MALI paper (Section D).

    Mechanism:
    ----------
    1. Clustering as Mass Rebalancing:
        Instead of a global uniform weight, we cluster the large domain to find 
        representatives. The mass of each representative is distributed equally 
        among its cluster members.
        
        Weight(point p) = 1 / |Cluster_Size|
        
        - Dense regions -> Large Clusters -> Small individual weights.
        - Sparse regions -> Small Clusters -> Large individual weights.
        
        This automatically fulfills the theoretical suggestion to "increase the 
        masses of samples belonging to low density regions," ensuring alignment 
        is driven by geometric structure rather than sampling density.

    Returns:
        T (sparse matrix): Shape (N_a, N_b) coupling matrix.
        
    -------------------------------------------------------------------------
    SCENARIO 1: Balanced (N_a == N_b)
    -------------------------------------------------------------------------
    - Direct bijection between points.
    
    -------------------------------------------------------------------------
    SCENARIO 2: N_a < N_b (Domain A is Small/Fixed)
    -------------------------------------------------------------------------
    - We call: run_adaptive_compression(post_fixed=post_a, post_to_compress=post_b)
    - Returns: T of shape (N_a, N_b).
    - Row Sums (N_a): Strictly 1.0 (Bijection to Centroids).
    - Col Sums (N_b): ~ 1/|Cluster| (Soft assignment via mass rebalancing).

    -------------------------------------------------------------------------
    SCENARIO 3: N_b < N_a (Domain B is Small/Fixed)
    -------------------------------------------------------------------------
    - We call: run_adaptive_compression(post_fixed=post_b, post_to_compress=post_a)
    - Returns: T_intermediate of shape (N_b, N_a).
    - We TRANSPOSE this to get final T of shape (N_a, N_b).
    """
    n_a = post_a.shape[0]
    n_b = post_b.shape[0]
    
    # --- Case 1: Balanced (Direct Bijection) ---
    if n_a == n_b:
        if verbose: print(f"OT: Balanced Mode ({n_a}x{n_b})")
        rank_schedule = rank_annealing.optimal_rank_schedule(n=n_a)
        (rows, cols, data), _ = HiRef.hiref_lr_fast(
            post_a, post_b,
            rank_schedule=rank_schedule,
            return_coupling=True
        )
        return sparse.coo_matrix((data, (rows, cols)), shape=(n_a, n_b)).tocsr()

    # --- Case 2: Domain A is Smaller (Compress B) ---
    elif n_a < n_b:
        if verbose: print(f"OT: Compressing B -> A ({n_b} -> {n_a})")
        # We fix A, cluster B to match size of A, then expand columns
        return run_adaptive_compression(post_fixed=post_a, post_to_compress=post_b, random_state=random_state)

    # --- Case 3: Domain B is Smaller (Compress A) ---
    else: # n_a > n_b
        if verbose: print(f"OT: Compressing A -> B ({n_a} -> {n_b})")
        # We fix B, cluster A to match size of B.
        # This gives T_ba (N_b x N_a). We transpose it to get T_ab (N_a x N_b).
        T_ba = run_adaptive_compression(post_fixed=post_b, post_to_compress=post_a, random_state=random_state)
        return T_ba.T.tocsr()

def run_adaptive_compression(post_fixed, post_to_compress, random_state=None):
    """
    Helper: Compresses `post_to_compress` to size of `post_fixed`, solves OT, then lifts.
    Returns matrix of shape (n_fixed, n_compress).
    
    Steps:
    1. Cluster the large domain (`post_to_compress`) into K clusters, where K = size of small domain (`post_fixed`).
    2. Solve bijective OT between `post_fixed` and the Centroids of the clusters.
    3. Distribute the mass from Centroids back to the original points in the large domain.
    """
    n_fixed = post_fixed.shape[0]          
    n_compress = post_to_compress.shape[0] 
    k = n_fixed  # Target clusters = size of small domain

    # 1. Clustering / Mass Rebalancing
    # We cluster the large domain to find 'k' representatives.
    # This implicitly defines the mass of each point based on local density.
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state)
    z = km.fit_predict(post_to_compress)        
    post_centroids = km.cluster_centers_        

    # 2. Solve Square OT (Small Domain <-> Centroids)
    # Map Small Domain <-> Centroids of Large Domain.
    # Since sizes match (n_fixed == k), HiRef forces a 1-to-1 matching.
    rank_schedule = rank_annealing.optimal_rank_schedule(n=n_fixed)
    (rows, cols, data), _ = HiRef.hiref_lr_fast(
        post_fixed, post_centroids,
        rank_schedule=rank_schedule,
        return_coupling=True
    )
    
    # T_coarse shape: (n_fixed, k)
    # Row sums = 1.0 (Perfect Bijection found)
    T_coarse = sparse.coo_matrix((data, (rows, cols)), shape=(n_fixed, k)).tocsr()

    # 3. Build Membership Matrix M (k x n_compress)
    # Distribute centroid mass equally to constituent points in Large Domain.
    # Weight = 1 / |Cluster_Size| (The "Mass Rebalancing" term)
    counts = np.bincount(z, minlength=k).astype(float)
    inv_counts = np.zeros_like(counts)
    inv_counts[counts > 0] = 1.0 / counts[counts > 0]
    
    M_membership = sparse.coo_matrix(
        (inv_counts[z], (z, np.arange(n_compress))),
        shape=(k, n_compress)
    ).tocsr()

    # 4. Lift to Full Resolution: T_final = T_coarse * M
    # T_intermediate = T_coarse @ M_membership
    # Shape: (n_fixed, n_compress)
    #
    # Verification:
    # - Row i (Small Domain point): Sums to 1.0.
    # - Col j (Large Domain point): Sums to 1/|Cluster|.
    return (T_coarse @ M_membership).tocsr()