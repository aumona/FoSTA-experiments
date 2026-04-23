from src.hiref import HiRef_fast as HiRef
from src.hiref import rank_annealing
from scipy import sparse
import numpy as np
import heapq


# # ------------------------------------------------------------
# # Universal OT Solver (The Update)
# # ------------------------------------------------------------
# def solve_compressed_hiref(post_a, post_b, verbose=0, random_state=None):
#     """
#     Universal Adaptive Transport Solver.
#     Automatically handles Balanced (N_a == N_b) and Imbalanced (N_a != N_b) cases.
    
#     This method implements a locally adaptive version of the "Mass Rebalancing" 
#     strategy described in the MALI paper (Section D).

#     Mechanism:
#     ----------
#     1. Clustering as Mass Rebalancing:
#         Instead of a global uniform weight, we cluster the large domain to find 
#         representatives. The mass of each representative is distributed equally 
#         among its cluster members.
        
#         Weight(point p) = 1 / |Cluster_Size|
        
#         - Dense regions -> Large Clusters -> Small individual weights.
#         - Sparse regions -> Small Clusters -> Large individual weights.
        
#         This automatically fulfills the theoretical suggestion to "increase the 
#         masses of samples belonging to low density regions," ensuring alignment 
#         is driven by geometric structure rather than sampling density.

#     Returns:
#         T (sparse matrix): Shape (N_a, N_b) coupling matrix.
        
#     -------------------------------------------------------------------------
#     SCENARIO 1: Balanced (N_a == N_b)
#     -------------------------------------------------------------------------
#     - Direct bijection between points.
    
#     -------------------------------------------------------------------------
#     SCENARIO 2: N_a < N_b (Domain A is Small/Fixed)
#     -------------------------------------------------------------------------
#     - We call: run_adaptive_compression(post_fixed=post_a, post_to_compress=post_b)
#     - Returns: T of shape (N_a, N_b).
#     - Row Sums (N_a): Strictly 1.0 (Bijection to Centroids).
#     - Col Sums (N_b): ~ 1/|Cluster| (Soft assignment via mass rebalancing).

#     -------------------------------------------------------------------------
#     SCENARIO 3: N_b < N_a (Domain B is Small/Fixed)
#     -------------------------------------------------------------------------
#     - We call: run_adaptive_compression(post_fixed=post_b, post_to_compress=post_a)
#     - Returns: T_intermediate of shape (N_b, N_a).
#     - We TRANSPOSE this to get final T of shape (N_a, N_b).
#     """
#     n_a = post_a.shape[0]
#     n_b = post_b.shape[0]
    
#     # --- Case 1: Balanced (Direct Bijection) ---
#     if n_a == n_b:
#         if verbose: print(f"OT: Balanced Mode ({n_a}x{n_b})")
#         rank_schedule = rank_annealing.optimal_rank_schedule(n=n_a)
#         (rows, cols, data), _ = HiRef.hiref_lr_fast(
#             post_a, post_b,
#             rank_schedule=rank_schedule,
#             return_coupling=True
#         )
#         return sparse.coo_matrix((data, (rows, cols)), shape=(n_a, n_b)).tocsr()

#     # --- Case 2: Domain A is Smaller (Compress B) ---
#     elif n_a < n_b:
#         if verbose: print(f"OT: Compressing B -> A ({n_b} -> {n_a})")
#         # We fix A, cluster B to match size of A, then expand columns
#         return run_adaptive_compression(post_fixed=post_a, post_to_compress=post_b, random_state=random_state)

#     # --- Case 3: Domain B is Smaller (Compress A) ---
#     else: # n_a > n_b
#         if verbose: print(f"OT: Compressing A -> B ({n_a} -> {n_b})")
#         # We fix B, cluster A to match size of B.
#         # This gives T_ba (N_b x N_a). We transpose it to get T_ab (N_a x N_b).
#         T_ba = run_adaptive_compression(post_fixed=post_b, post_to_compress=post_a, random_state=random_state)
#         return T_ba.T.tocsr()

# def run_adaptive_compression(post_fixed, post_to_compress, random_state=None):
#     """
#     Helper: Compresses `post_to_compress` to size of `post_fixed`, solves OT, then lifts.
#     Returns matrix of shape (n_fixed, n_compress).
    
#     Steps:
#     1. Cluster the large domain (`post_to_compress`) into K clusters, where K = size of small domain (`post_fixed`).
#     2. Solve bijective OT between `post_fixed` and the Centroids of the clusters.
#     3. Distribute the mass from Centroids back to the original points in the large domain.
#     """
#     n_fixed = post_fixed.shape[0]          
#     n_compress = post_to_compress.shape[0] 
#     k = n_fixed  # Target clusters = size of small domain

#     # 1. Clustering / Mass Rebalancing
#     # We cluster the large domain to find 'k' representatives.
#     # This implicitly defines the mass of each point based on local density.
#     km = MiniBatchKMeans(n_clusters=k, random_state=random_state)
#     z = km.fit_predict(post_to_compress)        
#     post_centroids = km.cluster_centers_        

#     # 2. Solve Square OT (Small Domain <-> Centroids)
#     # Map Small Domain <-> Centroids of Large Domain.
#     # Since sizes match (n_fixed == k), HiRef forces a 1-to-1 matching.
#     rank_schedule = rank_annealing.optimal_rank_schedule(n=n_fixed)
#     (rows, cols, data), _ = HiRef.hiref_lr_fast(
#         post_fixed, post_centroids,
#         rank_schedule=rank_schedule,
#         return_coupling=True
#     )
    
#     # T_coarse shape: (n_fixed, k)
#     # Row sums = 1.0 (Perfect Bijection found)
#     T_coarse = sparse.coo_matrix((data, (rows, cols)), shape=(n_fixed, k)).tocsr()

#     # 3. Build Membership Matrix M (k x n_compress)
#     # Distribute centroid mass equally to constituent points in Large Domain.
#     # Weight = 1 / |Cluster_Size| (The "Mass Rebalancing" term)
#     counts = np.bincount(z, minlength=k).astype(float)
#     inv_counts = np.zeros_like(counts)
#     inv_counts[counts > 0] = 1.0 / counts[counts > 0]
    
#     M_membership = sparse.coo_matrix(
#         (inv_counts[z], (z, np.arange(n_compress))),
#         shape=(k, n_compress)
#     ).tocsr()

#     # Scaling factor to match the total mass of source (biggest) domain
#     # This ensures the final coupling T has appropriate total mass.
#     scaling_factor = n_compress / n_fixed

#     # 4. Lift to Full Resolution: T_final = T_coarse * M
#     # T_intermediate = T_coarse @ M_membership
#     # Shape: (n_fixed, n_compress)
#     #
#     # Verification:
#     # - Row i (Small Domain point): Sums to 1.0.
#     # - Col j (Large Domain point): Sums to 1/|Cluster|.
#     return (T_coarse @ M_membership).tocsr() * scaling_factor





# def solve_dummy_hiref(
#     post_a, post_b,
#     verbose=0,
#     random_state=None,
#     dummy_mode="uniform",   # {"uniform","mean","bootstrap"}
#     extra_dummy=0,
#     eps=1e-12,
#     enforce_mali_marginals=True,
# ):
#     """
#     HiRef coupling for unequal sizes using dummy padding to a square problem.
#     Returns sparse T with shape (n_a, n_b).

#     If enforce_mali_marginals=True:
#       - target semantics match your current unbalanced wot mode:
#         a_i = 1, b_j = n_a/n_b
#       - i.e., row sums ~ 1, col sums ~ n_a/n_b, total mass ~ n_a
#     """
#     rng = np.random.default_rng(random_state)
#     post_a = np.asarray(post_a)
#     post_b = np.asarray(post_b)
#     n_a, d_a = post_a.shape
#     n_b, d_b = post_b.shape
#     if d_a != d_b:
#         raise ValueError(f"post_a and post_b must have same #dims, got {d_a} vs {d_b}")

#     # -------------------------
#     # Dummy generator
#     # -------------------------
#     def make_dummies(P, n_new):
#         if n_new <= 0:
#             return None
#         d = P.shape[1]
#         if dummy_mode == "uniform":
#             D = np.full((n_new, d), 1.0 / d, dtype=P.dtype)
#         elif dummy_mode == "mean":
#             mu = P.mean(axis=0, keepdims=True)
#             mu = np.clip(mu, eps, None)
#             mu = mu / mu.sum(axis=1, keepdims=True)
#             D = np.repeat(mu.astype(P.dtype), n_new, axis=0)
#         elif dummy_mode == "bootstrap":
#             # sample real points (helps stay on-manifold)
#             idx = rng.integers(0, P.shape[0], size=n_new)
#             D = P[idx].astype(P.dtype, copy=True)
#         else:
#             raise ValueError("dummy_mode must be {'uniform','mean','bootstrap'}")
#         return D

#     # -------------------------
#     # Solve square HiRef helper
#     # -------------------------
#     def solve_square(A, B, n):
#         rank_schedule = rank_annealing.optimal_rank_schedule(n=n)
#         (rows, cols, data), _ = HiRef.hiref_lr_fast(
#             A, B, rank_schedule=rank_schedule, return_coupling=True
#         )
#         return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

#     # -------------------------
#     # Case 1: balanced
#     # -------------------------
#     if n_a == n_b:
#         if verbose:
#             print(f"HiRef: balanced ({n_a}x{n_b})")
            
#         try:
#             T = solve_square(post_a, post_b, n=n_a)
#         except Exception as e:
#             if verbose:
#                 print(f"failure of HiRef might be due to prime n = {n_a}, retrying with {n_a} + 1...")
#             T = solve_square(post_a, post_b, n=n_a+1) # solve_square will fail in n is prime. this is a quick fix that adds a dummy point
#             T = T[:-1, :-1] # remove dummy row/col
            
#         # In balanced case, HiRef usually gives row/col sums ~1. Total mass ~ n_a.
#         return T

#     # -------------------------
#     # Case 2: n_a > n_b  (pad B)
#     # -------------------------
#     if n_a > n_b:
#         n_dummy = (n_a - n_b) + int(extra_dummy)
#         if verbose:
#             print(f"HiRef: pad B {n_b} -> {n_b + n_dummy} (solve {n_a}x{n_a})")

#         D = make_dummies(post_b, n_dummy)
#         post_b_pad = np.vstack([post_b, D]) if D is not None else post_b

#         T_pad = solve_square(post_a, post_b_pad, n=n_a)

#         # keep only real target columns
#         T = T_pad[:, :n_b].tocsr()

#         if enforce_mali_marginals:
#             # MALI unbalanced convention expects total mass = n_a,
#             # row sums ~ 1, col sums ~ n_a/n_b.
#             # After dropping dummy cols, some mass is missing -> rescale globally.
#             mass = float(T.sum())
#             if mass > eps:
#                 T = T * (float(n_a) / mass)

#         return T

#     # -------------------------
#     # Case 3: n_b > n_a  (pad A)
#     # -------------------------
#     else:
#         n_dummy = (n_b - n_a) + int(extra_dummy)
#         if verbose:
#             print(f"HiRef: pad A {n_a} -> {n_a + n_dummy} (solve {n_b}x{n_b})")

#         D = make_dummies(post_a, n_dummy)
#         post_a_pad = np.vstack([post_a, D]) if D is not None else post_a

#         # Solve square in the other direction (size n_b)
#         T_pad = solve_square(post_a_pad, post_b, n=n_b)

#         # keep only real source rows
#         T = T_pad[:n_a, :].tocsr()

#         if enforce_mali_marginals:
#             # Still want total mass = n_a
#             mass = float(T.sum())
#             if mass > eps:
#                 T = T * (float(n_a) / mass)

#         return T







# def get_next_smooth_number(n, max_prime=13):
#     """
#     Finds the smallest number >= n whose prime factors are all <= max_prime.
#     Solves the 'Prime Number' crash in HiRef.
#     """
#     def is_smooth(x):
#         if x <= 1: return True
#         temp = x
#         d = 2
#         while d * d <= temp:
#             while temp % d == 0:
#                 if d > max_prime: return False
#                 temp //= d
#             d += 1
#         if temp > 1 and temp > max_prime:
#             return False
#         return True

#     while not is_smooth(n):
#         n += 1
#     return n



# def solve_dummy_hiref(
#     post_a, post_b,
#     verbose=0,
#     random_state=None,
#     dummy_mode="bootstrap",  # "uniform", "mean", "bootstrap"
#     extra_dummy=0,
#     eps=1e-12,
#     enforce_mali_marginals=True,
# ):
#     """
#     Robust HiRef coupling for unequal sizes. 
#     1. Pads A and B to the nearest 'Smooth' Square size (preventing Prime crashes).
#     2. Solves square OT.
#     3. Slices back to original dimensions.
#     4. Renormalizes mass to equal n_a.
#     """
#     rng = np.random.default_rng(random_state)
#     post_a = np.asarray(post_a)
#     post_b = np.asarray(post_b)
#     n_a, d_a = post_a.shape
#     n_b, d_b = post_b.shape

#     if d_a != d_b:
#         raise ValueError(f"Dims mismatch: {d_a} vs {d_b}")

#     # 1. Determine Safe Square Size
#     # We take the max dimension, add extra_dummy, and then find the next "smooth" number
#     # to ensure HiRef's recursive rank schedule works efficiently.
#     target_raw = max(n_a, n_b) + int(extra_dummy)
#     n_target = get_next_smooth_number(target_raw)

#     if verbose > 0:
#         print(f"HiRef: Input {n_a}x{n_b} -> Solving Square {n_target}x{n_target}")

#     # 2. Dummy Generator
#     def pad_matrix(P, target_n):
#         curr_n = P.shape[0]
#         n_needed = target_n - curr_n
#         if n_needed <= 0:
#             return P
        
#         if dummy_mode == "uniform":
#             # Small uniform value (1/d)
#             D = np.full((n_needed, d_a), 1.0/d_a, dtype=P.dtype)
#         elif dummy_mode == "mean":
#             # Mean of the dataset
#             mu = P.mean(axis=0, keepdims=True)
#             D = np.repeat(mu, n_needed, axis=0)
#         elif dummy_mode == "bootstrap":
#             # Random samples from the data itself (Preserves Manifold Geometry)
#             idx = rng.integers(0, curr_n, size=n_needed)
#             D = P[idx]
#         else:
#             raise ValueError(f"Unknown dummy_mode: {dummy_mode}")
        
#         return np.vstack([P, D])

#     # 3. Prepare Inputs (Pad Both Sides)
#     # 
#     post_a_pad = pad_matrix(post_a, n_target)
#     post_b_pad = pad_matrix(post_b, n_target)

#     # 4. Solve Square HiRef
#     # Now dimensions match exactly: (n_target, d)
#     rank_schedule = rank_annealing.optimal_rank_schedule(n=n_target)
    
#     (rows, cols, data), _ = HiRef.hiref_lr_fast(
#         post_a_pad, post_b_pad, 
#         rank_schedule=rank_schedule, 
#         return_coupling=True
#     )
    
#     # 5. Reconstruct & Slice
#     # Create full square matrix
#     T_square = sparse.coo_matrix((data, (rows, cols)), shape=(n_target, n_target)).tocsr()
    
#     # Slice back to Real x Real dimensions
#     # This discards Real-to-Dummy and Dummy-to-Dummy connections
#     T = T_square[:n_a, :n_b]

#     # 6. Renormalize (CRITICAL)
#     # Slicing drops mass. We must re-scale so total mass equals n_a (source samples).
#     if enforce_mali_marginals:
#         current_mass = T.sum()
#         if current_mass < eps:
#             if verbose: print("Warning: HiRef mass collapsed (all real points matched to dummies).")
#             # Fallback: distribute mass uniformly or identity if shapes match? 
#             # Usually implies dummies were 'closer' than real points.
#         else:
#             scale_factor = float(n_a) / current_mass
#             T = T * scale_factor
            
#     return T


def get_next_admissible_size(
    n,
    hierarchy_depth=6,
    max_Q=int(2**10),
    max_rank=16,
):
    """
    Return the smallest integer >= n for which optimal_rank_schedule(...)
    succeeds under the given constraints.
    """
    n_try = int(n)

    while True:
        try:
            rank_annealing.optimal_rank_schedule(
                n=n_try,
                hierarchy_depth=hierarchy_depth,
                max_Q=max_Q,
                max_rank=max_rank,
            )
            return n_try
        except ValueError:
            n_try += 1

def solve_surjection_hiref(
    post_a, post_b,
    hierarchy_depth=6,
    max_Q=int(2**10),
    max_rank=16,
    verbose=0,
    random_state=None,
):
    """
    Simplified HiRef Surjection.
    1. Sets target size = smallest admissible size >= max(N_a, N_b).
    2. Pads the smaller domain to match target size (stratified copies).
    3. Solves square HiRef.
    4. Collapses copies back to original indices by summing.
    """
    rng = np.random.default_rng(random_state)
    post_a = np.asarray(post_a)
    post_b = np.asarray(post_b)

    n_a = post_a.shape[0]
    n_b = post_b.shape[0]

    # 1. Determine target size compatible with the exposed schedule parameters
    n_raw = max(n_a, n_b)
    n_target = get_next_admissible_size(
        n=n_raw,
        hierarchy_depth=hierarchy_depth,
        max_Q=max_Q,
        max_rank=max_rank,
    )

    if verbose:
        if n_target == n_raw:
            print(f"HiRef: Aligning A({n_a}) and B({n_b}). Target square size: {n_target}")
        else:
            print(
                f"HiRef: Aligning A({n_a}) and B({n_b}). "
                f"Raw target size: {n_raw}, padded target size: {n_target}"
            )

    def get_augmented_data(data, n_orig, n_dest):
        """
        Returns:
            data_aug: The stretched data (n_dest, d)
            map_indices: Array of shape (n_dest,) mapping new rows back to old rows
        """
        if n_orig == n_dest:
            return data, np.arange(n_orig, dtype=np.int64)

        n_repeats = n_dest // n_orig
        n_remainder = n_dest % n_orig

        idx_base = np.tile(np.arange(n_orig, dtype=np.int64), n_repeats)

        if n_remainder > 0:
            idx_rem = rng.choice(np.arange(n_orig, dtype=np.int64), n_remainder, replace=False)
            map_indices = np.concatenate([idx_base, idx_rem])
        else:
            map_indices = idx_base

        perm = rng.permutation(n_dest)
        map_indices = map_indices[perm]

        return data[map_indices], map_indices

    # 2. Prepare Augmented Inputs
    p_aug, map_a = get_augmented_data(post_a, n_a, n_target)
    s_aug, map_b = get_augmented_data(post_b, n_b, n_target)

    # 3. Solve HiRef (Square)
    rank_schedule = rank_annealing.optimal_rank_schedule(
        n=n_target,
        hierarchy_depth=hierarchy_depth,
        max_Q=max_Q,
        max_rank=max_rank,
    )

    (rows_aug, cols_aug, vals_aug), _ = HiRef.hiref_lr_fast(
        p_aug,
        s_aug,
        rank_schedule=rank_schedule,
        return_coupling=True,
    )

    # 4. Merge Duplicates (Collapse)
    rows_final = map_a[rows_aug]
    cols_final = map_b[cols_aug]

    T_final = sparse.coo_matrix(
        (vals_aug, (rows_final, cols_final)),
        shape=(n_a, n_b),
    ).tocsr()

    return T_final