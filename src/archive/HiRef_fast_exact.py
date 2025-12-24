from functools import partial
from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import lax, vmap

Array = jnp.ndarray
IndexArray = jnp.ndarray  # 1-D int array

# =========================
#  Utilities: LR sq-Euclidean factors  C ≈ A @ B^T
# =========================
def lr_sqeuclidean_factors(X: Array, Y: Array, rescale: bool = False) -> Tuple[Array, Array]:
    n, d = X.shape
    A = jnp.concatenate(
        [jnp.sum(X**2, axis=1, keepdims=True), jnp.ones((n, 1), X.dtype), -2.0 * X],
        axis=1,
    )  # (n, d+2)
    B = jnp.concatenate(
        [jnp.ones((n, 1), Y.dtype), jnp.sum(Y**2, axis=1, keepdims=True), Y],
        axis=1,
    )  # (n, d+2)
    if rescale:
        sA = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(A)), 1.0))
        sB = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(B)), 1.0))
        A = A / sA
        B = B / sB
    return A, B


# =========================
#  Log-domain Sinkhorn (balanced), with dual reuse
# =========================
def _cost_matrix(f: Array, g: Array, G: Array, eps: float) -> Array:
    return -(G - f[:, None] - g[None, :]) / eps

@jax.jit
def log_sinkhorn_project(
    G: Array, a: Array, b: Array, eps: float, max_iter: int = 200,
    f: Optional[Array] = None, g: Optional[Array] = None,
    recenter_every: int = 30,
) -> Tuple[Array, Array, Array]:
    """Project exp(-G/eps) to {P : P1=a, P^T1=b} via log-domain Sinkhorn. Returns (P, f, g)."""
    n, m = G.shape
    if f is None: f = jnp.zeros((n,), G.dtype)
    if g is None: g = jnp.zeros((m,), G.dtype)

    log_a = jnp.log(a)
    log_b = jnp.log(b)

    def body(k, carry):
        f_k, g_k = carry
        M = _cost_matrix(f_k, g_k, G, eps)
        f_n = f_k + eps * (log_a - jax.scipy.special.logsumexp(M, axis=1))
        M = _cost_matrix(f_n, g_k, G, eps)
        g_n = g_k + eps * (log_b - jax.scipy.special.logsumexp(M, axis=0))
        def recenter(pair):
            ff, gg = pair
            alpha = jnp.mean(ff)
            return ff - alpha, gg + alpha
        f_n, g_n = lax.cond((k % recenter_every) == 0, recenter, lambda x: x, (f_n, g_n))
        return f_n, g_n

    f_new, g_new = lax.fori_loop(0, max_iter, body, (f, g))
    P = jnp.exp(_cost_matrix(f_new, g_new, G, eps))
    return P, f_new, g_new


# =========================
#  Initialization: full-rank Q,R via Sinkhorn
# =========================
def initialize_couplings(a: Array, b: Array, g: Array, gamma: float,
                         max_iter: int = 50, key: Optional[jax.Array] = None) -> Tuple[Array, Array]:
    if key is None:
        key = jax.random.PRNGKey(0)
    kQ, kR = jax.random.split(key, 2)
    N1, N2, r = a.shape[0], b.shape[0], g.shape[0]
    Cq = jax.random.uniform(kQ, (N1, r), dtype=a.dtype)
    Cr = jax.random.uniform(kR, (N2, r), dtype=b.dtype)
    eps = 1.0 / gamma
    Q, _, _ = log_sinkhorn_project(Cq, a, g, eps, max_iter=max_iter)
    R, _, _ = log_sinkhorn_project(Cr, b, g, eps, max_iter=max_iter)
    return Q, R


# =========================
#  Two-sided LR-OT (uniform g), mirror descent + Sinkhorn projection
# =========================
def _loss_lr_two(Q: Array, R: Array, A: Array, B: Array, g: Array) -> Array:
    SA = Q.T @ A  # (r,k)
    RB = R.T @ B  # (r,k)
    return jnp.sum(jnp.sum(RB * SA, axis=1) / jnp.clip(g, 1e-18))

grad_Q = jax.grad(_loss_lr_two, argnums=0)
grad_R = jax.grad(_loss_lr_two, argnums=1)

@jax.jit
def _md_sinkhorn_step(Q: Array, R: Array, A: Array, B: Array,
                      a: Array, b: Array, g: Array, gamma: float,
                      fQ: Array, gQd: Array, fR: Array, gRd: Array) -> Tuple[Tuple, None]:
    gq = grad_Q(Q, R, A, B, g)
    gr = grad_R(Q, R, A, B, g)
    norm = jnp.maximum(jnp.max(jnp.abs(gq)), jnp.max(jnp.abs(gr)))
    gamma_k = gamma / jnp.clip(norm, 1e-18)
    eps = 1.0 / gamma_k
    
    GQ = gq - (1.0 / gamma_k) * jnp.log(jnp.clip(Q, 1e-32))
    GR = gr - (1.0 / gamma_k) * jnp.log(jnp.clip(R, 1e-32))
    
    Qn, fQn, gQn = log_sinkhorn_project(GQ, a, g, eps, max_iter=15, f=fQ, g=gQd)
    Rn, fRn, gRn = log_sinkhorn_project(GR, b, g, eps, max_iter=15, f=fR, g=gRd)
    return (Qn, Rn, fQn, gQn, fRn, gRn), None

@partial(jax.jit, static_argnums=(2, 3))
def lrot_lr(A, B, r, iters=60, gamma=60.0, key=None):
    n, m = A.shape[0], B.shape[0]
    a = jnp.full((n,), 1.0 / n, A.dtype)
    b = jnp.full((m,), 1.0 / m, B.dtype)
    g = jnp.full((r,), 1.0 / r, A.dtype)

    Q0, R0 = initialize_couplings(a, b, g, gamma, max_iter=50, key=key)
    fQ0 = jnp.zeros((n,), A.dtype); gQ0 = jnp.zeros((r,), A.dtype)
    fR0 = jnp.zeros((m,), B.dtype); gR0 = jnp.zeros((r,), B.dtype)

    def scan_body(carry, _):
        Qc, Rc, fQc, gQc, fRc, gRc = carry
        # call step with the *correct* ordering
        (Qn, Rn, fQn, gQn, fRn, gRn), _ = _md_sinkhorn_step(
            Qc, Rc, A, B, a, b, g, gamma, fQc, gQc, fRc, gRc
        )
        return (Qn, Rn, fQn, gQn, fRn, gRn), None

    (Q, R, _, _, _, _), _ = lax.scan(scan_body,
                                     (Q0, R0, fQ0, gQ0, fR0, gR0),
                                     xs=None, length=iters)
    return Q, R


@partial(jax.jit, static_argnames=('cap',))
def split_by_capacity_device(scores: jnp.ndarray, cap: int) -> jnp.ndarray:
    # scores: (N, r) -> top `cap` row indices per column
    # Returns indices of shape (r, cap), dtype int32
    _, idx = lax.top_k(scores.T, k=cap)   # idx: (r, cap) in [0, N)
    return idx.astype(jnp.int32)


# =========================
#  Optimized Strict Monge Kernels
# =========================

@partial(jax.jit, static_argnums=(1,))
def split_exact_partition(scores: jnp.ndarray, cap: int) -> jnp.ndarray:
    """
    JIT-compiled Exact Partitioner.
    Args:
        scores: (N, 2) array of potentials/probabilities
        cap: int, static argument for exact splitting size (N // 2)
    Returns:
        (2, cap) array of local indices.
    """
    # 1. Compute preference score (Cluster 0 vs Cluster 1)
    diff = scores[:, 0] - scores[:, 1]
    
    # 2. Sort Descending: Top 'cap' prefer Cluster 0
    # JAX's argsort is very fast on GPU
    sorted_idx = jnp.argsort(diff)[::-1]
    
    # 3. Exact Slice (Requires 'cap' to be static or concrete)
    idx_0 = sorted_idx[:cap]
    idx_1 = sorted_idx[cap:]
    
    return jnp.stack([idx_0, idx_1])

@partial(jax.jit, static_argnames=('r', 'iters', 'gamma'))
def _process_level_vmap(A_full, B_full, batch_idxX, batch_idxY, r, iters, gamma):
    """
    Processes an ENTIRE level of the hierarchy in parallel using vmap.
    """
    batch_size, block_size = batch_idxX.shape
    cap = block_size // r

    # Define the single-block logic
    def _single_block_step(ix, iy):
        # 1. Gather Data (Slice global factors)
        Ai = A_full[ix, :]
        Bi = B_full[iy, :]
        
        # 2. Solve OT (Rank r)
        Q, R = lrot_lr(Ai, Bi, r=r, iters=iters, gamma=gamma)
        
        # 3. Partition
        # Note: If r > 2, you'd need the approx partitioner. 
        # For Strict Monge, we assume r=2 here as per your request.
        Xi = split_exact_partition(Q, cap) # (r, cap)
        Yi = split_exact_partition(R, cap) # (r, cap)
        
        # 4. Map Local Indices -> Global Indices
        # Xi contains indices [0, block_size). We need values from ix.
        # jnp.take is efficient for this indirection.
        Xi_global = jnp.take(ix, Xi, axis=0) # (r, cap)
        Yi_global = jnp.take(iy, Yi, axis=0) # (r, cap)
        
        return Xi_global, Yi_global

    # Vectorize over the batch dimension
    # (Batch, r, cap)
    new_X, new_Y = vmap(_single_block_step)(batch_idxX, batch_idxY)
    
    return new_X, new_Y

def hiref_lr_fast(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    rank_schedule: List[int],
    base_rank: int = 1,
    iters_per_level: int = 60,
    gamma: float = 60.0,
    rescale_cost: bool = False,
    return_coupling: bool = False,
    dense_coupling: bool = False,
):
    """
    Fast HiRef with VMAP vectorization.
    Assumes X, Y are padded to powers of 2 (or capable of uniform division).
    """
    n = int(X.shape[0])
    A_full, B_full = lr_sqeuclidean_factors(X, Y, rescale=rescale_cost)

    # Initial Frontier: 1 block of size N
    # Shape: (1, n)
    frontier_X = jnp.array([jnp.arange(n)], dtype=jnp.int32)
    frontier_Y = jnp.array([jnp.arange(n)], dtype=jnp.int32)

    for r in rank_schedule:
        # Check current block size
        num_blocks, block_size = frontier_X.shape
        
        # If we hit base rank, stop refining
        if block_size <= base_rank:
            break

        # --- THE SPEEDUP: VMAP Process ---
        # Instead of a Python loop, we pass the entire frontier tensors
        # Output shape: (num_blocks, r, cap)
        next_X, next_Y = _process_level_vmap(
            A_full, B_full, frontier_X, frontier_Y, 
            r=r, iters=iters_per_level, gamma=gamma
        )
        
        # Reshape for next iteration
        # Flatten (num_blocks, r, cap) -> (num_blocks * r, cap)
        new_num_blocks = num_blocks * r
        cap = block_size // r
        
        frontier_X = next_X.reshape(new_num_blocks, cap)
        frontier_Y = next_Y.reshape(new_num_blocks, cap)

    # ==========================================
    #  Reconstruct Output (Sparse or Dense)
    # ==========================================
    
    # At the end, frontier_X/Y are (TotalBlocks, BaseRank)
    # If BaseRank=1, we essentially have the permutation list.
    
    if not return_coupling:
        # Convert tensor back to list of tuples for compatibility if needed
        # But returning the tensors is usually cleaner.
        return frontier_X, frontier_Y

    if dense_coupling:
        # Fast dense scatter
        P = jnp.zeros((n, n), X.dtype)
        
        # Helper to scatter uniform values
        def _scatter_block(ix, iy):
             # Creates a block of 1/size at (ix, iy)
             # If size=1, value is 1.0
             sz = ix.shape[0]
             val = 1.0 / sz
             # Broadcast indices
             grid_x = jnp.broadcast_to(ix[:, None], (sz, sz)).flatten()
             grid_y = jnp.broadcast_to(iy[None, :], (sz, sz)).flatten()
             return grid_x, grid_y, jnp.full((sz*sz,), val, dtype=X.dtype)
             
        # Vmap the scatter prep
        bx, by, bvals = vmap(_scatter_block)(frontier_X, frontier_Y)
        
        # Set values
        P = P.at[(bx.flatten(), by.flatten())].set(bvals.flatten())
        return P

    # Sparse Output (Strictly matches your requested format)
    # We can do this much faster than the Python list append
    
    rows = frontier_X.flatten()
    cols = frontier_Y.flatten()
    
    # If base_rank > 1, we need to replicate indices for the block
    if frontier_X.shape[1] > 1:
        # This part handles the "uniform block" logic for sparse matrices
        # But if you are doing Strict Monge, usually base_rank=1
        size = frontier_X.shape[1]
        
        def _make_sparse_block(ix, iy):
            rr = jnp.repeat(ix, size)
            cc = jnp.tile(iy, size)
            vv = jnp.full((size*size,), 1.0/size, dtype=X.dtype)
            return rr, cc, vv
            
        rows, cols, data = vmap(_make_sparse_block)(frontier_X, frontier_Y)
        rows = rows.flatten()
        cols = cols.flatten()
        data = data.flatten()
    else:
        # 1-to-1 mapping
        data = jnp.ones_like(rows, dtype=X.dtype)

    # Return structure matching your original code
    # Note: I return the tensors directly as 'frontier' for compatibility
    return (rows, cols, data), (frontier_X, frontier_Y)


# =========================
#  OT cost from leaf pairs (1-1), works with or without C
# =========================
def compute_ot_cost(
    monge_clus: List[Tuple[IndexArray, IndexArray]],
    X: Array,
    Y: Array,
    C: Optional[Array] = None,
    sq_euclidean: bool = True,
) -> Array:
    # Concatenate all leaf pairs once
    ix_list = []
    iy_list = []
    for idxX, idxY in monge_clus:
        # ensure 1-1 (skip empties; protect against mismatched sizes)
        if idxX.size == 0 or idxY.size == 0:
            continue
        size = int(min(idxX.size, idxY.size))
        ix_list.append(jnp.asarray(idxX[:size], dtype=jnp.int32))
        iy_list.append(jnp.asarray(idxY[:size], dtype=jnp.int32))

    if not ix_list:
        return jnp.array(0.0, dtype=X.dtype)

    ix = jnp.concatenate(ix_list, axis=0)
    iy = jnp.concatenate(iy_list, axis=0)

    n = X.shape[0]
    n_dtype = jnp.array(n, dtype=X.dtype)

    if C is not None:
        # One vectorized gather + sum
        vals = C[ix, iy]         # shape (K,)
        return jnp.sum(vals) / n_dtype

    # Compute from coordinates in one shot
    diff = X[ix] - Y[iy]         # shape (K, d)
    if sq_euclidean:
        # avoids sqrt; a single fused reduction
        return jnp.sum(diff * diff) / n_dtype
    else:
        # If you need L2, this is still one kernel
        return jnp.sum(jnp.linalg.norm(diff, axis=1)) / n_dtype


