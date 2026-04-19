from src.hiref import HiRef_fast as HiRef
from src.hiref import rank_annealing
from scipy import sparse
from scipy.optimize import linear_sum_assignment
import numpy as np
import heapq


def get_next_smooth_number(n):
    """
    Finds the smallest integer >= n that is 5-smooth
    (only prime factors 2, 3, and 5).
    This guarantees HiRef can factorize it completely using small ranks.
    """
    if n <= 1:
        return 1

    h = [1]
    seen = {1}

    while True:
        curr = heapq.heappop(h)

        if curr >= n:
            return curr

        for factor in [2, 3, 5]:
            nxt = curr * factor
            if nxt not in seen:
                seen.add(nxt)
                heapq.heappush(h, nxt)


# ============================================================
# Sparse bijection repair utilities
# ============================================================

def _greedy_unique_pairs(rows, cols, vals=None):
    """
    Keep only one pair per row and one pair per column.

    Parameters
    ----------
    rows, cols : array-like, shape (nnz,)
        Sparse pair indices.
    vals : array-like or None
        Optional scores used to prioritize which pairs to keep.
        If provided, larger values are kept first.

    Returns
    -------
    rows_keep, cols_keep : ndarray
        Deduplicated partial injective matching.
    """
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    if vals is None:
        order = np.arange(len(rows))
    else:
        vals = np.asarray(vals)
        order = np.argsort(-vals)

    used_rows = set()
    used_cols = set()
    keep = []

    for k in order:
        r = int(rows[k])
        c = int(cols[k])

        if r in used_rows or c in used_cols:
            continue

        used_rows.add(r)
        used_cols.add(c)
        keep.append(k)

    keep = np.asarray(keep, dtype=np.int32)
    return rows[keep], cols[keep]


def _complete_sparse_matching(rows_keep, cols_keep, X, Y):
    """
    Complete a partial injective matching into a full bijection by solving
    one final exact assignment problem on the unmatched padded rows/cols only.

    Parameters
    ----------
    rows_keep, cols_keep : ndarray
        Partial matching on the padded square problem.
    X, Y : ndarray, shape (n_target, d)
        Padded point clouds.

    Returns
    -------
    rows_full, cols_full : ndarray
        Full bijection on the padded square problem.
    """
    n_target = X.shape[0]

    used_rows = np.zeros(n_target, dtype=bool)
    used_cols = np.zeros(n_target, dtype=bool)
    used_rows[rows_keep] = True
    used_cols[cols_keep] = True

    rem_rows = np.flatnonzero(~used_rows)
    rem_cols = np.flatnonzero(~used_cols)

    if rem_rows.size == 0:
        return rows_keep, cols_keep

    Xr = np.asarray(X)[rem_rows]
    Yr = np.asarray(Y)[rem_cols]

    # Squared Euclidean cost on leftovers only
    x2 = np.sum(Xr**2, axis=1, keepdims=True)
    y2 = np.sum(Yr**2, axis=1, keepdims=True).T
    C = x2 + y2 - 2.0 * (Xr @ Yr.T)
    C = np.maximum(C, 0.0)

    rr, cc = linear_sum_assignment(C)

    rows_full = np.concatenate([rows_keep, rem_rows[rr].astype(np.int32)])
    cols_full = np.concatenate([cols_keep, rem_cols[cc].astype(np.int32)])

    return rows_full, cols_full


def _repair_padded_hiref_pairs(rows_aug, cols_aug, vals_aug, p_aug, s_aug, verbose=0):
    """
    Repair HiRef sparse output into a true bijection on the padded square problem.

    Steps
    -----
    1. Greedy deduplication using HiRef weights as priorities.
    2. Exact completion on the unmatched leftovers only.

    Returns
    -------
    rows_fix, cols_fix : ndarray
        Full bijection on the padded problem.
    """
    rows_aug = np.asarray(rows_aug, dtype=np.int32)
    cols_aug = np.asarray(cols_aug, dtype=np.int32)
    vals_aug = np.asarray(vals_aug)

    if verbose:
        n_target = p_aug.shape[0]
        row_counts = np.bincount(rows_aug, minlength=n_target)
        col_counts = np.bincount(cols_aug, minlength=n_target)
        print(
            "HiRef padded raw pairs | "
            f"unique rows: {(row_counts > 0).sum()}/{n_target}, "
            f"unique cols: {(col_counts > 0).sum()}/{n_target}, "
            f"max row multiplicity: {row_counts.max()}, "
            f"max col multiplicity: {col_counts.max()}"
        )

    rows_keep, cols_keep = _greedy_unique_pairs(rows_aug, cols_aug, vals_aug)

    if verbose:
        print(f"HiRef padded after greedy dedup | kept {len(rows_keep)} pairs")

    rows_fix, cols_fix = _complete_sparse_matching(rows_keep, cols_keep, p_aug, s_aug)

    if verbose:
        print(f"HiRef padded after exact completion | final {len(rows_fix)} pairs")

    return rows_fix, cols_fix


# ============================================================
# Main wrapper
# ============================================================

def solve_surjection_hiref(
    post_a,
    post_b,
    verbose=0,
    random_state=None,
    enforce_bijection=True,
):
    """
    Simplified HiRef Surjection.

    Pipeline
    --------
    1. Set padded target size = max(N_a, N_b).
    2. Oversample the smaller domain to size n_target.
    3. Solve square HiRef on the padded problem.
    4. Optionally repair the padded sparse HiRef output into a true bijection.
    5. Collapse padded copies back to original indices by summing.

    Returns
    -------
    T_final : scipy.sparse.csr_matrix, shape (n_a, n_b)
        Coupling matrix on the original problem.
    """
    rng = np.random.default_rng(random_state)
    post_a = np.asarray(post_a)
    post_b = np.asarray(post_b)

    n_a = post_a.shape[0]
    n_b = post_b.shape[0]

    n_target = max(n_a, n_b)

    if verbose:
        print(f"HiRef: Aligning A({n_a}) and B({n_b}). Target square size: {n_target}")

    def get_augmented_data(data, n_orig, n_dest):
        """
        Returns
        -------
        data_aug : ndarray, shape (n_dest, d)
            Oversampled data.
        map_indices : ndarray, shape (n_dest,)
            Mapping from padded rows back to original rows.
        """
        if n_orig == n_dest:
            return data, np.arange(n_orig, dtype=np.int32)

        n_repeats = n_dest // n_orig
        n_remainder = n_dest % n_orig

        idx_base = np.tile(np.arange(n_orig, dtype=np.int32), n_repeats)
        idx_rem = rng.choice(np.arange(n_orig, dtype=np.int32), n_remainder, replace=False)

        map_indices = np.concatenate([idx_base, idx_rem])

        # shuffle so duplicates are not clustered
        perm = rng.permutation(n_dest)
        map_indices = map_indices[perm]

        return data[map_indices], map_indices

    # --------------------------------------------------------
    # 1. Pad / oversample to the square problem
    # --------------------------------------------------------
    p_aug, map_a = get_augmented_data(post_a, n_a, n_target)
    s_aug, map_b = get_augmented_data(post_b, n_b, n_target)

    # --------------------------------------------------------
    # 2. Solve square HiRef on padded data
    # --------------------------------------------------------
    rank_schedule = rank_annealing.optimal_rank_schedule(n=n_target)

    (rows_aug, cols_aug, vals_aug), _ = HiRef.hiref_lr_fast(
        p_aug,
        s_aug,
        rank_schedule=rank_schedule,
        return_coupling=True,
    )

    # --------------------------------------------------------
    # 3. Minimal sparse repair on the padded problem
    # --------------------------------------------------------
    if enforce_bijection:
        rows_aug, cols_aug = _repair_padded_hiref_pairs(
            rows_aug,
            cols_aug,
            vals_aug,
            p_aug,
            s_aug,
            verbose=verbose,
        )
        vals_aug = np.ones(len(rows_aug), dtype=np.float32)
    else:
        rows_aug = np.asarray(rows_aug, dtype=np.int32)
        cols_aug = np.asarray(cols_aug, dtype=np.int32)
        vals_aug = np.asarray(vals_aug)

    # --------------------------------------------------------
    # 4. Collapse padded indices back to original indices
    # --------------------------------------------------------
    rows_final = map_a[rows_aug]
    cols_final = map_b[cols_aug]

    T_final = sparse.coo_matrix(
        (vals_aug, (rows_final, cols_final)),
        shape=(n_a, n_b),
    ).tocsr()

    if verbose:
        row_sums = np.asarray(T_final.sum(axis=1)).ravel()
        col_sums = np.asarray(T_final.sum(axis=0)).ravel()
        print(
            "Collapsed coupling stats | "
            f"total mass: {T_final.sum():.3f}, "
            f"row sum min/max: {row_sums.min():.3f}/{row_sums.max():.3f}, "
            f"col sum min/max: {col_sums.min():.3f}/{col_sums.max():.3f}"
        )

    return T_final