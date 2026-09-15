"""Shared seeded, stratified and nested paired-label masking."""
import numpy as np


def validate_label_mask_perc(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("LABEL_MASK_PERC must be a non-empty list of proportions.")
    if any(
        isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.integer, np.floating))
        or not np.isfinite(p) or not 0 <= p <= 1
        for p in values
    ):
        raise ValueError("LABEL_MASK_PERC values must be finite numbers in [0, 1].")
    if len(set(values)) != len(values):
        raise ValueError("LABEL_MASK_PERC must not contain duplicates.")


def make_label_visibility_mask(labels, proportion, seed, *, nested=False):
    """Mask floor(p * n) rows, stratified by label with seeded row selection."""
    validate_label_mask_perc([proportion])
    labels = np.asarray(labels)
    order = np.random.default_rng(seed).permutation(len(labels))
    n_masked = int(np.floor(len(labels) * proportion))
    if nested:
        # Interleave randomly ordered class members by their within-class
        # quantiles. Every masking level takes a prefix of this same ordering.
        priorities = np.empty(len(labels))
        shuffled_labels = labels[order]
        for label in np.unique(labels):
            positions = np.flatnonzero(shuffled_labels == label)
            priorities[positions] = (np.arange(len(positions)) + 0.5) / len(positions)
        masked = np.argsort(priorities, kind="stable")[:n_masked]
    else:
        masked = make_stratified_subsample_indices(
            labels[order], np.ones(len(labels), dtype=bool), n_masked
        )
    visible = np.ones(len(labels), dtype=bool)
    visible[order[masked]] = False
    return visible


def make_stratified_subsample_indices(labels, train_mask, max_sample, *, seed=None):
    if max_sample is None or len(labels) <= max_sample:
        return np.arange(len(labels))

    labels = np.asarray(labels).astype(str)
    train_mask = np.asarray(train_mask, dtype=bool)
    if labels.shape[0] != train_mask.shape[0]:
        raise ValueError(f"labels and train_mask must have equal length, got {labels.shape[0]} and {train_mask.shape[0]}.")

    strata = np.array([f"{label}|{int(is_train)}" for label, is_train in zip(labels, train_mask)])
    unique_strata, counts = np.unique(strata, return_counts=True)
    allocations = np.floor(counts * max_sample / len(labels)).astype(int)
    allocations = np.minimum(allocations, counts)

    positive = counts > 0
    allocations[(allocations == 0) & positive] = 1
    while allocations.sum() > max_sample:
        candidates = np.flatnonzero(allocations > 1)
        if candidates.size == 0:
            candidates = np.flatnonzero(allocations > 0)
        ratios = allocations[candidates] / counts[candidates]
        allocations[candidates[np.argmax(ratios)]] -= 1

    remainders = (counts * max_sample / len(labels)) - np.floor(counts * max_sample / len(labels))
    while allocations.sum() < max_sample:
        candidates = np.flatnonzero(allocations < counts)
        if candidates.size == 0:
            break
        ratios = remainders[candidates]
        allocations[candidates[np.argmax(ratios)]] += 1

    selected = []
    rng = np.random.default_rng(seed) if seed is not None else None
    for stratum, n_select in zip(unique_strata, allocations):
        if n_select > 0:
            candidates = np.flatnonzero(strata == stratum)
            if rng is not None:
                candidates = rng.permutation(candidates)
            selected.append(candidates[:n_select])
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=int)



def make_supervision_masks(labels, proportion, seed):
    """Return visible and evaluation masks over all matched pairs."""
    visible = make_label_visibility_mask(labels, proportion, seed + 17, nested=True)
    return visible, ~visible
