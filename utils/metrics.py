import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def as_int_labels(y):
    """Convert labels to int array with -1 for missing (supports NaN)."""
    y = np.asarray(y).ravel()
    if y.dtype.kind in ("f",):  # float -> may contain NaN
        y2 = y.copy()
        y2[np.isnan(y2)] = -1
        return y2.astype(int)
    return y.astype(int)


def label_transfer_accuracy(
    embedding,
    n_a,
    y_source,
    y_target_obs,
    y_target_true,
    mask_missing_target=None,
    n_neighbors=5,
    metric="euclidean",
    weights="uniform",
):
    """
    Label transfer in your experiment:
      - Train kNN on Source labeled points
      - Predict labels for Target points whose labels were masked/hidden
      - Compute accuracy using y_target_true (ground truth you saved before masking)

    Args
    ----
    embedding : (n_a + n_b, d) array
        Joint embedding where first n_a rows are Source, next n_b are Target (after removal).
    n_a : int
        Number of source points kept in the embedding.
    y_source : (n_a,) array-like
        Source labels (assumed observed; if you ever mask source too, it supports -1 as missing).
    y_target_obs : (n_b,) array-like
        Observed target labels AFTER masking/removal. Missing should be -1.
    y_target_true : (n_b,) array-like
        Ground-truth target labels AFTER removal (aligned with y_target_obs / keep_idx).
    mask_missing_target : (n_b,) bool array or None
        Boolean mask identifying which target points were intentionally hidden and should be evaluated.
        If None, defaults to (y_target_obs == -1).
    n_neighbors, metric, weights : passed to KNeighborsClassifier.

    Returns
    -------
    out : dict with keys
        - 'acc_missing' : accuracy on masked target points (None if no masked points)
        - 'acc_visible' : accuracy on visible-labeled target points (optional sanity check; None if none)
        - 'n_missing', 'n_visible'
        - 'y_pred_missing' : predicted labels for masked target points (None if none)
        - 'missing_idx' : indices (within target domain) that were evaluated
    """
    y_source = np.asarray(y_source).ravel()
    y_target_obs = np.asarray(y_target_obs).ravel()
    y_target_true = np.asarray(y_target_true).ravel()

    n_b = len(y_target_obs)
    if embedding.shape[0] != n_a + n_b:
        raise ValueError(f"embedding has {embedding.shape[0]} rows but expected {n_a + n_b} (n_a+n_b).")

    X_source = embedding[:n_a]
    X_target = embedding[n_a:]

    # Training mask on source (in case you later mask source too)
    train_mask = (y_source != -1)
    if not np.any(train_mask):
        return dict(
            acc_missing=None, acc_visible=None,
            n_missing=0, n_visible=0,
            y_pred_missing=None, missing_idx=np.array([], dtype=int),
            error="No labeled source points to train on."
        )

    # Which target points are "missing labels" to evaluate?
    if mask_missing_target is None:
        mask_missing_target = (y_target_obs == -1)
    else:
        mask_missing_target = np.asarray(mask_missing_target, dtype=bool).ravel()
        if mask_missing_target.shape[0] != n_b:
            raise ValueError("mask_missing_target must have shape (n_b,) matching target after removal.")

    mask_visible_target = ~mask_missing_target

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        n_jobs=None,  # sklearn KNN doesn't parallelize well for small dims; keep default
    )
    knn.fit(X_source[train_mask], y_source[train_mask])

    out = {}
    # Evaluate on masked (the meaningful metric in your setup)
    missing_idx = np.where(mask_missing_target)[0]
    out["missing_idx"] = missing_idx
    out["n_missing"] = int(mask_missing_target.sum())
    out["n_visible"] = int(mask_visible_target.sum())

    if out["n_missing"] > 0:
        y_pred_missing = knn.predict(X_target[mask_missing_target])
        out["y_pred_missing"] = y_pred_missing
        out["acc_missing"] = accuracy_score(y_target_true[mask_missing_target], y_pred_missing)
    else:
        out["y_pred_missing"] = None
        out["acc_missing"] = None

    # Optional sanity check: how well does transfer predict the *visible* target labels?
    # (Not the main metric; can diagnose domain shift / embedding issues)
    if out["n_visible"] > 0 and np.any(y_target_obs[mask_visible_target] != -1):
        y_pred_visible = knn.predict(X_target[mask_visible_target])
        out["acc_visible"] = accuracy_score(y_target_true[mask_visible_target], y_pred_visible)
    else:
        out["acc_visible"] = None

    return out