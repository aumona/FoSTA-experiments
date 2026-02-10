import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
    n_neighbors=20,
    metric="euclidean",
    weights="distance",
    mode="rf",                 # NEW: "knn" or "rf"
    train_on="source",          # NEW: "source" or "source+target_visible"
    rf_kwargs=None,             # NEW: optional dict passed to RandomForestClassifier
    random_state=None,          # NEW: makes RF reproducible if you want
):
    """
    Label transfer in your experiment:
      - Train a classifier on labeled points
      - Predict labels for Target points whose labels were masked/hidden
      - Compute accuracy using y_target_true (ground truth you saved before masking)

    NEW:
      - mode="knn" or "rf"
      - train_on="source" (default) or "source+target_visible"
    """
    y_source = np.asarray(y_source).ravel()
    y_target_obs = np.asarray(y_target_obs).ravel()
    y_target_true = np.asarray(y_target_true).ravel()

    n_b = len(y_target_obs)
    if embedding.shape[0] != n_a + n_b:
        raise ValueError(f"embedding has {embedding.shape[0]} rows but expected {n_a + n_b} (n_a+n_b).")

    X_source = embedding[:n_a]
    X_target = embedding[n_a:]

    # Which target points are "missing labels" to evaluate?
    if mask_missing_target is None:
        mask_missing_target = (y_target_obs == -1)
    else:
        mask_missing_target = np.asarray(mask_missing_target, dtype=bool).ravel()
        if mask_missing_target.shape[0] != n_b:
            raise ValueError("mask_missing_target must have shape (n_b,) matching target after removal.")

    mask_visible_target = ~mask_missing_target

    # -----------------------------
    # Build training set
    # -----------------------------
    train_mask_source = (y_source != -1)
    X_train = X_source[train_mask_source]
    y_train = y_source[train_mask_source]

    if train_on not in {"source", "source+target_visible"}:
        raise ValueError("train_on must be 'source' or 'source+target_visible'.")

    if train_on == "source+target_visible":
        # include target points that are visible-labeled (not -1)
        vis_labeled = mask_visible_target & (y_target_obs != -1)
        if np.any(vis_labeled):
            X_train = np.vstack([X_train, X_target[vis_labeled]])
            y_train = np.concatenate([y_train, y_target_obs[vis_labeled]])

    if X_train.shape[0] == 0:
        return dict(
            acc_missing=None, acc_visible=None,
            n_missing=int(mask_missing_target.sum()),
            n_visible=int(mask_visible_target.sum()),
            y_pred_missing=None, missing_idx=np.array([], dtype=int),
            error="No labeled points to train on (after train_on filtering)."
        )

    # -----------------------------
    # Choose model
    # -----------------------------
    mode = mode.lower()
    if mode == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=None,
        )
    elif mode == "rf":
        rf_kwargs = {} if rf_kwargs is None else dict(rf_kwargs)
        # default params (sklearn defaults) + optional reproducibility
        if random_state is not None and "random_state" not in rf_kwargs:
            rf_kwargs["random_state"] = random_state
        clf = RandomForestClassifier(**rf_kwargs)
    else:
        raise ValueError("mode must be 'knn' or 'rf'.")

    clf.fit(X_train, y_train)

    out = {}
    missing_idx = np.where(mask_missing_target)[0]
    out["missing_idx"] = missing_idx
    out["n_missing"] = int(mask_missing_target.sum())
    out["n_visible"] = int(mask_visible_target.sum())
    out["mode"] = mode
    out["train_on"] = train_on

    # Evaluate on masked (main metric)
    if out["n_missing"] > 0:
        y_pred_missing = clf.predict(X_target[mask_missing_target])
        out["y_pred_missing"] = y_pred_missing
        out["acc_missing"] = accuracy_score(y_target_true[mask_missing_target], y_pred_missing)
    else:
        out["y_pred_missing"] = None
        out["acc_missing"] = None

    # Optional sanity check on visible target labels (compare against true)
    vis_eval = mask_visible_target & (y_target_obs != -1)
    if np.any(vis_eval):
        y_pred_visible = clf.predict(X_target[vis_eval])
        out["acc_visible"] = accuracy_score(y_target_true[vis_eval], y_pred_visible)
    else:
        out["acc_visible"] = None

    return out