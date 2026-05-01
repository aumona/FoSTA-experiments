import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

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


  
    


def calc_frac_idx(x1_mat, x2_mat, true_pairs=None):
    """
    Returns fraction closer than true match for each sample (as an array)
    input: x1_mat: embedding matrix of domain 1 (n_samples x n_dims)
           x2_mat: embedding matrix of domain 2 (n_samples x n_dims)
           rows of x1_mat and x2_mat correspond to true matches
           
    if true_pairs is provided, it should be a list of tuples of the same length as x1_mat, indicating the indices of true matches in the other domain
    otherwise, it is assumed that the rows correspond to true matches
    the same index may appear multiple times if the datasets are unbalanced
    
    returns: fracs: list of fractions for each sample
            x: list of sample indices (1 to n_samples)
    """       
        
    
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))

        # get index of the true pairing in the other domain
        if true_pairs is not None:
            true_idx = true_pairs[row_idx][0]
        else:
            true_idx = row_idx
            
        true_nbr = euc_dist[true_idx]
        sort_euc_dist = sorted(euc_dist)
        rank = sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x




def calc_frac_idx_faster(x1_mat, x2_mat, true_pairs=None):
    """
    Fast version supporting both NumPy arrays and SciPy CSR matrices.
    """
    # 1. Compute pairwise distances
    # sklearn's pairwise_distances handles CSR vs Dense automatically
    dist_mat = pairwise_distances(x1_mat, x2_mat, metric='euclidean')
    
    nsamp = x1_mat.shape[0]
    n_targets = x2_mat.shape[0]
    
    # 2. Get the indices for the true matches
    if true_pairs is not None:
        # p[0] based on your original logic
        true_indices = np.array([p[0] for p in true_pairs])
    else:
        true_indices = np.arange(nsamp)
    
    # 3. Extract the distances of the true matches
    # dist_mat[row_indices, col_indices]
    true_dists = dist_mat[np.arange(nsamp), true_indices]
    
    # 4. Calculate rank
    # Compare each row to its corresponding true_dist
    # true_dists[:, None] reshapes to (nsamp, 1) to allow broadcasting
    ranks = np.sum(dist_mat < true_dists[:, np.newaxis], axis=1)
    
    # 5. Calculate fractions
    denom = max(1, n_targets - 1)
    fracs = ranks.astype(float) / denom
    x = np.arange(1, nsamp + 1)
    
    return fracs.tolist(), x.tolist()

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat, true_pairs_1to2=None, true_pairs_2to1=None):
    """
    Metric from SCOT: "FOSCTTM"
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    
    if true_pairs_1to2 and true_pairs_2to1 are provided, they should be lists of tuples indicating the indices of true matches in the other domain
    """
    fracs1,xs = calc_frac_idx_faster(x1_mat, x2_mat, true_pairs = true_pairs_1to2)
    fracs2,xs = calc_frac_idx_faster(x2_mat, x1_mat, true_pairs=true_pairs_2to1)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return np.array(fracs)

def calc_frac_idx_bygroup(x1_mat,x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x

def calc_domainAveraged_FOSCTTM_bygroup(x1_mat, x2_mat, classes):
    """
    Metric from SCOT: "FOSCTTM"
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return fracs


    

def test_transfer_accuracy(data1, data2, type1, type2):
    """
    Metric from UnionCom: "Label Transfer Accuracy"
    """
    Min = np.minimum(len(data1), len(data2))
    k = np.maximum(10, (len(data1) + len(data2))*0.01)
    k = k.astype(np.int)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data2, type2)
    type1_predict = knn.predict(data1)
    # np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    return count / len(type1)