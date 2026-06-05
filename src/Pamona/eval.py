import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
import random
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances


def _alignment_score_rng(random_state):
    if random_state is None:
        return random.Random()
    return random.Random(random_state)


def calc_frac_idx(x1_mat, x2_mat, ids1=None, ids2=None):
    """
    Returns FOSCTTM fractions from x1_mat to x2_mat.

    If ids1 and ids2 are None:
        assumes one-to-one correspondence by row index.

    If ids1 and ids2 are provided:
        valid matches for x1[i] are all x2[j] with ids2[j] == ids1[i].

        For multiple valid matches, we use the farthest true match.
        This measures the fraction of incorrect samples that are closer
        than the worst-ranked true match.
    """
    fracs = []
    x = []

    nsamp1 = x1_mat.shape[0]
    nsamp2 = x2_mat.shape[0]

    use_ids = ids1 is not None and ids2 is not None

    if use_ids:
        ids1 = np.asarray(ids1)
        ids2 = np.asarray(ids2)

    for row_idx in range(nsamp1):
        euc_dist = np.sqrt(
            np.sum(np.square(x1_mat[row_idx, :] - x2_mat), axis=1)
        )

        if use_ids:
            valid = np.where(ids2 == ids1[row_idx])[0]

            if len(valid) == 0:
                fracs.append(np.nan)
                x.append(-1)
                continue

            invalid_mask = np.ones(nsamp2, dtype=bool)
            invalid_mask[valid] = False

            # Farthest true match = strict many-match FOSCTTM
            true_nbr = np.max(euc_dist[valid])

            rank = np.sum(euc_dist[invalid_mask] < true_nbr)
            frac = float(rank) / max(np.sum(invalid_mask), 1)

            # store nearest valid match index, for compatibility/debugging
            x.append(valid[np.argmin(euc_dist[valid])] + 1)

        else:
            true_nbr = euc_dist[row_idx]
            rank = np.sum(euc_dist < true_nbr)
            frac = float(rank) / max(nsamp2 - 1, 1)

            x.append(row_idx + 1)

        fracs.append(frac)

    return fracs, x


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat, ids1=None, ids2=None):
    """
    Domain-averaged FOSCTTM.

    If ids1 and ids2 are None:
        preserves the original one-to-one behavior:
        returns one averaged score per paired sample.

    If ids1 and ids2 are provided:
        supports one-to-many / many-to-many correspondences.

        Since domains may have different sizes, index-wise averaging is
        not meaningful. We return the concatenated directional scores:
            x1 -> x2 scores followed by x2 -> x1 scores.

        Downstream code using np.mean(fracs) still works.
    """
    fracs1, _ = calc_frac_idx(x1_mat, x2_mat, ids1=ids1, ids2=ids2)
    fracs2, _ = calc_frac_idx(x2_mat, x1_mat, ids1=ids2, ids2=ids1)

    if ids1 is None and ids2 is None:
        fracs = []
        for i in range(len(fracs1)):
            fracs.append((fracs1[i] + fracs2[i]) / 2)
        return fracs

    return fracs1 + fracs2
    

def test_transfer_accuracy(
    data1,
    data2,
    type1,
    type2,
    return_classwise_probabilities=False,
):
    """
    Metric from UnionCom: "Label Transfer Accuracy"

    If return_classwise_probabilities is True, returns a dictionary with:
        - accuracy: label transfer accuracy
        - predicted_labels: predicted labels for data1
        - classwise_probabilities: per-sample class probabilities for data1
        - classes: class order for the probability matrix columns

    By default, returns only the scalar accuracy for backwards compatibility.
    """
    Min = np.minimum(len(data1), len(data2))
    k = np.maximum(10, (len(data1) + len(data2))*0.01)
    k = k.astype(int)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data2, type2)
    type1_predict = knn.predict(data1)
    # np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    accuracy = count / len(type1)

    if return_classwise_probabilities:
        return {
            "accuracy": accuracy,
            "predicted_labels": type1_predict,
            "classwise_probabilities": knn.predict_proba(data1),
            "classes": knn.classes_,
        }

    return accuracy


def test_alignment_score(
    data1_shared,
    data2_shared,
    data1_specific=None,
    data2_specific=None,
    random_state=None,
):

    N = 2
    rng = _alignment_score_rng(random_state)

    if len(data1_shared) < len(data2_shared):
        data1 = data1_shared
        data2 = data2_shared
    else:
        data2 = data1_shared
        data1 = data2_shared
    data2 = data2[rng.sample(range(len(data2)), len(data1))]
    k = np.maximum(10, (len(data1) + len(data2))*0.01)
    k = k.astype(int)

    data = np.vstack((data1, data2))

    bar_x1 = 0
    for i in range(len(data1)):
        diffMat = data1[i] - data
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k+1]
        for j in NearestN:
            if j < len(data1):
                bar_x1 += 1
    bar_x1 = bar_x1 / len(data1)

    bar_x2 = 0
    for i in range(len(data2)):
        diffMat = data2[i] - data
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k+1]
        for j in NearestN:
            if j >= len(data1):
                bar_x2 += 1
    bar_x2 = bar_x2 / len(data2)

    bar_x = (bar_x1 + bar_x2) / 2

    score = 0
    score += 1 - (bar_x - k/N) / (k - k/N)

    data_specific = None
    flag = 0
    if data1_specific is not None:
        data_specific = data1_specific
        if data2_specific is not None:
            data_specific = np.vstack((data_specific, data2_specific))
            flag=1
    else:
        if data2_specific is not None:
            data_specific = data2_specific

    if data_specific is None:
        return score
    else:
        bar_specific1 = 0
        bar_specific2 = 0
        data = np.vstack((data, data_specific))
        if flag==0: # only one of data1_specific and data2_specific is not None
            for i in range(len(data_specific)):
                diffMat = data_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)):
                        bar_specific1 += 1
            bar_specific = bar_specific1
            
        else: # both data1_specific and data2_specific are not None
            for i in range(len(data1_specific)):
                diffMat = data1_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)) and j < (len(data1)+len(data2)+len(data1_specific)):
                        bar_specific1 += 1
       
            for i in range(len(data2_specific)):
                diffMat = data2_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)+len(data1_specific)):
                        bar_specific2 += 1
    
            bar_specific = bar_specific1 + bar_specific2

        bar_specific = bar_specific / len(data_specific)

        score += (bar_specific - k/N) / (k - k/N)

        return score / 2


def test_alignment_score_sparse(
    data1_shared,
    data2_shared,
    data1_specific=None,
    data2_specific=None,
    random_state=None,
):
    def get_bar(query, full_data, k, start_idx, end_idx):
        # euclidean_distances specifically supports sparse CSR/CSC
        dist = euclidean_distances(query, full_data, squared=True)
        # Find k+1 nearest (including self)
        nn = np.argpartition(dist, k + 1, axis=1)[:, 1:k + 1]
        # Count neighbors within the specified index range
        return np.sum((nn >= start_idx) & (nn < end_idx))

    # Balance datasets
    rng = _alignment_score_rng(random_state)
    if data1_shared.shape[0] < data2_shared.shape[0]:
        data1 = data1_shared
        data2 = data2_shared[rng.sample(range(data2_shared.shape[0]), data1_shared.shape[0])]
    else:
        data1 = data2_shared
        data2 = data1_shared[rng.sample(range(data1_shared.shape[0]), data2_shared.shape[0])]

    n1, n2 = data1.shape[0], data2.shape[0]
    k = max(10, int((n1 + n2) * 0.01))
    
    # Generic vstack helper
    vstack = sparse.vstack if sparse.issparse(data1) else np.vstack
    data = vstack((data1, data2))

    # Calculate shared bars
    bar_x1 = get_bar(data1, data, k, 0, n1) / n1
    bar_x2 = get_bar(data2, data, k, n1, n1 + n2) / n2
    
    score = 1 - (((bar_x1 + bar_x2) / 2) - k/2) / (k - k/2)

    # Specific data handling
    specs = [d for d in [data1_specific, data2_specific] if d is not None]
    if not specs:
        return score

    data_spec = vstack(specs)
    full_data = vstack((data, data_spec))
    
    n_spec1 = data1_specific.shape[0] if data1_specific is not None else 0
    n_spec2 = data2_specific.shape[0] if data2_specific is not None else 0
    offset = n1 + n2

    if data1_specific is not None and data2_specific is not None:
        b1 = get_bar(data1_specific, full_data, k, offset, offset + n_spec1)
        b2 = get_bar(data2_specific, full_data, k, offset + n_spec1, offset + n_spec1 + n_spec2)
        bar_spec = (b1 + b2) / (n_spec1 + n_spec2)
    else:
        bar_spec = get_bar(data_spec, full_data, k, offset, offset + data_spec.shape[0]) / data_spec.shape[0]

    score_spec = (bar_spec - k/2) / (k - k/2)
    return (score + score_spec) / 2
