import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
import random
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

def calc_frac_idx(x1_mat,x2_mat):
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

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
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


def test_alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None):

    N = 2

    if len(data1_shared) < len(data2_shared):
        data1 = data1_shared
        data2 = data2_shared
    else:
        data2 = data1_shared
        data1 = data2_shared
    data2 = data2[random.sample(range(len(data2)), len(data1))]
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


def test_alignment_score_sparse(data1_shared, data2_shared, data1_specific=None, data2_specific=None):
    def get_bar(query, full_data, k, start_idx, end_idx):
        # euclidean_distances specifically supports sparse CSR/CSC
        dist = euclidean_distances(query, full_data, squared=True)
        # Find k+1 nearest (including self)
        nn = np.argpartition(dist, k + 1, axis=1)[:, 1:k + 1]
        # Count neighbors within the specified index range
        return np.sum((nn >= start_idx) & (nn < end_idx))

    # Balance datasets
    if data1_shared.shape[0] < data2_shared.shape[0]:
        data1, data2 = data1_shared, data2_shared[random.sample(range(data2_shared.shape[0]), data1_shared.shape[0])]
    else:
        data2, data1 = data1_shared, data2_shared[random.sample(range(data2_shared.shape[0]), data1_shared.shape[0])]

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