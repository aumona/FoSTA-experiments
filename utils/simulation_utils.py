import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# UTILS FOR THE SIMULATED BATCHES BENCHMARK -------------------------

def split_and_transform_batch(adata, noise_std=1, dropout_prob=0.3,new_batch_key="simulated_batch", embedding_basis="X", seed=42):
    # split and transformation on one batch
    # args:
    #   noise_std: standard deviation of the Gaussian noise to add
    #   dropout_prob: probability of dropout to apply
    #   new_batch_key: key to use to store the membership of each sample to the new simulated batches (in adata.obs). 
    #   embedding_basis: basis to apply the transformations on (e.g. "X_pca" or "X")
    # returns: modified adata with a new column in adata.obs with the new batch info, and the second half of the batch transformed with noise and dropout
    
    n = len(adata)
    idxs = np.arange(n)
    rng = np.random.default_rng(seed= seed)
    idxs = rng.permutation(idxs)

    # split this batch in half
    idxs0 = idxs[0:n//2]
    idxs1 = idxs[n//2:]

    # create a new column with the new batch info
    adata.obs[new_batch_key] = "batch_0"
    pandas_idxs1 = adata.obs.index[idxs1]
    adata.obs.loc[pandas_idxs1, new_batch_key] = "batch_1"

    # apply transformations to the embedding of the second half     
    if embedding_basis =="X":   
        adata.X[idxs1] = add_noise(adata.X[idxs1], noise_std, random_state=seed)
        adata.X[idxs1] = dropout(adata.X[idxs1], dropout_prob, random_state=seed)
    else:
        adata.obsm[embedding_basis][idxs1] = add_noise(adata.obsm[embedding_basis][idxs1], noise_std, random_state=seed)
        adata.obsm[embedding_basis][idxs1] = dropout(adata.obsm[embedding_basis][idxs1], dropout_prob, random_state=seed)

    return adata


def split_and_transform_batch_stratified(adata, label_key="cell_type", noise_std=1, dropout_prob=0.3, 
                                        new_batch_key="simulated_batch", embedding_basis="X", seed=42):
    rng = np.random.default_rng(seed=seed)
    
    # Initialize the new batch column
    adata.obs[new_batch_key] = "batch_0"
    
    # We will collect all indices destined for batch_1 (the transformed batch)
    all_idxs1 = []
    
    unique_classes = adata.obs[label_key].unique()
    
    for cls in unique_classes:
        # Get integer positional indices for this class
        cls_indices = np.where(adata.obs[label_key] == cls)[0]
        n_cls = len(cls_indices)
        
        # Shuffle the class-specific indices
        shuffled_cls_indices = rng.permutation(cls_indices)
        
        # Split in half (ensuring at least 1 in each if n >= 2)
        split_idx = max(1, n_cls // 2)
        # Note: If n_cls is 1, it stays in batch_0 by default. 
        # If you want it in batch_1, you'd adjust this logic.
        
        cls_batch1 = shuffled_cls_indices[split_idx:]
        all_idxs1.extend(cls_batch1)
    
    # Convert list to array for advanced indexing
    all_idxs1 = np.array(all_idxs1)
    
    # Update the obs column using index names
    batch1_index_names = adata.obs.index[all_idxs1]
    adata.obs.loc[batch1_index_names, new_batch_key] = "batch_1"

    # Apply transformations to the embedding of the second half (batch_1)
    if embedding_basis == "X":
        # Ensure we are working with the actual data (handling sparse matrices if necessary)
        # add_noise and dropout need to handle the specific data type
        adata.X[all_idxs1] = add_noise(adata.X[all_idxs1], noise_std, random_state=seed)
        adata.X[all_idxs1] = dropout(adata.X[all_idxs1], dropout_prob, random_state=seed)
    else:
        adata.obsm[embedding_basis][all_idxs1] = add_noise(adata.obsm[embedding_basis][all_idxs1], 
                                                          noise_std, random_state=seed)
        adata.obsm[embedding_basis][all_idxs1] = dropout(adata.obsm[embedding_basis][all_idxs1], 
                                                         dropout_prob, random_state=seed)

    return adata

def global_label_masking(adata, masking_frac = 0.5, label_key="cell_type", batch_key="batch", random_state=42):
    '''
    Does a masking of the labels per batch, stratified over classes.
    Ensures at least one sample per class per batch remains unmasked.
    Stores the masked labels in a new column in adata.obs named "{label_key}_masked"
    creates a column "mask_indices" in adata.obs that indicates which samples were masked (1 if masked, 0 if not masked)
    '''
    
    adata.uns["global_masking_fraction"] = masking_frac  # store the masking fraction in adata.uns for reference
    rng = np.random.default_rng(seed=random_state)
    new_col = f"{label_key}_masked"
    
    # Initialize with original labels
    adata.obs[new_col] = adata.obs[label_key].copy()
    adata.obs['mask_indices'] = 0  # Initialize the mask_indices column
    
    # Iterate through each batch
    for batch in adata.obs[batch_key].unique():
        batch_indices = adata.obs[adata.obs[batch_key] == batch].index
        
        # Get unique classes present in THIS batch
        batch_classes = adata.obs.loc[batch_indices, label_key].unique()
        
        for cls in batch_classes:
            # Identify indices for this specific class in this specific batch
            cls_batch_indices = adata.obs[(adata.obs[batch_key] == batch) & 
                                          (adata.obs[label_key] == cls)].index
            
            n_total = len(cls_batch_indices)
            
            # Calculate masking count (guaranteeing 1 remains)
            n_to_mask = int(masking_frac * n_total)
            n_to_mask = min(n_to_mask, n_total - 1)
            
            if n_to_mask > 0:
                cur_mask_indices = rng.choice(cls_batch_indices, size=n_to_mask, replace=False)
                # Ensure "Unknown" is a valid category
                col = adata.obs[new_col]
                if isinstance(col.dtype, pd.CategoricalDtype):
                    if "Unknown" not in col.cat.categories:
                        adata.obs[new_col] = col.cat.add_categories(["Unknown"])

                # Apply the "Unknown" label to the selected indices
                adata.obs.loc[cur_mask_indices, new_col] = "Unknown"
                adata.obs.loc[cur_mask_indices, 'mask_indices'] = 1  # add a column to indicate which samples were masked so we can retrieve it
                # mask_indices = np.concatenate([mask_indices, cur_mask_indices])
                # mask_indices.extend(cur_mask_indices.tolist())
        # mask_indices = list(set(mask_indices))
    return adata
    
    
def ensure_label_intersection(adata, labels=None, label_key=None, batch_key="batch", replace_by = "Unknown"): 
    # updates adata[label_key] to ensures that all labels are present in both batches by removing samples of labels that are missing in one batch
    # the labels that are not shared are set to Unknown
    # can either provide labels as a series, or a label_key to use from adata.obs. if labels is provided, label_key is ignored. if labels is not provided, label_key must be provided and must be a column in adata.obs.
    # replace_by: value to replace the non-shared labels with. default is "Unknown". can be set to np.nan or -1 if you want to use a numeric label for the non-shared labels.
    
    
    batch1 = adata.obs[batch_key].unique()[0]
    try:
        batch2 = adata.obs[batch_key].unique()[1]
    except:
        print(f"Only one batch found in adata.obs[{batch_key}]. No need to ensure label intersection.")
        return labels, set(), set()
        
    if labels is None:
        labels = adata.obs[label_key]
    # find labels that are missing in one of the batches
    labels_missing_in_batch1 = set(labels[adata.obs[batch_key]==batch2]) - set(labels[adata.obs[batch_key]==batch1])   
    labels_missing_in_batch2 = set(labels[adata.obs[batch_key]==batch1]) - set(labels[adata.obs[batch_key]==batch2])  

    ## remove all cells that belong to these labels
    # adata = adata[~adata.obs[label_key].isin(labels_missing_in_batch1.union(labels_missing_in_batch2))]
    
    # replace these labels with "Unknown" instead of removing the cells, so that we can still run methods that can handle missing labels and compare with those that can't
    new_labels = labels.replace(list(labels_missing_in_batch1.union(labels_missing_in_batch2)), replace_by)
    
    if len(labels_missing_in_batch1) > 0 or len(labels_missing_in_batch2) > 0:
        print(f"Labels missing in batch 1: {labels_missing_in_batch1}")
        print(f"Labels missing in batch 2: {labels_missing_in_batch2}")
        print(f"Replacing these labels with '{replace_by}' in the {label_key} column")
    
    return new_labels, labels_missing_in_batch1, labels_missing_in_batch2



def clean_and_encode_labels(adata, 
                            label_key="cell_type", 
                            batch_key=None,
                            encoded_label_key = "cell_type_cleaned_encoded", 
                            min_cells=20):
    # cell types with fewer than 20 cells are set as -1 (only for in our methods, others use original column "cell_type")
    # args:
    #   adata: anndata object
    #   label_key: key in adata.obs with the original labels
    #   new_label_key: key in adata.obs to store the cleaned and encoded labels
    #   min_cells: minimum number of cells required to keep a cell type
    # returns: adata with a new column in adata.obs, named new_label_key, containing the cleaned and encoded labels
    
    cell_types = adata.obs[label_key]
    
    # remove rare cell types
    cell_type_counts = cell_types.value_counts()
    rare_cell_types = cell_type_counts[cell_type_counts < min_cells].index
    cell_types_cleaned = cell_types.replace(rare_cell_types, np.nan) # also replace rare cell types with nan so they are encoded as -1
    
    
    # remove dataset specific cell types (those that are only present in one batch)
    # this function modifies adata.obs[clean_label_key] by replacing the dataset-specific cell types with "nan"
    if batch_key is not None:
        cell_types_cleaned, labels_missing_in_batch1, labels_missing_in_batch2 = ensure_label_intersection(adata, labels=cell_types_cleaned, batch_key=batch_key, replace_by=np.nan)
    
    # encode the labels as integers
    cell_types = cell_types_cleaned.replace("Unknown", np.nan) # replace the unknowns with nan so they are encoded as -1
    cell_types_clean_encoded, uniques = pd.factorize(cell_types_cleaned, use_na_sentinel=True, sort=True)
    adata.obs[encoded_label_key]  = cell_types_clean_encoded
    
    return adata


def remove_dataset_specific_cells(adata, batch_key, label_key):
    batch1 = adata.obs[batch_key].unique()[0]
    batch2 = adata.obs[batch_key].unique()[1]
    # find labels that are missing in one of the batches
    labels_missing_in_batch1 = set(adata.obs[label_key][adata.obs[batch_key]==batch2]) - set(adata.obs[label_key][adata.obs[batch_key]==batch1])   
    labels_missing_in_batch2 = set(adata.obs[label_key][adata.obs[batch_key]==batch1]) - set(adata.obs[label_key][adata.obs[batch_key]==batch2])  

    # remove all cells that belong to these labels
    adata = adata[~adata.obs[label_key].isin(labels_missing_in_batch1.union(labels_missing_in_batch2))]
    return adata, labels_missing_in_batch1, labels_missing_in_batch2


def preprocess_adata(adata, batch_key, n_top_genes=2000, n_pcs=30):
    # find the 2000 highly variable genes and compute PCA. subset the adata to the highly variable genes.
    # if pca is 0, do not run pca
    # if n_top_genes is 0, do not run hvg selection
    
    # PCA AND HVG SELECTION
    if n_top_genes > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="cell_ranger", batch_key= batch_key)
    if n_pcs > 0:
        sc.tl.pca(adata, n_comps=n_pcs, mask_var="highly_variable", svd_solver='arpack')  # svd_solver='arpack' is for reproducibility, more stable than randomized

    # subset to the highly variable genes so that each method has the same input.
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    return adata




# UTILS FOR THE SIMULATED MULTIMODAL BENCHMARK -------------------------

def add_noise(batch, sigma=0.1, random_state=42):
    rng = np.random.default_rng(seed=random_state)
    noise = rng.normal(0, sigma, batch.shape)
    return batch + noise


def random_rotate(batch, random_state=42):
    rng = np.random.default_rng(seed=random_state)
    n_features = batch.shape[1]
    # generate a random rotation matrix using QR decomposition
    random_matrix = rng.normal(size=(n_features, n_features))
    q, r = np.linalg.qr(random_matrix)
    rotation_matrix = q
    return np.dot(batch, rotation_matrix)

def dropout(batch, drop_prob=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    mask = rng.binomial(1, 1 - drop_prob, size=batch.shape)
    try:
        return batch * mask
    except Exception:
        return batch.multiply(mask)
    
    
    
def random_feature_split(df, random_state=42):
    # splits dataset randomly into two halves feature-wise
    # df is assumed to not contain the target variable
    # returns df1 and df2 containing half the features each

    n_features = df.shape[1]
    df1 = df.sample(n = int(n_features/2), axis=1, random_state=random_state)
    df2 = df.drop(df1.columns, axis=1)

    return df1, df2

def get_feature_importances(df, labels):
    rf = RandomForestRegressor(n_estimators=100) # train a RF to get feature importances
    rf.fit(df, labels)
    importances = rf.feature_importances_

    # sort the features by importance
    feature_names = df.columns
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)
    return feature_importances

def importance_split(df, labels):
    # splits dataset into important and not important features
    # df is assumed to not contain the target variable (labels)
    # returns:  df1 containing the most important features
    #           df2 containing the least important features
    
    feature_importances = get_feature_importances(df, labels)
    
    n_features = df.shape[1]
    top_features = feature_importances.index[:int(n_features/2)]
    bottom_features = feature_importances.index[int(n_features/2):]
    
    df1 = df[top_features]
    df2 = df[bottom_features]
    
    return df1, df2



def alternating_importance_split(df, labels):
    # splits dataset into two so that both have a mix of important and less important features
    # df is assumed to not contain the target variable
    # returns df1 and df2 containing half the features
    
    feature_importances = get_feature_importances(df, labels)
    features1 = feature_importances.index[::2]
    features2 = feature_importances.index[1::2]
    
    df1 = df[features1]
    df2 = df[features2]
    
    return df1, df2

def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())


def add_gaussian_noise_features_split(
    df,
    signal_to_noise_ratio=0.1,
    sigma=1.0,
    random_state=42,
):
    # Creates two domains:
    # - df1: signal features only
    # - df2: signal features + Gaussian noise features
    # Assumes df is already standardized.

    n_samples = df.shape[0]
    n_noise_features = max(1, int((1 / signal_to_noise_ratio) * df.shape[1]))

    rng = np.random.default_rng(seed=random_state)
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(n_samples, n_noise_features),
    )

    noise_df = pd.DataFrame(
        noise,
        index=df.index,
        columns=[f"noise_{i}" for i in range(n_noise_features)],
    )

    df2 = pd.concat([df.copy(), noise_df], axis=1)
    return df.copy(), df2


def mask_labels(labels, mask_fraction, random_state=42):
    # masks a fraction of the labels by setting them to -1
    rng = np.random.default_rng(seed=random_state)
    mask_indices = rng.choice(labels.shape[0], size = (int(mask_fraction * len(labels))), replace=False)
    labels_masked = labels.copy()
    labels_masked[mask_indices] = -1  # or np.nan
    return labels_masked


def mask_labels_stratified(labels, mask_fraction, random_state=42):
    rng = np.random.default_rng(seed=random_state)
    labels_masked = np.copy(labels)
    
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        # Get indices belonging to the current class
        cls_indices = np.where(labels == cls)[0]
        n_total = len(cls_indices)
        
        # Calculate mask size: 
        # We ensure at least 1 remains by capping the mask size at (n_total - 1)
        n_to_mask = int(mask_fraction * n_total)
        n_to_mask = min(n_to_mask, n_total - 1)
        
        if n_to_mask > 0:
            mask_indices = rng.choice(cls_indices, size=n_to_mask, replace=False)
            labels_masked[mask_indices] = -1
            
    return labels_masked
