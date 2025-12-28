import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.datasets import fetch_openml
from scipy.ndimage import rotate

def kernel2Dist(K):
    D = np.diag(np.diag(K))
    Di = np.linalg.inv(D)
    Kn = np.dot(Di**(1/2), np.dot(K, Di**(1/2)))
    Kn = np.clip(Kn, a_min = 0, a_max = 1)
    return 1 - Kn

def dataprep(data, label_col_idx=0, transform='normalize', global_transform=False, cat_to_numeric=True):
    """
    This method normalizes or standardizes all non-categorical variables in an array.
    All categorical variables are kept.
    Categorical variables as supposed to be ordered, so that we can transform them to numerical
    and standardize them the same way as continuous variables to be on the same scale.
    
    If transform = "normalize", categorical variables are scaled from 0 to 1. The highest value
    is assigned the value of 1, the lowest value is assigned the value of 0.
    If global_transform = True, the normalization is done globally (useful for image-like data), otherwise it is done feature-wise
                                                                                                    (useful for tabular data).
    """

    data = data.copy()
    categorical_cols = []

    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64':
            categorical_cols.append(col)

    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y = data.pop(label)
        x = data
    else:
        x = data
        y = None

    for col in x.columns:
        if col in categorical_cols and cat_to_numeric:
            x[col] = pd.Categorical(x[col]).codes

    # Ensure all features in x are floats
    x = x.astype(float)
    if not global_transform:
        if transform == 'standardize':
            for col in x.columns:
                std_dev = x[col].std()
                if std_dev == 0:  # Handle constant feature
                    x[col] = 0
                else:
                    x[col] = (x[col] - x[col].mean()) / std_dev
        elif transform == 'normalize':
            for col in x.columns:
                range_val = x[col].max() - x[col].min()
                if range_val == 0:  # Handle constant feature
                    x[col] = 0
                else:
                    x[col] = (x[col] - x[col].min()) / range_val
    else:
        if transform == 'standardize':
            global_mean = x.values.mean()
            global_std = x.values.std()
            denom = global_std if global_std != 0 else 1
            x = (x - global_mean) / denom
        elif transform == 'normalize':
            global_min = x.values.min()
            global_max = x.values.max()
            denom = global_max - global_min if global_max != global_min else 1
            x = (x - global_min) / denom                    
    if y is not None:
        y_encoded = LabelEncoder().fit_transform(y)

    return x, y_encoded
    

def load_data(data_path, data_name, processing=True, transform='normalize', global_transform=False):
    '''
    Load data from a path. Automatically handles cases where a single CSV file or paired _train and _test CSV files are provided.
    '''
    train_file = os.path.join(data_path, data_name + "_train.csv")
    test_file = os.path.join(data_path, data_name + "_test.csv")
    single_file = os.path.join(data_path, data_name + ".csv")

    if os.path.exists(train_file) and os.path.exists(test_file):
        # Case 1: Paired _train and _test CSVs
        train_data = pd.read_csv(train_file, sep=',')
        test_data = pd.read_csv(test_file, sep=',')
        # Concatenate train and test datasets
        data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
        n_train = train_data.shape[0]
    elif os.path.exists(single_file):
        # Case 2: Single CSV file
        data = pd.read_csv(single_file, sep=',')
        n_train = None
    else:
        # Raise an error if neither case is satisfied
        raise FileNotFoundError(f"Neither '{data_name}.csv' nor '{data_name}_train.csv' and '{data_name}_test.csv' found in '{data_path}'.")

    if processing:
        X, y = dataprep(data, label_col_idx=0, transform=transform, global_transform=global_transform, cat_to_numeric=True)

        # Convert X and y to NumPy arrays
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
    else:
        X, y = data.iloc[:, 1:], data.iloc[:, 0]

    return X, y, n_train


def load_paired_mnist_rotated(n_samples=1000, rotation_range=(30, 90), seed=42):
    """
    Selects n_samples images.
    Source = Original Images
    Target = SAME Images, but rotated.
    Ground Truth = Identity Matrix.
    """
    print("Fetching MNIST (caching enabled)...")
    try:
        X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        return None, None, None, None
        
    y_raw = y_raw.astype(int)
    rng = np.random.RandomState(seed)
    
    # Subsample indices
    indices = rng.choice(len(X_raw), n_samples, replace=False)
    
    # Raw Images (28x28)
    X_imgs = X_raw[indices].reshape(-1, 28, 28)
    labels = y_raw[indices]
    
    # Source: Original Flattened
    X_source_flat = X_imgs.reshape(n_samples, -1)
    
    # Target: Rotate the SAME images
    print(f"Rotating target images between {rotation_range[0]} and {rotation_range[1]} degrees...")
    X_tgt_rot = []
    for img in X_imgs:
        angle = rng.uniform(rotation_range[0], rotation_range[1])
        # Reshape=False keeps it 28x28 (crops corners if needed)
        X_tgt_rot.append(rotate(img, angle, reshape=False, mode='nearest'))
    
    X_target_flat = np.array(X_tgt_rot).reshape(n_samples, -1)
    
    return X_source_flat, labels, X_target_flat, labels