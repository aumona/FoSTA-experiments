import os
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.datasets import fetch_openml
from scipy.ndimage import rotate

def print_mat_stats(name, M):
    """
    Print basic mass / row / col statistics for a matrix.
    Works for dense numpy arrays and scipy sparse matrices.
    """
    def _sparse_value_stats(data, total_count):
        data = np.asarray(data, dtype=float)
        implicit_zeros = int(total_count - data.size)

        if total_count <= 0:
            return dict(min=np.nan, median=np.nan, max=np.nan, mean=np.nan)

        if data.size == 0:
            return dict(min=0.0, median=0.0, max=0.0, mean=0.0)

        val_min = min(float(data.min()), 0.0) if implicit_zeros > 0 else float(data.min())
        val_max = max(float(data.max()), 0.0) if implicit_zeros > 0 else float(data.max())
        val_mean = float(data.sum() / total_count)

        explicit_zeros = int(np.sum(data == 0))
        zero_count = implicit_zeros + explicit_zeros
        nonzero_data = np.sort(data[data != 0])
        neg_count = int(np.sum(nonzero_data < 0))

        def kth_value(k):
            if k < neg_count:
                return float(nonzero_data[k])
            if k < neg_count + zero_count:
                return 0.0
            return float(nonzero_data[k - zero_count])

        mid = total_count // 2
        if total_count % 2:
            val_median = kth_value(mid)
        else:
            val_median = 0.5 * (kth_value(mid - 1) + kth_value(mid))

        return dict(min=val_min, median=val_median, max=val_max, mean=val_mean)

    def _print_distribution(label, values=None, stats=None):
        if stats is None:
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                stats = dict(min=np.nan, median=np.nan, max=np.nan, mean=np.nan)
            else:
                stats = {
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                }
        print(
            f"  {label:<12}: min={stats['min']:.4f}, median={stats['median']:.4f}, "
            f"max={stats['max']:.4f}, mean={stats['mean']:.4f}"
        )

    def _dense_value_stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            stats = {
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
                "mean": np.nan,
            }
            return stats
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }

    # Total mass
    total_mass = M.sum()

    # Row / column sums
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    col_sums = np.asarray(M.sum(axis=0)).ravel()

    n_rows, n_cols = M.shape
    n_values = n_rows * n_cols
    n_diag = min(n_rows, n_cols)
    n_offdiag = n_values - n_diag

    # Value distributions include implicit sparse zeros without densifying.
    if sparse.issparse(M):
        all_value_stats = _sparse_value_stats(M.data, n_values)
        diag_values = np.asarray(M.diagonal(), dtype=float)

        coo = M.tocoo()
        offdiag_data = coo.data[coo.row != coo.col]
        offdiag_stats = _sparse_value_stats(offdiag_data, n_offdiag)
    else:
        M = np.asarray(M)
        all_value_stats = _dense_value_stats(M.ravel())
        diag_values = np.diag(M)
        offdiag_stats = _dense_value_stats(M[~np.eye(n_rows, n_cols, dtype=bool)])

    print(f"\n{name}:")
    print(f"  Total mass : {total_mass:.4f}")
    _print_distribution("Row sums", row_sums)
    _print_distribution("Col sums", col_sums)
    if sparse.issparse(M):
        _print_distribution("Values", stats=all_value_stats)
        _print_distribution("Diagonal", diag_values)
        _print_distribution("Off-diagonal", stats=offdiag_stats)
    else:
        _print_distribution("Values", stats=all_value_stats)
        _print_distribution("Diagonal", diag_values)
        _print_distribution("Off-diagonal", stats=offdiag_stats)
    print(f"  Non-zero entries: {100.0 * (M.nnz if sparse.issparse(M) else np.count_nonzero(M)) / n_values:.4f}%")


def kernel2Dist(K):
    D = np.diag(np.diag(K))
    Di = np.linalg.inv(D)
    Kn = np.dot(Di**(1/2), np.dot(K, Di**(1/2)))
    Kn = np.clip(Kn, a_min = 0, a_max = 1)
    return 1 - Kn

def dataprep(data, label_col_idx=0, transform='standardize', global_transform=False, cat_to_numeric=True):
    """
    This method normalizes or standardizes all non-categorical variables in an array.
    All categorical variables are kept.
    Categorical variables as supposed to be ordered, so that we can transform them to numerical
    and standardize them the same way as continuous variables to be on the same scale.
    
    If transform = "standardize", categorical variables are standardized.
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

def load_paired_mnist_rotated(n_samples=1000, rotation_range=(30, 30), seed=42):
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

    # Scale pixel values to [0, 1]
    X_source_flat = X_source_flat.astype(float) / 255.0
    X_target_flat = X_target_flat.astype(float) / 255.0
    
    return X_source_flat, labels, X_target_flat, labels




def load_paired_mnist_sensor_defect(n_samples=1000, noise_ratio=0.2, seed=42):
    """
    Simulates a 'Sensor Defect' (Impulsive Noise) scenario.
    
    Source = Clean Images
    Target = Images with 'Dead' (0) and 'Hot' (1) pixels.
    
    Realistic Scenario:
    - Training data is gathered in a controlled environment (clean).
    - Deployment data comes from a cheap or damaged sensor (noisy).
    
    Why RF wins:
    - kNN L2 distance explodes because of the high-contrast noise spikes.
    - RF survives because the majority of trees will split on non-corrupted pixels.
    """
    print("Fetching MNIST...")
    try:
        X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        return None, None, None, None
        
    y_raw = y_raw.astype(int)
    rng = np.random.RandomState(seed)
    
    # 1. Subsample indices
    indices = rng.choice(len(X_raw), n_samples, replace=False)
    
    # 2. Source: Clean Images (0-1)
    X_clean = X_raw[indices].astype(float) / 255.0
    labels = y_raw[indices]
    
    # 3. Target: Copy Source and Add Salt & Pepper Noise
    print(f"Corrupting {noise_ratio:.0%} of target pixels (Sensor Defects)...")
    X_noisy = X_clean.copy()
    
    # Generate random mask for noise locations
    n_pixels = X_noisy.size
    n_corrupt = int(n_pixels * noise_ratio)
    
    # Choose random coordinates to corrupt (flattened)
    mask_indices = rng.choice(n_pixels, n_corrupt, replace=False)
    
    # Create the noise values: 50% chance of 0 (Dead), 50% chance of 1 (Hot)
    salt_pepper = rng.choice([0.0, 1.0], size=n_corrupt)
    
    # Apply noise
    X_noisy.flat[mask_indices] = salt_pepper
    
    return X_clean, labels, X_noisy, labels

# --- Visualization Helper ---
def visualize_sensor_defect(source, target, idx=0):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.title("Source (Clean)")
    plt.imshow(source[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Target (Sensor Defect)")
    plt.imshow(target[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    s, y, t, _ = load_paired_mnist_sensor_defect(n_samples=1000, noise_ratio=0.15)
    visualize_sensor_defect(s, t, idx=0)
