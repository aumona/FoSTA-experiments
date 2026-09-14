"""
HAR/AVE/RGBD/Sketchy multimodal alignment benchmark.

Each data table stores labels in the first column and modality features in the
remaining columns. Train rows stay labeled; test rows are masked as -1.
"""
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
np.int = int
import pandas as pd
import phate
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

from src.fosta import FoSTA
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf
from src.mali import MALI
from src.Pamona.eval import (
    calc_domainAveraged_FOSCTTM,
    test_alignment_score,
    test_transfer_accuracy,
)
from src.pamona import Pamona
from experiment_utils import (
    DEFAULT_MEMORY_SAMPLE_INTERVAL_SEC,
    append_result_row,
    coerce_embedding_array,
    profile_fit_transform,
    save_embedding_plots,
    save_embeddings,
    seed_everything,
    write_json,
)


# =============================================================================
# CONFIG
# =============================================================================
DATASETS = [
    "sketchy_resnet18",
    "sketchy_dinov2base",
    "ave", 
    "har", 
    "rgbd_resnet18",
    "rgbd_dinov2base",
]  # Any of: "har", "ave", "rgbd_resnet18", "rgbd_dinov2base", "sketchy_resnet18", "sketchy_dinov2base"

HAR_DATA_ROOT = PROJECT_ROOT / "data_har"
HAR_ACC_TRAIN_PATH = HAR_DATA_ROOT / "har_acc_total_train.pkl"
HAR_ACC_TEST_PATH = HAR_DATA_ROOT / "har_acc_total_test.pkl"
HAR_GYRO_TRAIN_PATH = HAR_DATA_ROOT / "har_gyro_train.pkl"
HAR_GYRO_TEST_PATH = HAR_DATA_ROOT / "har_gyro_test.pkl"

RGBD_DATA_ROOT = PROJECT_ROOT / "data_rgbd"
RGBD_RESNET18_PHOTO_PATH = RGBD_DATA_ROOT / "rgbd_photo_resnet18_embeddings.npy"
RGBD_RESNET18_DEPTH_PATH = RGBD_DATA_ROOT / "rgbd_depth_resnet18_embeddings.npy"
RGBD_DINOV2BASE_PHOTO_PATH = RGBD_DATA_ROOT / "rgbd_photo_dinov2base_embeddings.npy"
RGBD_DINOV2BASE_DEPTH_PATH = RGBD_DATA_ROOT / "rgbd_depth_dinov2base_embeddings.npy"
RGBD_LABEL_MAP_PATH = RGBD_DATA_ROOT / "rgbd_label_map.json"
RGBD_TRAIN_FRACTION = 0.50
STD_NORMALIZE_RGBD = False   # True works well for DINO, but False is in accordance with MALI and common practice to visualize neural network features. Also, False better for ResNet18

SKETCHY_DATA_ROOT = PROJECT_ROOT / "data_sketchy"
SKETCHY_RESNET18_PHOTO_PATH = SKETCHY_DATA_ROOT / "photo_resnet18_embeddings.npy"
SKETCHY_RESNET18_SKETCH_PATH = SKETCHY_DATA_ROOT / "sketch_resnet18_embeddings.npy"
SKETCHY_DINOV2BASE_PHOTO_PATH = SKETCHY_DATA_ROOT / "photo_dinov2base_embeddings.npy"
SKETCHY_DINOV2BASE_SKETCH_PATH = SKETCHY_DATA_ROOT / "sketch_dinov2base_embeddings.npy"
SKETCHY_LABEL_DICT_PATH = SKETCHY_DATA_ROOT / "label_dic"
LABEL_MASKING_SKETCH_TARGET = 0.50

AVE_DATA_ROOT = PROJECT_ROOT / "data_ave"
AVE_AUDIO_TRAIN_PATH = AVE_DATA_ROOT / "train_audio_feature.npy"
AVE_AUDIO_TEST_PATH = AVE_DATA_ROOT / "test_audio_feature.npy"
AVE_VIDEO_TRAIN_PATH = AVE_DATA_ROOT / "train_visual_feature.npy"
AVE_VIDEO_TEST_PATH = AVE_DATA_ROOT / "test_visual_feature.npy"

SEEDS = [11784, 39041, 56089, 79121, 4386721]
MAX_SAMPLE_BY_DATASET = {
    "sketchy_resnet18": None,
    "sketchy_dinov2base": None,
    "ave": None,
    "har": None,
    "rgbd_resnet18": 15000,
    "rgbd_dinov2base": 15000,
}  # Set a dataset value to None to disable deterministic stratified subsampling.
N_COMPONENTS = 2
N_JOBS = -1
# Max seconds to allow a model `fit_transform` to run. Set to None to disable timeout.
MAX_FIT_TRANSFORM_SEC = None  # in seconds, set to None to disable

LABEL_TRANSFER_TOP_KS = (1, 5, 10)


MODELS_TO_RUN = [
    "Unintegrated",
    # "Unintegrated_PHATE",
    "FoSTA_t2",
    "FoSTA_tauto",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    # "Pamona",  # Not running Pamona for RGB-D 15k due to exaggerated runtime; can be enabled if desired and resources allow
]

FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "t": 2,
        "n_jobs": N_JOBS,
    },
    "FoSTA_tauto": {
        "t": 'auto',
        "n_jobs": N_JOBS,
    }
}

PAMONA_CONFIG = {
    "epsilon": 0.1,  # We bump it for faster convergence, but 0.001 is the default in the repo
    "max_iter": 100,  # We decrease for faster results, but 1000 is the default in the repo
    "tol": 1e-5,   # We increase for faster convergence, but 1e-9 is the default in the repo
}

SUPERVISED_CLASSES = {
    "FoSTA": FoSTA,
    "KEMAlin": KEMAlin,
    "KEMArbf": KEMArbf,
    "MALI": MALI,
    "Pamona": Pamona,
}

DATASET_CONFIGS = {
    "har": {
        "domain_a_name": "har_acc",
        "domain_b_name": "har_gyro",
        "kind": "har_pickle_train_test",
        "domain_a_train_path": HAR_ACC_TRAIN_PATH,
        "domain_a_test_path": HAR_ACC_TEST_PATH,
        "domain_b_train_path": HAR_GYRO_TRAIN_PATH,
        "domain_b_test_path": HAR_GYRO_TEST_PATH,
        "results_dir": PROJECT_ROOT / "results_har",
    },
    "rgbd_resnet18": {
        "domain_a_name": "rgbd_photo",
        "domain_b_name": "rgbd_depth",
        "embedding_name": "resnet18",
        "kind": "npy_single_file",
        "domain_a_path": RGBD_RESNET18_PHOTO_PATH,
        "domain_b_path": RGBD_RESNET18_DEPTH_PATH,
        "results_dir": PROJECT_ROOT / "results_rgbd" / "resnet18",
    },
    "rgbd_dinov2base": {
        "domain_a_name": "rgbd_photo",
        "domain_b_name": "rgbd_depth",
        "embedding_name": "dinov2base",
        "kind": "npy_single_file",
        "domain_a_path": RGBD_DINOV2BASE_PHOTO_PATH,
        "domain_b_path": RGBD_DINOV2BASE_DEPTH_PATH,
        "results_dir": PROJECT_ROOT / "results_rgbd" / "dinov2base",
    },
    "sketchy_resnet18": {
        "domain_a_name": "sketchy_photo",
        "domain_b_name": "sketchy_sketch",
        "embedding_name": "resnet18",
        "kind": "sketchy_npy_object_id",
        "domain_a_path": SKETCHY_RESNET18_PHOTO_PATH,
        "domain_b_path": SKETCHY_RESNET18_SKETCH_PATH,
        "results_dir": PROJECT_ROOT / "results_sketchy" / "resnet18",
    },
    "sketchy_dinov2base": {
        "domain_a_name": "sketchy_photo",
        "domain_b_name": "sketchy_sketch",
        "embedding_name": "dinov2base",
        "kind": "sketchy_npy_object_id",
        "domain_a_path": SKETCHY_DINOV2BASE_PHOTO_PATH,
        "domain_b_path": SKETCHY_DINOV2BASE_SKETCH_PATH,
        "results_dir": PROJECT_ROOT / "results_sketchy" / "dinov2base",
    },
    "ave": {
        "domain_a_name": "ave_audio",
        "domain_b_name": "ave_video",
        "kind": "npy_train_test",
        "domain_a_train_path": AVE_AUDIO_TRAIN_PATH,
        "domain_a_test_path": AVE_AUDIO_TEST_PATH,
        "domain_b_train_path": AVE_VIDEO_TRAIN_PATH,
        "domain_b_test_path": AVE_VIDEO_TEST_PATH,
        "results_dir": PROJECT_ROOT / "results_ave",
    },
}

DATASET = None
DATASET_CONFIG = None


# =============================================================================
# DATA AND BOOKKEEPING
# =============================================================================
def set_active_dataset(dataset):
    global DATASET, DATASET_CONFIG
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"DATASET must be one of {sorted(DATASET_CONFIGS)}, got {dataset!r}.")
    DATASET = dataset
    DATASET_CONFIG = DATASET_CONFIGS[dataset]


def validate_datasets():
    for dataset in DATASETS:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"DATASETS entries must be in {sorted(DATASET_CONFIGS)}, got {dataset!r}.")


def validate_max_samples():
    unknown = sorted(set(MAX_SAMPLE_BY_DATASET).difference(DATASET_CONFIGS))
    if unknown:
        raise ValueError(f"MAX_SAMPLE_BY_DATASET contains unknown datasets: {unknown}.")
    for dataset in DATASETS:
        validate_max_sample(dataset)


def validate_config():
    if DATASET_CONFIG["kind"] in {"har_pickle_train_test", "npy_train_test"}:
        paths = [
            DATASET_CONFIG["domain_a_train_path"],
            DATASET_CONFIG["domain_a_test_path"],
            DATASET_CONFIG["domain_b_train_path"],
            DATASET_CONFIG["domain_b_test_path"],
        ]
    elif DATASET_CONFIG["kind"] == "npy_single_file":
        paths = [
            DATASET_CONFIG["domain_a_path"],
            DATASET_CONFIG["domain_b_path"],
            RGBD_LABEL_MAP_PATH,
        ]
    elif DATASET_CONFIG["kind"] == "sketchy_npy_object_id":
        paths = [
            DATASET_CONFIG["domain_a_path"],
            DATASET_CONFIG["domain_b_path"],
            SKETCHY_LABEL_DICT_PATH,
        ]
    else:
        raise ValueError(f"Unknown dataset kind: {DATASET_CONFIG['kind']}")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")


def save_experiment_metadata(output_dir, timestamp):
    metadata = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp": timestamp,
        "dataset": DATASET,
        "domain_a": {
            "name": DATASET_CONFIG["domain_a_name"],
        },
        "domain_b": {
            "name": DATASET_CONFIG["domain_b_name"],
        },
        "label_column": 0,
        "feature_columns": "1:",
        "masked_rows": "test",
        "masked_label_value": -1,
        "seeds": SEEDS,
        "max_sample": get_max_sample(),
        "max_sample_by_dataset": MAX_SAMPLE_BY_DATASET,
        "n_components": N_COMPONENTS,
        "n_jobs": N_JOBS,
        "max_fit_transform_sec": MAX_FIT_TRANSFORM_SEC,
        "runtime_measurement": (
            "Wall-clock seconds measured inside the isolated worker, starting "
            "immediately before fit_transform and ending immediately after it "
            "returns."
        ),
        "peak_memory_measurement": (
            "Peak process-tree virtual-address-space increase relative to the "
            "immediately pre-fit_transform baseline, sampled every "
            f"{DEFAULT_MEMORY_SAMPLE_INTERVAL_SEC} seconds."
        ),
        "models_to_run": MODELS_TO_RUN,
        "fosta_configs": FOSTA_CONFIGS,
    }
    if DATASET_CONFIG["kind"] in {"har_pickle_train_test", "npy_train_test"}:
        metadata["domain_a"].update({
            "train_path": str(DATASET_CONFIG["domain_a_train_path"].relative_to(PROJECT_ROOT)),
            "test_path": str(DATASET_CONFIG["domain_a_test_path"].relative_to(PROJECT_ROOT)),
        })
        metadata["domain_b"].update({
            "train_path": str(DATASET_CONFIG["domain_b_train_path"].relative_to(PROJECT_ROOT)),
            "test_path": str(DATASET_CONFIG["domain_b_test_path"].relative_to(PROJECT_ROOT)),
        })
        if DATASET_CONFIG["kind"] == "npy_train_test":
            metadata["normalization"] = "none"
    elif DATASET_CONFIG["kind"] == "npy_single_file":
        metadata["domain_a"]["path"] = str(DATASET_CONFIG["domain_a_path"].relative_to(PROJECT_ROOT))
        metadata["domain_b"]["path"] = str(DATASET_CONFIG["domain_b_path"].relative_to(PROJECT_ROOT))
        metadata["rgbd_embedding_name"] = DATASET_CONFIG["embedding_name"]
        metadata["rgbd_label_map_path"] = str(RGBD_LABEL_MAP_PATH.relative_to(PROJECT_ROOT))
        metadata["rgbd_train_fraction"] = RGBD_TRAIN_FRACTION
        metadata["std_normalize_rgbd"] = STD_NORMALIZE_RGBD
        metadata["rgbd_split"] = "deterministic stratified by label in original row order"
    elif DATASET_CONFIG["kind"] == "sketchy_npy_object_id":
        metadata["domain_a"]["path"] = str(DATASET_CONFIG["domain_a_path"].relative_to(PROJECT_ROOT))
        metadata["domain_b"]["path"] = str(DATASET_CONFIG["domain_b_path"].relative_to(PROJECT_ROOT))
        metadata["sketchy_embedding_name"] = DATASET_CONFIG["embedding_name"]
        metadata["sketchy_label_dict_path"] = str(SKETCHY_LABEL_DICT_PATH.relative_to(PROJECT_ROOT))
        metadata["label_column"] = "-2"
        metadata["feature_columns"] = ":-2"
        metadata["object_id_column"] = "-1"
        metadata["masked_rows"] = "target/sketch only"
        metadata["label_masking_sketch_target"] = LABEL_MASKING_SKETCH_TARGET
        metadata["sketchy_target_sampling"] = "one random target example per source object ID, ordered to match source"
        metadata["normalization"] = "none"
    else:
        raise ValueError(f"Unknown dataset kind: {DATASET_CONFIG['kind']}")
    write_json(output_dir / "experiment_metadata.json", metadata)


class _CompatStringArray(pd.arrays.StringArray):
    def __setstate__(self, state):
        _, ndarray = state
        pd.arrays.StringArray.__init__(self, ndarray)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.arrays" and name == "StringArray":
            return _CompatStringArray
        return super().find_class(module, name)


def load_frame(path):
    try:
        df = pd.read_pickle(path)
    except NotImplementedError:
        # Older HAR pickles may store pandas StringArray state incompatible with
        # newer pandas; rebuild only that array type and leave the frame intact.
        with path.open("rb") as f:
            df = _CompatUnpickler(f).load()
    if df.ndim != 2 or df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {df.shape}")
    return df


def split_xy(df):
    labels = df.iloc[:, 0].astype(str).to_numpy()
    features = df.iloc[:, 1:].to_numpy(dtype=float)
    if np.any(~np.isfinite(features)):
        raise ValueError("Features contain non-finite values.")
    return features, labels


def split_xy_array(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with at least 2 columns in {path}, got {arr.shape}")

    labels = arr[:, 0].astype(str)
    features = np.asarray(arr[:, 1:], dtype=float)
    if np.any(~np.isfinite(features)):
        raise ValueError(f"{path.name} features contain non-finite values.")
    return features, labels


def split_xy_pickled_array(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with at least 2 columns in {path}, got {arr.shape}")

    labels = arr[:, 0].astype(str)
    features = np.asarray(arr[:, 1:], dtype=float)
    if np.any(~np.isfinite(features)):
        raise ValueError(f"{path.name} features contain non-finite values.")
    return features, labels


def coerce_integral_column(values, name):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        return values.astype(str)
    return rounded.astype(int).astype(str)


def split_xy_object_id_array(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected 2D array with at least 3 columns in {path}, got {arr.shape}")

    features = np.asarray(arr[:, :-2], dtype=float)
    labels = coerce_integral_column(arr[:, -2], f"{path.name} labels")
    object_ids = coerce_integral_column(arr[:, -1], f"{path.name} object IDs")
    if np.any(~np.isfinite(features)):
        raise ValueError(f"{path.name} features contain non-finite values.")
    return features, labels, object_ids


def load_array_labels(path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with at least 2 columns in {path}, got {arr.shape}")
    return np.asarray(arr[:, 0]).astype(str)


def make_stratified_train_mask(labels, train_fraction):
    if not 0 < train_fraction < 1:
        raise ValueError(f"RGBD_TRAIN_FRACTION must be in (0, 1), got {train_fraction}.")

    labels = np.asarray(labels).astype(str)
    train_mask = np.zeros(labels.shape[0], dtype=bool)
    for label in np.unique(labels):
        label_idx = np.flatnonzero(labels == label)
        n_train = int(np.floor(label_idx.size * train_fraction))
        if label_idx.size > 1:
            n_train = min(max(n_train, 1), label_idx.size - 1)
        else:
            n_train = 1
        train_mask[label_idx[:n_train]] = True
    return train_mask


def make_source_object_mask(source_labels, source_object_ids, target_object_ids, frac, seed):
    source_labels = np.asarray(source_labels).astype(str)
    source_object_ids = np.asarray(source_object_ids).astype(str)
    target_object_ids = np.asarray(target_object_ids).astype(str)
    if not 0 <= frac <= 1:
        raise ValueError(f"LABEL_MASKING_SKETCH_TARGET must be in [0, 1], got {frac}.")
    if frac <= 0:
        return np.zeros(target_object_ids.shape[0], dtype=bool)
    if np.unique(source_object_ids).size != source_object_ids.size:
        raise ValueError("Expected Sketchy source/photo object IDs to be unique.")

    rng = np.random.default_rng(seed)
    masked_object_ids = []
    for label in np.unique(source_labels):
        class_object_ids = source_object_ids[source_labels == label]
        n_mask = int(np.floor(class_object_ids.size * frac))
        if n_mask > 0:
            masked_object_ids.extend(rng.choice(class_object_ids, size=n_mask, replace=False))

    return np.isin(target_object_ids, masked_object_ids)


def select_one_target_per_source_object(
    x_target,
    labels_target,
    object_ids_target,
    object_ids_source,
    seed,
):
    object_ids_source_str = np.asarray(object_ids_source).astype(str)
    object_ids_target_str = np.asarray(object_ids_target).astype(str)

    if np.unique(object_ids_source_str).size != object_ids_source_str.size:
        raise ValueError("Expected Sketchy source/photo object IDs to be unique.")

    rng = np.random.default_rng(seed)
    selected_indices = []
    for object_id in object_ids_source_str:
        candidates = np.flatnonzero(object_ids_target_str == object_id)
        if candidates.size == 0:
            raise ValueError(f"Sketchy target domain has no example for source object ID {object_id!r}.")
        selected_indices.append(int(rng.choice(candidates)))

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_object_ids = np.asarray(object_ids_target)[selected_indices].astype(str)
    if not np.array_equal(selected_object_ids, object_ids_source_str):
        raise ValueError("Selected Sketchy target object IDs are not aligned with source object IDs.")

    return (
        x_target[selected_indices],
        np.asarray(labels_target)[selected_indices],
        selected_object_ids,
        selected_indices,
    )


def get_max_sample(dataset=None):
    dataset = DATASET if dataset is None else dataset
    if dataset not in MAX_SAMPLE_BY_DATASET:
        raise ValueError(f"Missing max-sample entry for dataset {dataset!r}.")
    return MAX_SAMPLE_BY_DATASET[dataset]


def validate_max_sample(dataset=None):
    dataset = DATASET if dataset is None else dataset
    max_sample = get_max_sample(dataset)
    if max_sample is None:
        return
    if not isinstance(max_sample, int) or isinstance(max_sample, bool) or max_sample <= 0:
        raise ValueError(
            f"MAX_SAMPLE_BY_DATASET[{dataset!r}] must be None or a positive int, got {max_sample!r}."
        )


def make_stratified_subsample_indices(labels, train_mask, max_sample):
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
    for stratum, n_select in zip(unique_strata, allocations):
        if n_select > 0:
            selected.append(np.flatnonzero(strata == stratum)[:n_select])
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=int)


def apply_subsample(x, y_raw, train_mask, indices):
    return x[indices], y_raw[indices], train_mask[indices]


def standardize_domain(x):
    return StandardScaler().fit_transform(x)


def preprocess_rgbd_domain(x):
    if STD_NORMALIZE_RGBD:
        return standardize_domain(x)
    return x


def make_pair_dict(
    x_a,
    y_a_raw,
    train_mask_a,
    x_b,
    y_b_raw,
    train_mask_b,
    label_encoder,
    display_classes=None,
    original_n_samples=None,
    source_size=None,
    target_size=None,
    extra_metadata=None,
):
    y_a_true = label_encoder.transform(y_a_raw)
    y_b_true = label_encoder.transform(y_b_raw)

    y_a_model = y_a_true.copy()
    y_b_model = y_b_true.copy()
    y_a_model[~train_mask_a] = -1
    y_b_model[~train_mask_b] = -1

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"Expected paired domains with equal rows, got {x_a.shape[0]} and {x_b.shape[0]}.")

    pair = {
        "x_a": x_a,
        "x_b": x_b,
        "labels_a_true": y_a_true.astype(int),
        "labels_b_true": y_b_true.astype(int),
        "labels_a_model": y_a_model.astype(int),
        "labels_b_model": y_b_model.astype(int),
        "train_mask_a": train_mask_a,
        "train_mask_b": train_mask_b,
        "original_n_samples": int(original_n_samples if original_n_samples is not None else x_a.shape[0]),
        "source_size": int(source_size if source_size is not None else x_a.shape[0]),
        "target_size": int(target_size if target_size is not None else x_b.shape[0]),
        "source_n_features": int(x_a.shape[1]),
        "target_n_features": int(x_b.shape[1]),
        "classes": label_encoder.classes_,
        "display_classes": (
            np.asarray(display_classes).astype(str)
            if display_classes is not None
            else label_encoder.classes_.astype(str)
        ),
    }
    if extra_metadata is not None:
        pair.update(extra_metadata)
    return pair


def load_rgbd_label_names(classes):
    with RGBD_LABEL_MAP_PATH.open() as f:
        name_to_id = json.load(f)

    id_to_name = {int(label_id): str(name) for name, label_id in name_to_id.items()}
    display_names = []
    for label in classes:
        try:
            label_id = int(float(label))
        except ValueError as exc:
            raise ValueError(f"RGBD label {label!r} is not numeric and cannot be mapped.") from exc

        if label_id not in id_to_name:
            raise ValueError(f"RGBD label id {label_id} is missing from {RGBD_LABEL_MAP_PATH}.")
        display_names.append(id_to_name[label_id])

    return np.asarray(display_names, dtype=str)


def load_sketchy_label_names(classes):
    with SKETCHY_LABEL_DICT_PATH.open("rb") as f:
        raw_mapping = pickle.load(f)

    code_to_name = {str(int(code)): str(name) for code, name in raw_mapping.items()}
    display_names = []
    for label in classes:
        try:
            label_key = str(int(float(label)))
        except ValueError:
            label_key = str(label)
        display_names.append(code_to_name.get(label_key, str(label)))
    return np.asarray(display_names, dtype=str)


def load_train_test_domain(train_path, test_path, split_func):
    x_train, y_train_raw = split_func(train_path)
    x_test, y_test_raw = split_func(test_path)

    x = np.vstack([x_train, x_test])
    y_raw = np.concatenate([y_train_raw, y_test_raw])

    train_mask = np.r_[np.ones(len(y_train_raw), dtype=bool), np.zeros(len(y_test_raw), dtype=bool)]
    return x, y_raw, train_mask


def split_xy_frame_path(path):
    return split_xy(load_frame(path))


def load_rgbd_domain(path, train_mask):
    x, y_raw = split_xy_array(path)
    return x, y_raw, train_mask


def build_har_pair():
    validate_max_sample()
    max_sample = get_max_sample()
    acc_train = load_frame(DATASET_CONFIG["domain_a_train_path"])
    acc_test = load_frame(DATASET_CONFIG["domain_a_test_path"])
    gyro_train = load_frame(DATASET_CONFIG["domain_b_train_path"])
    gyro_test = load_frame(DATASET_CONFIG["domain_b_test_path"])

    raw_labels = np.concatenate([
        acc_train.iloc[:, 0].astype(str).to_numpy(),
        acc_test.iloc[:, 0].astype(str).to_numpy(),
        gyro_train.iloc[:, 0].astype(str).to_numpy(),
        gyro_test.iloc[:, 0].astype(str).to_numpy(),
    ])
    label_encoder = LabelEncoder().fit(raw_labels)

    x_a, y_a_raw, train_mask_a = load_train_test_domain(
        DATASET_CONFIG["domain_a_train_path"],
        DATASET_CONFIG["domain_a_test_path"],
        split_xy_frame_path,
    )
    x_b, y_b_raw, train_mask_b = load_train_test_domain(
        DATASET_CONFIG["domain_b_train_path"],
        DATASET_CONFIG["domain_b_test_path"],
        split_xy_frame_path,
    )
    original_n_samples = len(y_a_raw)

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_a, max_sample)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)

    # Scale each modality after train/test union and optional subsampling.
    x_a = standardize_domain(x_a)
    x_b = standardize_domain(x_b)

    return make_pair_dict(
        x_a,
        y_a_raw,
        train_mask_a,
        x_b,
        y_b_raw,
        train_mask_b,
        label_encoder,
        original_n_samples=original_n_samples,
    )


def build_ave_pair():
    validate_max_sample()
    max_sample = get_max_sample()
    x_a, y_a_raw, train_mask_a = load_train_test_domain(
        DATASET_CONFIG["domain_a_train_path"],
        DATASET_CONFIG["domain_a_test_path"],
        split_xy_pickled_array,
    )
    x_b, y_b_raw, train_mask_b = load_train_test_domain(
        DATASET_CONFIG["domain_b_train_path"],
        DATASET_CONFIG["domain_b_test_path"],
        split_xy_pickled_array,
    )

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"Expected paired AVE domains with equal rows, got {x_a.shape[0]} and {x_b.shape[0]}.")
    if not np.array_equal(train_mask_a, train_mask_b):
        raise ValueError("AVE train/test masks must be identical across audio and video domains.")
    if not np.array_equal(y_a_raw, y_b_raw):
        raise ValueError("Expected AVE audio/video labels to match in the same row order.")

    label_encoder = LabelEncoder().fit(np.concatenate([y_a_raw, y_b_raw]))
    original_n_samples = len(y_a_raw)

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_a, max_sample)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)

    return make_pair_dict(
        x_a,
        y_a_raw,
        train_mask_a,
        x_b,
        y_b_raw,
        train_mask_b,
        label_encoder,
        original_n_samples=original_n_samples,
    )


def build_rgbd_pair():
    validate_max_sample()
    max_sample = get_max_sample()
    labels_a = load_array_labels(DATASET_CONFIG["domain_a_path"])
    labels_b = load_array_labels(DATASET_CONFIG["domain_b_path"])
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError(
            f"Expected paired RGBD domains with equal rows, got {labels_a.shape[0]} and {labels_b.shape[0]}."
        )
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Expected RGBD modalities to have identical labels in the same row order.")

    label_encoder = LabelEncoder().fit(np.concatenate([labels_a, labels_b]))
    display_classes = load_rgbd_label_names(label_encoder.classes_)
    train_mask = make_stratified_train_mask(labels_a, RGBD_TRAIN_FRACTION)

    x_a, y_a_raw, train_mask_a = load_rgbd_domain(DATASET_CONFIG["domain_a_path"], train_mask)
    x_b, y_b_raw, train_mask_b = load_rgbd_domain(DATASET_CONFIG["domain_b_path"], train_mask)

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_a, max_sample)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)
    if not np.array_equal(train_mask_a, train_mask_b):
        raise ValueError("RGBD label masking must be identical in both modalities.")

    x_a = preprocess_rgbd_domain(x_a)
    x_b = preprocess_rgbd_domain(x_b)

    return make_pair_dict(
        x_a,
        y_a_raw,
        train_mask_a,
        x_b,
        y_b_raw,
        train_mask_b,
        label_encoder,
        display_classes=display_classes,
        original_n_samples=len(labels_a),
    )


def build_sketchy_pair(seed):
    validate_max_sample()
    max_sample = get_max_sample()
    x_a, y_a_raw, object_ids_a = split_xy_object_id_array(DATASET_CONFIG["domain_a_path"])
    x_b_full, y_b_raw_full, object_ids_b_full = split_xy_object_id_array(DATASET_CONFIG["domain_b_path"])
    original_n_samples = len(y_a_raw)
    source_size = x_a.shape[0]
    target_size = x_b_full.shape[0]

    x_b, y_b_raw, object_ids_b, target_selected_indices = select_one_target_per_source_object(
        x_b_full,
        y_b_raw_full,
        object_ids_b_full,
        object_ids_a,
        seed + 11,
    )

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"Expected paired Sketchy domains with equal rows, got {x_a.shape[0]} and {x_b.shape[0]}.")

    label_encoder = LabelEncoder().fit(np.concatenate([y_a_raw, y_b_raw]))
    display_classes = load_sketchy_label_names(label_encoder.classes_)
    train_mask_a = np.ones(len(y_a_raw), dtype=bool)
    train_mask_b = ~make_source_object_mask(
        y_a_raw,
        object_ids_a,
        object_ids_b,
        LABEL_MASKING_SKETCH_TARGET,
        seed + 17,
    )

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_b, max_sample)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)
    object_ids_a = np.asarray(object_ids_a).astype(str)[subsample_idx]
    object_ids_b = np.asarray(object_ids_b).astype(str)[subsample_idx]
    target_selected_indices = target_selected_indices[subsample_idx]

    if not np.array_equal(object_ids_a, object_ids_b):
        raise ValueError("Sketchy source and selected target object IDs are not aligned after subsampling.")

    return make_pair_dict(
        x_a,
        y_a_raw,
        train_mask_a,
        x_b,
        y_b_raw,
        train_mask_b,
        label_encoder,
        display_classes=display_classes,
        original_n_samples=original_n_samples,
        source_size=source_size,
        target_size=target_size,
        extra_metadata={
            "object_ids_a": object_ids_a,
            "object_ids_b": object_ids_b,
            "target_selected_indices": target_selected_indices,
        },
    )


def build_pair(seed=None):
    if DATASET_CONFIG["kind"] == "har_pickle_train_test":
        return build_har_pair()
    if DATASET_CONFIG["kind"] == "npy_train_test":
        return build_ave_pair()
    if DATASET_CONFIG["kind"] == "npy_single_file":
        return build_rgbd_pair()
    if DATASET_CONFIG["kind"] == "sketchy_npy_object_id":
        if seed is None:
            raise ValueError("Sketchy pair construction requires a seed.")
        return build_sketchy_pair(seed)
    raise ValueError(f"Unknown dataset kind: {DATASET_CONFIG['kind']}")


def save_pair_metadata(output_dir, pair):
    metadata = {
        "labels_a": pair["labels_a_true"],
        "labels_b": pair["labels_b_true"],
        "labels_a_obs": pair["labels_a_model"],
        "labels_b_obs": pair["labels_b_model"],
        "train_mask_a": pair["train_mask_a"],
        "train_mask_b": pair["train_mask_b"],
        "classes": pair["classes"],
        "display_classes": pair["display_classes"],
        "source_size": pair["source_size"],
        "target_size": pair["target_size"],
        "source_n_features": pair["source_n_features"],
        "target_n_features": pair["target_n_features"],
    }
    for key in ("object_ids_a", "object_ids_b", "target_selected_indices"):
        if key in pair:
            metadata[key] = pair[key]
    np.savez_compressed(output_dir / f"{DATASET}_labels.npz", **metadata)


# =============================================================================
# METHODS
# =============================================================================
def prepare_method_fit(method_name, pair, seed):
    """Construct one method and its fit arguments before profiling begins."""
    if method_name == "Unintegrated":
        x = np.vstack([pair["x_a"], pair["x_b"]])
        model = PCA(n_components=N_COMPONENTS, random_state=seed)
        return method_name, model, (x,)

    if method_name == "Unintegrated_PHATE":
        x = np.vstack([pair["x_a"], pair["x_b"]])
        model = phate.PHATE(n_components=N_COMPONENTS, random_state=seed)
        return method_name, model, (x,)

    fit_args = (
        pair["x_a"],
        pair["x_b"],
        pair["labels_a_model"],
        pair["labels_b_model"],
    )
    if method_name.startswith("FoSTA"):
        model = FoSTA(
            n_components=N_COMPONENTS,
            random_state=seed,
            **FOSTA_CONFIGS[method_name],
        )
    elif method_name == "Pamona":
        model = Pamona(
            n_components=N_COMPONENTS,
            random_state=seed,
            **PAMONA_CONFIG,
        )
    else:
        model = SUPERVISED_CLASSES[method_name](
            n_components=N_COMPONENTS,
            random_state=seed,
        )
    return method_name, model, fit_args


# =============================================================================
# METRICS AND OUTPUT
# =============================================================================
def label_transfer_metric_name(top_k):
    return f"label_transfer_top{top_k}"


def result_column_order():
    return (
        [
            "dataset",
            "method",
            "n_original_full_samples",
            "source_size",
            "target_size",
            "source_n_features",
            "target_n_features",
            "n_train_samples",
            "n_test_samples",
            "n_unique_classes",
        ]
        + [label_transfer_metric_name(top_k) for top_k in LABEL_TRANSFER_TOP_KS]
        + ["alignment_score", "FOSCTTM", "seed", "runtime_sec", "peak_mem_mb", "status"]
    )


def pair_result_metadata(pair):
    train_mask_key = "train_mask_b" if DATASET_CONFIG["kind"] == "sketchy_npy_object_id" else "train_mask_a"
    train_mask = np.asarray(pair[train_mask_key], dtype=bool)
    labels = np.concatenate([pair["labels_a_true"], pair["labels_b_true"]])
    return {
        "n_original_full_samples": int(pair["original_n_samples"]),
        "source_size": int(pair["source_size"]),
        "target_size": int(pair["target_size"]),
        "source_n_features": int(pair["source_n_features"]),
        "target_n_features": int(pair["target_n_features"]),
        "n_train_samples": int(np.count_nonzero(train_mask)),
        "n_test_samples": int(train_mask.size - np.count_nonzero(train_mask)),
        "n_unique_classes": int(np.unique(labels).size),
    }


def order_result_row(row):
    ordered_cols = result_column_order()
    return {
        col: row[col]
        for col in ordered_cols
        if col in row
    } | {
        col: value
        for col, value in row.items()
        if col not in ordered_cols
    }


def make_plot_specs(pair):
    n_a = pair["x_a"].shape[0]
    modalities = np.array(
        [DATASET_CONFIG["domain_a_name"]] * n_a
        + [DATASET_CONFIG["domain_b_name"]] * pair["x_b"].shape[0]
    )
    label_ids = np.concatenate([pair["labels_a_true"], pair["labels_b_true"]]).astype(int)
    labels = pair["display_classes"][label_ids].astype(str)

    return [
        ("modality", modalities, "tab10", "Modality"),
        ("labels", labels, "colorblind", "Ground Truth Label"),
    ]




def _directional_topk_label_transfer(train_x, train_y, test_x, test_y, top_ks=LABEL_TRANSFER_TOP_KS):
    transfer = test_transfer_accuracy(
        data1=test_x,
        data2=train_x,
        type1=test_y,
        type2=train_y,
        return_classwise_probabilities=True,
    )

    scores = {}
    probabilities = transfer["classwise_probabilities"]
    classes = transfer["classes"]
    n_classes = len(classes)

    for top_k in top_ks:
        if top_k == 1:
            scores[top_k] = float(transfer["accuracy"])
            continue

        effective_k = min(top_k, n_classes)
        if effective_k == n_classes:
            top_classes = np.broadcast_to(classes, (probabilities.shape[0], n_classes))
        else:
            kth = n_classes - effective_k
            top_class_idx = np.argpartition(probabilities, kth, axis=1)[:, kth:]
            top_classes = classes[top_class_idx]
        scores[top_k] = float(np.mean(np.any(top_classes == test_y[:, None], axis=1)))

    return scores


def bidirectional_label_transfer_topk(emb_a, emb_b, pair, top_ks=LABEL_TRANSFER_TOP_KS):
    test_mask_a = ~pair["train_mask_a"]
    test_mask_b = ~pair["train_mask_b"]
    directional_scores = []

    # Domain A labeled train points predict masked test points in domain B.
    if np.any(test_mask_b):
        directional_scores.append(_directional_topk_label_transfer(
            train_x=emb_a[pair["train_mask_a"]],
            train_y=pair["labels_a_true"][pair["train_mask_a"]],
            test_x=emb_b[test_mask_b],
            test_y=pair["labels_b_true"][test_mask_b],
            top_ks=top_ks,
        ))

    # Domain B labeled train points predict masked test points in domain A.
    if np.any(test_mask_a):
        directional_scores.append(_directional_topk_label_transfer(
            train_x=emb_b[pair["train_mask_b"]],
            train_y=pair["labels_b_true"][pair["train_mask_b"]],
            test_x=emb_a[test_mask_a],
            test_y=pair["labels_a_true"][test_mask_a],
            top_ks=top_ks,
        ))

    if not directional_scores:
        return {top_k: np.nan for top_k in top_ks}

    return {
        top_k: float(np.mean([scores[top_k] for scores in directional_scores]))
        for top_k in top_ks
    }


def benchmark_method(method_name, embedding, pair, output_dir, seed):
    emb = coerce_embedding_array(embedding, n_components=N_COMPONENTS)
    n_a = pair["x_a"].shape[0]
    n_b = pair["x_b"].shape[0]
    if emb.shape[0] != n_a + n_b:
        raise ValueError(f"Embedding has {emb.shape[0]} rows; expected {n_a + n_b}.")

    emb_a = emb[:n_a]
    emb_b = emb[n_a:]
    method_dir = output_dir / method_name
    save_embeddings(method_dir, method_name, emb, n_a, names=("domain_a", "domain_b"))
    save_embedding_plots(method_dir, method_name, emb, make_plot_specs(pair), ext="pdf")

    label_transfer_scores = bidirectional_label_transfer_topk(emb_a, emb_b, pair)

    row = {
        "method": method_name,
        "alignment_score": float(test_alignment_score(emb_a, emb_b, random_state=seed + 101)),
        "FOSCTTM": float(np.nanmean(calc_domainAveraged_FOSCTTM(emb_a, emb_b))),
    }
    row.update({
        label_transfer_metric_name(top_k): label_transfer_scores[top_k]
        for top_k in LABEL_TRANSFER_TOP_KS
    })
    return row


# =============================================================================
# MAIN
# =============================================================================
def main():
    validate_datasets()
    validate_max_samples()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = PROJECT_ROOT / "results_multimodal" / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    results_csv = root_dir / "results_multimodal.csv"

    all_rows = []
    for dataset in DATASETS:
        set_active_dataset(dataset)
        validate_config()
        is_sketchy = DATASET_CONFIG["kind"] == "sketchy_npy_object_id"
        pair = None if is_sketchy else build_pair()
        pair_metadata = None if is_sketchy else pair_result_metadata(pair)

        dataset_dir = root_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_experiment_metadata(dataset_dir, timestamp)
        if not is_sketchy:
            save_pair_metadata(dataset_dir, pair)

        # Track methods that have failed for this dataset; skip remaining seeds for them.
        failed_methods = set()

        for seed in SEEDS:
            print(f"\n### STARTING DATASET: {dataset} | SEED: {seed} ###")
            seed_dir = dataset_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if is_sketchy:
                pair = build_pair(seed)
                pair_metadata = pair_result_metadata(pair)
                save_pair_metadata(seed_dir, pair)

            for method_name in MODELS_TO_RUN:
                if method_name in failed_methods:
                    out_name = next(iter(FOSTA_CONFIGS)) if method_name == "FoSTA" else method_name
                    print(f"Skipping {method_name} for dataset {dataset} (previous failure)")
                    row = {
                        "dataset": dataset,
                        "method": out_name,
                        **pair_metadata,
                        "alignment_score": np.nan,
                        "FOSCTTM": np.nan,
                        "seed": seed,
                        "runtime_sec": np.nan,
                        "peak_mem_mb": np.nan,
                        "status": "skipped: previous failure",
                    }
                    row.update({
                        label_transfer_metric_name(top_k): np.nan
                        for top_k in LABEL_TRANSFER_TOP_KS
                    })
                    all_rows.append(row)
                    append_result_row(results_csv, order_result_row(row))
                    continue
                out_name = next(iter(FOSTA_CONFIGS)) if method_name == "FoSTA" else method_name
                runtime_sec = np.nan
                try:
                    print(f"Running {method_name}...")
                    seed_everything(seed)
                    peak_mem_mb = np.nan
                    (
                        (out_name, embedding),
                        runtime_sec,
                        peak_mem_mb,
                    ) = profile_fit_transform(
                        prepare_method_fit,
                        (method_name, pair, seed),
                        timeout_sec=MAX_FIT_TRANSFORM_SEC,
                    )
                    row = benchmark_method(out_name, embedding, pair, seed_dir, seed)
                    row.update(
                        dataset=dataset,
                        **pair_metadata,
                        seed=seed,
                        runtime_sec=runtime_sec,
                        peak_mem_mb=peak_mem_mb,
                        status="ok",
                    )
                except Exception as exc:
                    runtime_sec = getattr(exc, "runtime_sec", runtime_sec)
                    peak_mem_mb = getattr(exc, "peak_mem_mb", peak_mem_mb)
                    if isinstance(exc, TimeoutError) or "fit_transform timeout" in str(exc):
                        status_text = "Crash: timeout"
                    else:
                        status_text = f"error: {exc}"
                    # mark this method as failed for this dataset so we skip remaining seeds
                    failed_methods.add(method_name)
                    row = {
                        "dataset": dataset,
                        "method": out_name,
                        **pair_metadata,
                        "alignment_score": np.nan,
                        "FOSCTTM": np.nan,
                        "seed": seed,
                        "runtime_sec": runtime_sec,
                        "peak_mem_mb": peak_mem_mb if 'peak_mem_mb' in locals() else np.nan,
                        "status": status_text,
                    }
                    row.update({
                        label_transfer_metric_name(top_k): np.nan
                        for top_k in LABEL_TRANSFER_TOP_KS
                    })

                all_rows.append(row)
                append_result_row(results_csv, order_result_row(row))
                if row["status"] == "ok":
                    topk_text = " | ".join(
                        f"Top{top_k}={row[label_transfer_metric_name(top_k)]:.4f}"
                        for top_k in LABEL_TRANSFER_TOP_KS
                    )
                    print(
                        f"  {topk_text} | "
                        f"AS={row['alignment_score']:.4f} | "
                        f"FOSCTTM={row['FOSCTTM']:.4f} | "
                        f"{row['runtime_sec']:.1f}s"
                    )
                else:
                    print(f"  FAILED: {row['status']}")

    results_df = pd.DataFrame(all_rows).sort_values(["dataset", "seed", "method"], kind="stable")
    ordered_cols = result_column_order()
    results_df = results_df[
        [col for col in ordered_cols if col in results_df.columns]
        + [col for col in results_df.columns if col not in ordered_cols]
    ]
    results_df.to_csv(results_csv, index=False)
    print(f"\nFinished. Results saved to: {root_dir}")
    print(f"Metrics CSV: {results_csv}")


if __name__ == "__main__":
    main()
