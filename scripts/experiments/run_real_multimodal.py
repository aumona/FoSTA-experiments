"""
HAR/AVE/RGBD/Sketchy multimodal alignment benchmark.

Each data table stores labels in the first column and modality features in the
remaining columns (Sketchy stores labels/object IDs in the last two columns).
Existing splits are concatenated and ignored; a seeded test split is created
using TEST_PERC for every dataset. Training pairs share label masks across domains. Label transfer
scores average both directions on test pairs; alignment metrics use all rows.
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
    "har", 
    "sketchy_resnet18",
    "sketchy_dinov2base",
    "ave", 
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

SKETCHY_DATA_ROOT = PROJECT_ROOT / "data_sketchy"
SKETCHY_RESNET18_PHOTO_PATH = SKETCHY_DATA_ROOT / "photo_resnet18_embeddings.npy"
SKETCHY_RESNET18_SKETCH_PATH = SKETCHY_DATA_ROOT / "sketch_resnet18_embeddings.npy"
SKETCHY_DINOV2BASE_PHOTO_PATH = SKETCHY_DATA_ROOT / "photo_dinov2base_embeddings.npy"
SKETCHY_DINOV2BASE_SKETCH_PATH = SKETCHY_DATA_ROOT / "sketch_dinov2base_embeddings.npy"
SKETCHY_LABEL_DICT_PATH = SKETCHY_DATA_ROOT / "label_dic"

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
}  # Set a dataset value to None to keep every observation.
SUBSAMPLE_SEED = 2026  # Fixed selection, independent of experimental seeds.
N_COMPONENTS = 2
N_JOBS = -1
# Max seconds to allow a model `fit_transform` to run. Set to None to disable timeout.
MAX_FIT_TRANSFORM_SEC = None  # in seconds, set to None to disable
LABEL_TRANSFER_TOP_KS = (1, 5, 10)
TEST_PERC = 0.2  # Shared held-out pair fraction for label transfer in every dataset.
# Fractions of labels masked within the 80% supervision/training pool, not
# the full dataset (pool size is 1 - TEST_PERC). The 20% test pairs are fixed
# within each seed and always unlabeled. Training masks are shared across
# paired modalities and nested across masking levels.
LABEL_MASK_PERC = [0]
# LABEL_MASK_PERC = [0.2,0.4,0.6,0.8]


MODELS_TO_RUN = [
    "Unintegrated",
    # "Unintegrated_PHATE",
    "FoSTA_t2",
    "FoSTA_tauto",
    "KEMAlin",
    "KEMArbf",
    "MALI_t2",
    "MALI_tauto",
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

MALI_CONFIGS = {
    "MALI_t2": {
        "t": 2,
        "n_jobs": N_JOBS,
    },
    "MALI_tauto": {
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
        "label_mask_perc": LABEL_MASK_PERC,
        "predefined_splits": "concatenated; original split membership ignored",
        "test_perc": TEST_PERC,
        "test_split": "seeded stratified shared split for every dataset",
        "masked_rows": "test labels always hidden; training mask shared by matching rows in both domains",
        "mask_count": "floor(p * number of training pairs), with the same mask shared across domains",
        "label_transfer": "equal average of A-labeled-training to B-test and B-labeled-training to A-test",
        "normalization": "StandardScaler per complete modality" if DATASET == "har" else "none",
        "masked_label_value": -1,
        "seeds": SEEDS,
        "max_sample": get_max_sample(),
        "max_sample_by_dataset": MAX_SAMPLE_BY_DATASET,
        "subsample_seed": SUBSAMPLE_SEED,
        "subsampling": "seeded-random within label strata; fixed across experimental seeds",
        "training_masks": "prefixes of one seeded stratified ordering, nested across masking levels",
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
        "mali_configs": MALI_CONFIGS,
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
    elif DATASET_CONFIG["kind"] == "npy_single_file":
        metadata["domain_a"]["path"] = str(DATASET_CONFIG["domain_a_path"].relative_to(PROJECT_ROOT))
        metadata["domain_b"]["path"] = str(DATASET_CONFIG["domain_b_path"].relative_to(PROJECT_ROOT))
        metadata["rgbd_embedding_name"] = DATASET_CONFIG["embedding_name"]
        metadata["rgbd_label_map_path"] = str(RGBD_LABEL_MAP_PATH.relative_to(PROJECT_ROOT))
    elif DATASET_CONFIG["kind"] == "sketchy_npy_object_id":
        metadata["domain_a"]["path"] = str(DATASET_CONFIG["domain_a_path"].relative_to(PROJECT_ROOT))
        metadata["domain_b"]["path"] = str(DATASET_CONFIG["domain_b_path"].relative_to(PROJECT_ROOT))
        metadata["sketchy_embedding_name"] = DATASET_CONFIG["embedding_name"]
        metadata["sketchy_label_dict_path"] = str(SKETCHY_LABEL_DICT_PATH.relative_to(PROJECT_ROOT))
        metadata["label_column"] = "-2"
        metadata["feature_columns"] = ":-2"
        metadata["object_id_column"] = "-1"
        metadata["sketchy_target_sampling"] = "one random target example per source object ID, ordered to match source"
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


def apply_label_masking(base_pair, proportion, seed):
    """Reserve shared test pairs, then mask labels only within training pairs."""
    labels = base_pair["labels_a_true"]
    if not np.array_equal(labels, base_pair["labels_b_true"]):
        raise ValueError("Joint masking requires matching labels in paired row order.")
    validate_label_mask_perc([TEST_PERC])
    if not 0 < TEST_PERC < 1:
        raise ValueError("TEST_PERC must be strictly between 0 and 1.")
    test_mask = ~make_label_visibility_mask(labels, TEST_PERC, seed + 7)
    if test_mask.shape != labels.shape or not test_mask.any() or test_mask.all():
        raise ValueError("The loaded pair must contain both training and held-out test rows.")
    training_pool = ~test_mask
    visible = np.zeros(len(labels), dtype=bool)
    visible[training_pool] = make_label_visibility_mask(
        labels[training_pool], proportion, seed + 17, nested=True
    )
    pair = base_pair.copy()
    pair["test_mask"] = test_mask
    for domain in ("a", "b"):
        observed = pair[f"labels_{domain}_true"].copy()
        observed[~visible] = -1
        pair[f"train_mask_{domain}"] = visible.copy()
        pair[f"labels_{domain}_model"] = observed
    pair["label_mask_perc"] = float(proportion)
    return pair


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


def load_combined_domain(domain, split_func):
    """Concatenate stored splits, discarding their original membership."""
    parts = [
        split_func(DATASET_CONFIG[f"domain_{domain}_{split}_path"])
        for split in ("train", "test")
    ]
    return np.vstack([x for x, _ in parts]), np.concatenate([y for _, y in parts])


def split_xy_frame_path(path):
    return split_xy(load_frame(path))


def load_dataset_pair(seed=0):
    """Load aligned rows, standardize HAR only, then optionally subsample."""
    validate_max_sample()
    kind = DATASET_CONFIG["kind"]
    extra = {}
    if kind in {"har_pickle_train_test", "npy_train_test"}:
        loader = split_xy_frame_path if kind == "har_pickle_train_test" else split_xy_pickled_array
        x_a, y_a = load_combined_domain("a", loader)
        x_b, y_b = load_combined_domain("b", loader)
        if kind == "har_pickle_train_test":
            # Fit each scaler on the entire concatenated modality, before subsampling.
            x_a = StandardScaler().fit_transform(x_a)
            x_b = StandardScaler().fit_transform(x_b)
    elif kind == "npy_single_file":
        x_a, y_a = split_xy_array(DATASET_CONFIG["domain_a_path"])
        x_b, y_b = split_xy_array(DATASET_CONFIG["domain_b_path"])
    elif kind == "sketchy_npy_object_id":
        x_a, y_a, ids_a = split_xy_object_id_array(DATASET_CONFIG["domain_a_path"])
        x_b_full, y_b_full, ids_b_full = split_xy_object_id_array(DATASET_CONFIG["domain_b_path"])
        x_b, y_b, ids_b, selected = select_one_target_per_source_object(
            x_b_full, y_b_full, ids_b_full, ids_a, seed + 11
        )
        extra = {"object_ids_a": ids_a, "object_ids_b": ids_b, "target_selected_indices": selected}
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    if x_a.shape[0] != x_b.shape[0] or not np.array_equal(y_a, y_b):
        raise ValueError("Paired domains must have matching labels in the same row order.")
    if not len(y_a):
        raise ValueError("Cannot benchmark an empty dataset.")
    original_n_samples = len(y_a)
    encoder = LabelEncoder().fit(np.concatenate([y_a, y_b]))
    display_classes = None
    if kind == "npy_single_file":
        display_classes = load_rgbd_label_names(encoder.classes_)
    elif kind == "sketchy_npy_object_id":
        display_classes = load_sketchy_label_names(encoder.classes_)

    # Subsample once, independently of original splits and masking proportions.
    indices = make_stratified_subsample_indices(
        y_a, np.ones(len(y_a), dtype=bool), get_max_sample(), seed=SUBSAMPLE_SEED
    )
    visible = np.ones(len(indices), dtype=bool)
    return make_pair_dict(
        x_a[indices], y_a[indices], visible.copy(),
        x_b[indices], y_b[indices], visible.copy(), encoder,
        display_classes=display_classes,
        original_n_samples=original_n_samples,
        extra_metadata={key: np.asarray(value)[indices] for key, value in extra.items()},
    )


def build_pair(seed=0, label_mask_perc=None):
    """Convenience entry point for the scaling runner and other callers."""
    if label_mask_perc is None:
        validate_label_mask_perc(LABEL_MASK_PERC)
        label_mask_perc = LABEL_MASK_PERC[0]
    return apply_label_masking(load_dataset_pair(seed), label_mask_perc, seed)


def save_pair_metadata(output_dir, pair):
    metadata = {
        "label_mask_perc": pair["label_mask_perc"],
        "labels_a": pair["labels_a_true"],
        "labels_b": pair["labels_b_true"],
        "labels_a_obs": pair["labels_a_model"],
        "labels_b_obs": pair["labels_b_model"],
        "train_mask_a": pair["train_mask_a"],
        "train_mask_b": pair["train_mask_b"],
        "test_mask": pair["test_mask"],
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
    elif method_name in MALI_CONFIGS:
        model = MALI(
            n_components=N_COMPONENTS,
            random_state=seed,
            **MALI_CONFIGS[method_name],
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
            "label_mask_perc",
            "n_original_full_samples",
            "source_size",
            "target_size",
            "source_n_features",
            "target_n_features",
            "n_train_samples",
            "n_test_samples",
            "n_train_samples_b",
            "n_test_samples_b",
            "n_labeled_train_samples",
            "n_masked_train_samples",
            "n_unique_classes",
        ]
        + [label_transfer_metric_name(top_k) for top_k in LABEL_TRANSFER_TOP_KS]
        + ["alignment_score", "FOSCTTM", "seed", "runtime_sec", "peak_mem_mb", "status"]
    )


def pair_result_metadata(pair):
    train_mask = np.asarray(pair["train_mask_a"], dtype=bool)
    test_mask = np.asarray(pair["test_mask"], dtype=bool)
    n_test = int(test_mask.sum())
    n_train = int((~test_mask).sum())
    labels = np.concatenate([pair["labels_a_true"], pair["labels_b_true"]])
    return {
        "label_mask_perc": pair["label_mask_perc"],
        "n_original_full_samples": int(pair["original_n_samples"]),
        "source_size": int(pair["source_size"]),
        "target_size": int(pair["target_size"]),
        "source_n_features": int(pair["source_n_features"]),
        "target_n_features": int(pair["target_n_features"]),
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "n_train_samples_b": n_train,
        "n_test_samples_b": n_test,
        "n_labeled_train_samples": int(train_mask.sum()),
        "n_masked_train_samples": n_train - int(train_mask.sum()),
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
    directional_scores = []
    for source, target, train_x, test_x in (("a", "b", emb_a, emb_b), ("b", "a", emb_b, emb_a)):
        train_mask = pair[f"train_mask_{source}"]
        test_mask = pair["test_mask"]
        # Never substitute a one-direction score for the bidirectional average.
        if not np.any(train_mask) or not np.any(test_mask):
            return {top_k: np.nan for top_k in top_ks}
        directional_scores.append(_directional_topk_label_transfer(
            train_x=train_x[train_mask],
            train_y=pair[f"labels_{source}_true"][train_mask],
            test_x=test_x[test_mask],
            test_y=pair[f"labels_{target}_true"][test_mask],
            top_ks=top_ks,
        ))
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
def run_method(method_name, pair, output_dir, seed):
    """Profile and score one method; retain a complete row on failure."""
    row = {
        "dataset": DATASET,
        "method": method_name,
        **pair_result_metadata(pair),
        "seed": seed,
        "runtime_sec": np.nan,
        "peak_mem_mb": np.nan,
        "alignment_score": np.nan,
        "FOSCTTM": np.nan,
        **{label_transfer_metric_name(k): np.nan for k in LABEL_TRANSFER_TOP_KS},
    }
    try:
        seed_everything(seed)
        (out_name, embedding), runtime, memory = profile_fit_transform(
            prepare_method_fit, (method_name, pair, seed),
            timeout_sec=MAX_FIT_TRANSFORM_SEC,
        )
        row.update(runtime_sec=runtime, peak_mem_mb=memory)
        row.update(benchmark_method(out_name, embedding, pair, output_dir, seed))
        row["status"] = "ok"
    except Exception as exc:
        row["runtime_sec"] = getattr(exc, "runtime_sec", row["runtime_sec"])
        row["peak_mem_mb"] = getattr(exc, "peak_mem_mb", row["peak_mem_mb"])
        row["status"] = "Crash: timeout" if isinstance(exc, TimeoutError) else f"error: {exc}"
    return order_result_row(row)


def main():
    validate_datasets()
    validate_max_samples()
    validate_label_mask_perc(LABEL_MASK_PERC)
    if not DATASETS or not SEEDS or not MODELS_TO_RUN:
        raise ValueError("DATASETS, SEEDS, and MODELS_TO_RUN must not be empty.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = PROJECT_ROOT / "results_multimodal" / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    results_csv = root_dir / "results_multimodal.csv"
    all_rows = []
    for dataset in DATASETS:
        set_active_dataset(dataset)
        validate_config()
        is_sketchy = DATASET_CONFIG["kind"] == "sketchy_npy_object_id"
        base_pair = None if is_sketchy else load_dataset_pair()
        dataset_dir = root_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_experiment_metadata(dataset_dir, timestamp)
        for seed in SEEDS:
            if is_sketchy:
                base_pair = load_dataset_pair(seed)
            for proportion in LABEL_MASK_PERC:
                pair = apply_label_masking(base_pair, proportion, seed)
                run_dir = dataset_dir / f"mask_{float(proportion)}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                save_pair_metadata(run_dir, pair)
                print(f"\n### {dataset} | mask={proportion} | seed={seed} ###")
                for method_name in MODELS_TO_RUN:
                    print(f"Running {method_name}...")
                    row = run_method(method_name, pair, run_dir, seed)
                    all_rows.append(row)
                    append_result_row(results_csv, row)
                    print(f"  {row['status']} | Top1={row['label_transfer_top1']:.4f} "
                          f"| {row['runtime_sec']:.1f}s")

    results_df = pd.DataFrame(all_rows).sort_values(
        ["dataset", "label_mask_perc", "seed", "method"], kind="stable"
    )
    results_df.to_csv(results_csv, index=False)
    print(f"\nFinished. Metrics CSV: {results_csv}")


if __name__ == "__main__":
    main()
