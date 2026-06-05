"""
HAR/RGBD multimodal alignment benchmark.

Each data table stores labels in the first column and modality features in the
remaining columns. Train rows stay labeled; test rows are masked as -1.
"""
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rf-mali-matplotlib")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from ad_experiment_utils import (
    append_result_row,
    coerce_embedding_array,
    save_embedding_plots,
    save_embeddings,
    seed_everything,
    write_json,
)


# =============================================================================
# CONFIG
# =============================================================================
DATASET = "rgbd"  # "har" or "rgbd"

HAR_DATA_ROOT = PROJECT_ROOT / "data_har"
HAR_ACC_TRAIN_PATH = HAR_DATA_ROOT / "har_acc_total_train.pkl"
HAR_ACC_TEST_PATH = HAR_DATA_ROOT / "har_acc_total_test.pkl"
HAR_GYRO_TRAIN_PATH = HAR_DATA_ROOT / "har_gyro_train.pkl"
HAR_GYRO_TEST_PATH = HAR_DATA_ROOT / "har_gyro_test.pkl"

RGBD_DATA_ROOT = PROJECT_ROOT / "data_rgbd"
RGBD_PHOTO_PATH = RGBD_DATA_ROOT / "rgbd_photo_resnet18_embeddings.npy"
RGBD_DEPTH_PATH = RGBD_DATA_ROOT / "rgbd_depth_resnet18_embeddings.npy"
RGBD_LABEL_MAP_PATH = RGBD_DATA_ROOT / "rgbd_label_map.json"
RGBD_TRAIN_FRACTION = 0.70

SEEDS = [39041, 56089, 79121]
MAX_SAMPLE = 2000  # Set to an int for deterministic stratified subsampling per domain.
N_COMPONENTS = 2
N_JOBS = -1

LABEL_TRANSFER_TOP_KS = (1, 5, 10)


MODELS_TO_RUN = [
    # "Unintegrated",
    # "Unintegrated_PHATE",
    # "FoSTA",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    # "Pamona",
]

FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "mu": 1,
        "t": 2,
        "n_estimators": 100,
        "n_jobs": N_JOBS,
    },
    # "FoSTA_tauto": {
    #     "t": 'auto',
    #     "n_estimators": 1000,
    #     "n_jobs": N_JOBS,
    # },
    # "FoSTA_t2_kerf": {
    #     "mu": 1,
    #     "kernel_method": "kerf",
    #     "t": 2,
    #     "n_estimators": 1000,
    # },
}

MALI_CONFIG = {
    "t": 2,
    "n_jobs": N_JOBS,
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
    "rgbd": {
        "domain_a_name": "rgbd_photo",
        "domain_b_name": "rgbd_depth",
        "kind": "npy_single_file",
        "domain_a_path": RGBD_PHOTO_PATH,
        "domain_b_path": RGBD_DEPTH_PATH,
        "results_dir": PROJECT_ROOT / "results_rgbd",
    },
}

if DATASET not in DATASET_CONFIGS:
    raise ValueError(f"DATASET must be one of {sorted(DATASET_CONFIGS)}, got {DATASET!r}.")

DATASET_CONFIG = DATASET_CONFIGS[DATASET]
RESULTS_ROOT = DATASET_CONFIG["results_dir"]
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA AND BOOKKEEPING
# =============================================================================
def validate_config():
    if DATASET_CONFIG["kind"] == "har_pickle_train_test":
        paths = [
            DATASET_CONFIG["domain_a_train_path"],
            DATASET_CONFIG["domain_a_test_path"],
            DATASET_CONFIG["domain_b_train_path"],
            DATASET_CONFIG["domain_b_test_path"],
        ]
    else:
        paths = [
            DATASET_CONFIG["domain_a_path"],
            DATASET_CONFIG["domain_b_path"],
            RGBD_LABEL_MAP_PATH,
        ]

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
        "max_sample": MAX_SAMPLE,
        "n_components": N_COMPONENTS,
        "n_jobs": N_JOBS,
        "models_to_run": MODELS_TO_RUN,
        "fosta_configs": FOSTA_CONFIGS,
        "mali_config": MALI_CONFIG,
    }
    if DATASET_CONFIG["kind"] == "har_pickle_train_test":
        metadata["domain_a"].update({
            "train_path": str(DATASET_CONFIG["domain_a_train_path"].relative_to(PROJECT_ROOT)),
            "test_path": str(DATASET_CONFIG["domain_a_test_path"].relative_to(PROJECT_ROOT)),
        })
        metadata["domain_b"].update({
            "train_path": str(DATASET_CONFIG["domain_b_train_path"].relative_to(PROJECT_ROOT)),
            "test_path": str(DATASET_CONFIG["domain_b_test_path"].relative_to(PROJECT_ROOT)),
        })
    else:
        metadata["domain_a"]["path"] = str(DATASET_CONFIG["domain_a_path"].relative_to(PROJECT_ROOT))
        metadata["domain_b"]["path"] = str(DATASET_CONFIG["domain_b_path"].relative_to(PROJECT_ROOT))
        metadata["rgbd_label_map_path"] = str(RGBD_LABEL_MAP_PATH.relative_to(PROJECT_ROOT))
        metadata["rgbd_train_fraction"] = RGBD_TRAIN_FRACTION
        metadata["rgbd_split"] = "deterministic stratified by label in original row order"
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


def validate_max_sample():
    if MAX_SAMPLE is None:
        return
    if not isinstance(MAX_SAMPLE, int) or isinstance(MAX_SAMPLE, bool) or MAX_SAMPLE <= 0:
        raise ValueError(f"MAX_SAMPLE must be None or a positive int, got {MAX_SAMPLE!r}.")


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


def make_pair_dict(
    x_a,
    y_a_raw,
    train_mask_a,
    x_b,
    y_b_raw,
    train_mask_b,
    label_encoder,
    display_classes=None,
):
    y_a_true = label_encoder.transform(y_a_raw)
    y_b_true = label_encoder.transform(y_b_raw)

    y_a_model = y_a_true.copy()
    y_b_model = y_b_true.copy()
    y_a_model[~train_mask_a] = -1
    y_b_model[~train_mask_b] = -1

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"Expected paired domains with equal rows, got {x_a.shape[0]} and {x_b.shape[0]}.")

    return {
        "x_a": x_a,
        "x_b": x_b,
        "labels_a_true": y_a_true.astype(int),
        "labels_b_true": y_b_true.astype(int),
        "labels_a_model": y_a_model.astype(int),
        "labels_b_model": y_b_model.astype(int),
        "train_mask_a": train_mask_a,
        "train_mask_b": train_mask_b,
        "classes": label_encoder.classes_,
        "display_classes": (
            np.asarray(display_classes).astype(str)
            if display_classes is not None
            else label_encoder.classes_.astype(str)
        ),
    }


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


def load_har_domain(train_path, test_path):
    x_train, y_train_raw = split_xy(load_frame(train_path))
    x_test, y_test_raw = split_xy(load_frame(test_path))

    x = np.vstack([x_train, x_test])
    y_raw = np.concatenate([y_train_raw, y_test_raw])

    train_mask = np.r_[np.ones(len(y_train_raw), dtype=bool), np.zeros(len(y_test_raw), dtype=bool)]
    return x, y_raw, train_mask


def load_rgbd_domain(path, train_mask):
    x, y_raw = split_xy_array(path)
    return x, y_raw, train_mask


def build_har_pair():
    validate_max_sample()
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

    x_a, y_a_raw, train_mask_a = load_har_domain(
        DATASET_CONFIG["domain_a_train_path"],
        DATASET_CONFIG["domain_a_test_path"],
    )
    x_b, y_b_raw, train_mask_b = load_har_domain(
        DATASET_CONFIG["domain_b_train_path"],
        DATASET_CONFIG["domain_b_test_path"],
    )

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_a, MAX_SAMPLE)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)

    # Scale each modality after train/test union and optional subsampling.
    x_a = standardize_domain(x_a)
    x_b = standardize_domain(x_b)

    return make_pair_dict(x_a, y_a_raw, train_mask_a, x_b, y_b_raw, train_mask_b, label_encoder)


def build_rgbd_pair():
    validate_max_sample()
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

    subsample_idx = make_stratified_subsample_indices(y_a_raw, train_mask_a, MAX_SAMPLE)
    x_a, y_a_raw, train_mask_a = apply_subsample(x_a, y_a_raw, train_mask_a, subsample_idx)
    x_b, y_b_raw, train_mask_b = apply_subsample(x_b, y_b_raw, train_mask_b, subsample_idx)
    if not np.array_equal(train_mask_a, train_mask_b):
        raise ValueError("RGBD label masking must be identical in both modalities.")

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
        display_classes=display_classes,
    )


def build_pair():
    if DATASET_CONFIG["kind"] == "har_pickle_train_test":
        return build_har_pair()
    if DATASET_CONFIG["kind"] == "npy_single_file":
        return build_rgbd_pair()
    raise ValueError(f"Unknown dataset kind: {DATASET_CONFIG['kind']}")


def save_pair_metadata(output_dir, pair):
    np.savez_compressed(
        output_dir / f"{DATASET}_labels.npz",
        labels_a=pair["labels_a_true"],
        labels_b=pair["labels_b_true"],
        labels_a_obs=pair["labels_a_model"],
        labels_b_obs=pair["labels_b_model"],
        train_mask_a=pair["train_mask_a"],
        train_mask_b=pair["train_mask_b"],
        classes=pair["classes"],
        display_classes=pair["display_classes"],
    )


# =============================================================================
# METHODS
# =============================================================================
def run_unintegrated_pca(pair, seed):
    x = np.vstack([pair["x_a"], pair["x_b"]])
    return PCA(n_components=N_COMPONENTS, random_state=seed).fit_transform(x)


def run_unintegrated_phate(pair, seed):
    x = np.vstack([pair["x_a"], pair["x_b"]])
    return phate.PHATE(n_components=N_COMPONENTS, random_state=seed).fit_transform(x)


def run_supervised_method(method_name, pair, seed):
    if method_name == "FoSTA":
        for out_name, params in FOSTA_CONFIGS.items():
            model = FoSTA(n_components=N_COMPONENTS, random_state=seed, **params)
            embedding = model.fit_transform(
                pair["x_a"], pair["x_b"], pair["labels_a_model"], pair["labels_b_model"]
            )
            return out_name, embedding

    if method_name == "MALI":
        model = MALI(n_components=N_COMPONENTS, random_state=seed, **MALI_CONFIG)
    else:
        model = SUPERVISED_CLASSES[method_name](n_components=N_COMPONENTS, random_state=seed)

    embedding = model.fit_transform(pair["x_a"], pair["x_b"], pair["labels_a_model"], pair["labels_b_model"])
    return method_name, embedding


def run_method(method_name, pair, seed):
    if method_name == "Unintegrated":
        return method_name, run_unintegrated_pca(pair, seed)
    if method_name == "Unintegrated_PHATE":
        return method_name, run_unintegrated_phate(pair, seed)
    return run_supervised_method(method_name, pair, seed)


# =============================================================================
# METRICS AND OUTPUT
# =============================================================================
def label_transfer_metric_name(top_k):
    return f"label_transfer_top{top_k}"


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
    validate_config()
    pair = build_pair()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = RESULTS_ROOT / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_metadata(root_dir, timestamp)
    save_pair_metadata(root_dir, pair)
    results_csv = root_dir / f"results_{DATASET}.csv"

    all_rows = []
    for seed in SEEDS:
        print(f"\n### STARTING SEED: {seed} ###")
        seed_dir = root_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for method_name in MODELS_TO_RUN:
            start = time.perf_counter()
            out_name = next(iter(FOSTA_CONFIGS)) if method_name == "FoSTA" else method_name
            try:
                print(f"Running {method_name}...")
                seed_everything(seed)
                out_name, embedding = run_method(method_name, pair, seed)
                row = benchmark_method(out_name, embedding, pair, seed_dir, seed)
                row.update(seed=seed, runtime_sec=float(time.perf_counter() - start), status="ok")
            except Exception as exc:
                row = {
                    "method": out_name,
                    "alignment_score": np.nan,
                    "FOSCTTM": np.nan,
                    "seed": seed,
                    "runtime_sec": float(time.perf_counter() - start),
                    "status": f"error: {exc}",
                }
                row.update({
                    label_transfer_metric_name(top_k): np.nan
                    for top_k in LABEL_TRANSFER_TOP_KS
                })

            all_rows.append(row)
            append_result_row(results_csv, row)
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

    pd.DataFrame(all_rows).sort_values(["seed", "method"], kind="stable").to_csv(results_csv, index=False)
    print(f"\nFinished. Results saved to: {root_dir}")
    print(f"Metrics CSV: {results_csv}")


if __name__ == "__main__":
    main()
