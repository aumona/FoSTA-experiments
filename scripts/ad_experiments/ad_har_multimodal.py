"""
HAR accelerometer-to-gyroscope multimodal alignment benchmark.

Each HAR pickle stores labels in the first column and modality features in the
remaining columns. Train rows stay labeled; test rows are masked as -1.
"""
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
    python_random_seed,
    save_embedding_plots,
    save_embeddings,
    seed_everything,
    write_json,
)


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data_har"
ACC_TRAIN_PATH = DATA_ROOT / "har_acc_total_train.pkl"
ACC_TEST_PATH = DATA_ROOT / "har_acc_total_test.pkl"
GYRO_TRAIN_PATH = DATA_ROOT / "har_gyro_train.pkl"
GYRO_TEST_PATH = DATA_ROOT / "har_gyro_test.pkl"

SEEDS = [39041, 56089, 79121]
N_COMPONENTS = 2
N_JOBS = -1

MODELS_TO_RUN = [
    "Unintegrated",
    "Unintegrated_PHATE",
    "FoSTA",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    "Pamona",
]

FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "mu": 3,
        "unlabeled_coupling": "predict_shared",
        "t": 2,
        "class_weight": "balanced_subsample",
        "n_estimators": 1000,
        "n_jobs": N_JOBS,
    },
    # "FoSTA_tauto": {
    #     "unlabeled_coupling": "predict_shared",
    #     "t": 'auto',
    #     "class_weight": "balanced_subsample",
    #     "n_estimators": 1000,
    #     "n_jobs": N_JOBS,
    # },
    # "FoSTA_t2_kerf": {
    #     "unlabeled_coupling": "predict_shared",
    #     "kernel_method": "kerf",
    #     "t": 2,
    #     "class_weight": "balanced_subsample",
    #     "n_estimators": 1000,
    # },
}

MALI_CONFIG = {
    # "t": 'auto',
    # "n_jobs": N_JOBS,
    # 'distances': 'none' 
}

SUPERVISED_CLASSES = {
    "FoSTA": FoSTA,
    "KEMAlin": KEMAlin,
    "KEMArbf": KEMArbf,
    "MALI": MALI,
    "Pamona": Pamona,
}

RESULTS_ROOT = PROJECT_ROOT / "results_har"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA AND BOOKKEEPING
# =============================================================================
def validate_config():
    for path in [ACC_TRAIN_PATH, ACC_TEST_PATH, GYRO_TRAIN_PATH, GYRO_TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")


def save_experiment_metadata(output_dir, timestamp):
    metadata = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp": timestamp,
        "domain_a": {
            "name": "har_acc",
            "train_path": str(ACC_TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "test_path": str(ACC_TEST_PATH.relative_to(PROJECT_ROOT)),
        },
        "domain_b": {
            "name": "har_gyro",
            "train_path": str(GYRO_TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "test_path": str(GYRO_TEST_PATH.relative_to(PROJECT_ROOT)),
        },
        "label_column": 0,
        "feature_columns": "1:",
        "masked_rows": "test",
        "masked_label_value": -1,
        "seeds": SEEDS,
        "n_components": N_COMPONENTS,
        "n_jobs": N_JOBS,
        "models_to_run": MODELS_TO_RUN,
        "fosta_configs": FOSTA_CONFIGS,
        "mali_config": MALI_CONFIG,
    }
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
        raise ValueError("HAR features contain non-finite values.")
    return features, labels


def load_domain(train_path, test_path, label_encoder, mask_test_labels=False):
    x_train, y_train_raw = split_xy(load_frame(train_path))
    x_test, y_test_raw = split_xy(load_frame(test_path))

    x = np.vstack([x_train, x_test])
    y_true = label_encoder.transform(np.concatenate([y_train_raw, y_test_raw]))

    # Scale each modality after train/test union so every input feature is centered.
    x = StandardScaler().fit_transform(x)

    train_mask = np.r_[np.ones(len(y_train_raw), dtype=bool), np.zeros(len(y_test_raw), dtype=bool)]
    y_model = y_true.copy()
    if mask_test_labels:
        y_model[~train_mask] = -1
    return x, y_true, y_model, train_mask


def build_pair():
    acc_train = load_frame(ACC_TRAIN_PATH)
    acc_test = load_frame(ACC_TEST_PATH)
    gyro_train = load_frame(GYRO_TRAIN_PATH)
    gyro_test = load_frame(GYRO_TEST_PATH)

    raw_labels = np.concatenate([
        acc_train.iloc[:, 0].astype(str).to_numpy(),
        acc_test.iloc[:, 0].astype(str).to_numpy(),
        gyro_train.iloc[:, 0].astype(str).to_numpy(),
        gyro_test.iloc[:, 0].astype(str).to_numpy(),
    ])
    label_encoder = LabelEncoder().fit(raw_labels)

    x_a, y_a_true, y_a_model, train_mask_a = load_domain(
        ACC_TRAIN_PATH,
        ACC_TEST_PATH,
        label_encoder,
        mask_test_labels=True,
    )
    x_b, y_b_true, y_b_model, train_mask_b = load_domain(
        GYRO_TRAIN_PATH,
        GYRO_TEST_PATH,
        label_encoder,
        mask_test_labels=True,
    )

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
    }


def save_pair_metadata(output_dir, pair):
    np.savez_compressed(
        output_dir / "har_labels.npz",
        labels_a=pair["labels_a_true"],
        labels_b=pair["labels_b_true"],
        labels_a_obs=pair["labels_a_model"],
        labels_b_obs=pair["labels_b_model"],
        train_mask_a=pair["train_mask_a"],
        train_mask_b=pair["train_mask_b"],
        classes=pair["classes"],
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
def make_plot_specs(pair):
    n_a = pair["x_a"].shape[0]
    modalities = np.array(["har_acc"] * n_a + ["har_gyro"] * pair["x_b"].shape[0])
    label_ids = np.concatenate([pair["labels_a_true"], pair["labels_b_true"]]).astype(int)
    labels = pair["classes"][label_ids].astype(str)

    return [
        ("modality", modalities, "tab10", "Modality"),
        ("labels", labels, "tab20", "Ground Truth Label"),
    ]


def bidirectional_label_transfer(emb_a, emb_b, pair):
    test_mask_a = ~pair["train_mask_a"]
    test_mask_b = ~pair["train_mask_b"]
    scores = []

    # Domain A labeled train points predict masked test points in domain B.
    if np.any(test_mask_b):
        scores.append(test_transfer_accuracy(
            data1=emb_b[test_mask_b],
            data2=emb_a[pair["train_mask_a"]],
            type1=pair["labels_b_true"][test_mask_b],
            type2=pair["labels_a_true"][pair["train_mask_a"]],
        ))

    # Domain B labeled train points predict masked test points in domain A.
    if np.any(test_mask_a):
        scores.append(test_transfer_accuracy(
            data1=emb_a[test_mask_a],
            data2=emb_b[pair["train_mask_b"]],
            type1=pair["labels_a_true"][test_mask_a],
            type2=pair["labels_b_true"][pair["train_mask_b"]],
        ))

    return float(np.mean(scores)) if scores else np.nan


def deterministic_alignment_score(emb_a, emb_b, seed):
    with python_random_seed(seed):
        return test_alignment_score(emb_a, emb_b)


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

    return {
        "method": method_name,
        "label_transfer": bidirectional_label_transfer(emb_a, emb_b, pair),
        "alignment_score": float(deterministic_alignment_score(emb_a, emb_b, seed + 101)),
        "FOSCTTM": float(np.nanmean(calc_domainAveraged_FOSCTTM(emb_a, emb_b))),
    }


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
    results_csv = root_dir / "results_har.csv"

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
                    "label_transfer": np.nan,
                    "alignment_score": np.nan,
                    "FOSCTTM": np.nan,
                    "seed": seed,
                    "runtime_sec": float(time.perf_counter() - start),
                    "status": f"error: {exc}",
                }

            all_rows.append(row)
            append_result_row(results_csv, row)
            if row["status"] == "ok":
                print(
                    f"  Acc={row['label_transfer']:.4f} | "
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
