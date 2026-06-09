"""
Sketchy photo-to-sketch alignment benchmark.

Input arrays store features in all columns except the final two, labels in the
penultimate column, and object IDs in the final column.
"""
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rf-mali-matplotlib")

import numpy as np
np.int = int
import pandas as pd
import phate
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

from src.fosta import FoSTA
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf
from src.mali import MALI
from src.Pamona.eval import test_alignment_score, test_transfer_accuracy, calc_domainAveraged_FOSCTTM
from src.pamona import Pamona
from scripts.ad_experiments.ad_experiment_utils import save_embedding_plots, save_embeddings


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = ROOT / "data_sketchy"
# SOURCE_DATA_PATH = DATA_ROOT / "photo_resnet18_embeddings.npy"
SOURCE_DATA_PATH = DATA_ROOT / "photo_dinov2_embeddings.npy"

# TARGET_DATA_PATH = DATA_ROOT / "sketch_resnet18_embeddings.npy"
TARGET_DATA_PATH = DATA_ROOT / "sketch_dinov2_embeddings.npy"

LABEL_DICT_PATH = DATA_ROOT / "label_dic"

SEEDS = [39041, 56089, 79121]
LABEL_MASKING_LEVEL_B = 0.50
N_COMPONENTS = 2
L2_NORMALIZE = False

# Count-based methods from ad_tree_pair are intentionally excluded here because
# the DINO features are signed and we use the same raw input for every method.
MODELS_TO_RUN = [
    # "Unintegrated",
    # "Unintegrated_PHATE",
    "FoSTA",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    "Pamona",
]


FOSTA_CONFIGS = {
    # "FoSTA_tauto": {
    #     "unlabeled_coupling": "predict_shared",
    #     "t": "auto",
    #     "class_weight": "balanced_subsample",
    #     "n_estimators": 500,
    # }
    "FoSTA_t2": {
        "t": 2,
        "n_estimators": 100,
    }
}

SUPERVISED_CLASSES = {
    "FoSTA": FoSTA,
    "KEMAlin": KEMAlin,
    "KEMArbf": KEMArbf,
    # "MALI": MALI,
    # "Pamona": Pamona,
}

RESULTS_ROOT = ROOT / "results_sketchy"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA AND BOOKKEEPING
# =============================================================================
def validate_probability(name, value):
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def validate_config():
    validate_probability("LABEL_MASKING_LEVEL_B", LABEL_MASKING_LEVEL_B)
    for path in [SOURCE_DATA_PATH, TARGET_DATA_PATH, LABEL_DICT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")


def save_experiment_metadata(output_dir, timestamp):
    metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "timestamp": timestamp,
        "source_path": str(SOURCE_DATA_PATH.relative_to(ROOT)),
        "target_path": str(TARGET_DATA_PATH.relative_to(ROOT)),
        "label_dict_path": str(LABEL_DICT_PATH.relative_to(ROOT)),
        "feature_columns": ":-2",
        "label_column": "-2",
        "object_id_column": "-1",
        "target_mask_unit": "source_object_id",
        "seeds": SEEDS,
        "label_masking_level_b": LABEL_MASKING_LEVEL_B,
        "n_components": N_COMPONENTS,
        "l2_normalize": L2_NORMALIZE,
        "models_to_run": MODELS_TO_RUN,
        "excluded_ad_tree_pair_methods": ["scVI", "scANVI", "LIGER", "Scanorama"],
        "fosta_configs": FOSTA_CONFIGS,
    }
    with (output_dir / "experiment_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def append_result_row(results_csv, row):
    pd.DataFrame([row]).to_csv(
        results_csv,
        mode="a",
        header=not results_csv.exists(),
        index=False,
    )


def load_label_name_mapping():
    with LABEL_DICT_PATH.open("rb") as f:
        raw_mapping = pickle.load(f)
    return {int(code): str(name) for code, name in raw_mapping.items()}


def map_label_names(labels):
    code_to_name = load_label_name_mapping()
    names = []
    for label in np.asarray(labels):
        try:
            names.append(code_to_name[int(label)])
        except (KeyError, TypeError, ValueError):
            names.append(str(label))
    return np.asarray(names)


def coerce_integral_column(values, name):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        return values.astype(str)
    return rounded.astype(int)


def load_domain(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected 2D array with at least 3 columns, got {path}: {arr.shape}")

    x = np.asarray(arr[:, :-2], dtype=float)
    labels = coerce_integral_column(arr[:, -2], f"{path.name} labels")
    object_ids = coerce_integral_column(arr[:, -1], f"{path.name} object IDs")

    if np.any(~np.isfinite(x)):
        raise ValueError(f"{path.name} features contain non-finite values.")

    return x, labels, object_ids


def l2_normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def make_source_object_mask(source_labels, source_object_ids, target_object_ids, frac, seed):
    source_labels = np.asarray(source_labels).astype(str)
    source_object_ids = np.asarray(source_object_ids).astype(str)
    target_object_ids = np.asarray(target_object_ids).astype(str)
    if frac <= 0:
        return np.zeros(target_object_ids.shape[0], dtype=bool)
    if np.unique(source_object_ids).size != source_object_ids.size:
        raise ValueError("Expected source/photo object IDs to be unique.")

    rng = np.random.default_rng(seed)
    masked_object_ids = []
    for label in np.unique(source_labels):
        class_object_ids = source_object_ids[source_labels == label]
        n_mask = int(np.floor(class_object_ids.size * frac))
        if n_mask > 0:
            masked_object_ids.extend(rng.choice(class_object_ids, size=n_mask, replace=False))

    return np.isin(target_object_ids, masked_object_ids)


def build_pair(seed):
    x_source, labels_source, object_ids_source = load_domain(SOURCE_DATA_PATH)
    x_target, labels_target_true, object_ids_target = load_domain(TARGET_DATA_PATH)

    if L2_NORMALIZE:
        x_source = l2_normalize_rows(x_source)
        x_target = l2_normalize_rows(x_target)

    target_mask = make_source_object_mask(
        labels_source,
        object_ids_source,
        object_ids_target,
        LABEL_MASKING_LEVEL_B,
        seed + 17,
    )
    labels_source_model = np.asarray(labels_source).copy()
    labels_target_model = np.asarray(labels_target_true).copy()
    labels_target_model[target_mask] = -1

    return {
        "x_source": x_source,
        "x_target": x_target,
        "labels_source_true": np.asarray(labels_source),
        "labels_target_true": np.asarray(labels_target_true),
        "labels_source_model": labels_source_model,
        "labels_target_model": labels_target_model,
        "object_ids_source": np.asarray(object_ids_source),
        "object_ids_target": np.asarray(object_ids_target),
        "target_mask": target_mask,
    }


def save_pair_metadata(output_dir, pair):
    np.savez_compressed(
        output_dir / "sketchy_labels_and_ids.npz",
        source_labels=pair["labels_source_true"],
        target_labels=pair["labels_target_true"],
        target_labels_obs=pair["labels_target_model"],
        source_object_ids=pair["object_ids_source"],
        target_object_ids=pair["object_ids_target"],
        target_label_mask=pair["target_mask"],
    )


# =============================================================================
# METHODS AND METRICS
# =============================================================================
def coerce_embedding_array(embedding):
    if isinstance(embedding, (list, tuple)):
        embedding = embedding[0] if len(embedding) == 1 else np.vstack([np.asarray(x) for x in embedding])

    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}.")
    if np.any(~np.isfinite(embedding)):
        raise ValueError("Embedding contains non-finite values.")
    return embedding[:, :N_COMPONENTS]


def run_unintegrated_pca(x_source, x_target, seed):
    x = np.vstack([x_source, x_target])
    return PCA(n_components=N_COMPONENTS, random_state=seed).fit_transform(x)


def run_unintegrated_phate(x_source, x_target, seed):
    x = np.vstack([x_source, x_target])
    return phate.PHATE(n_components=N_COMPONENTS, random_state=seed).fit_transform(x)


def run_method(method_name, pair, seed):
    if method_name == "Unintegrated":
        return method_name, run_unintegrated_pca(pair["x_source"], pair["x_target"], seed)

    if method_name == "Unintegrated_PHATE":
        return method_name, run_unintegrated_phate(pair["x_source"], pair["x_target"], seed)

    return run_supervised_method(method_name, pair, seed)


def run_supervised_method(method_name, pair, seed):
    if method_name == "FoSTA":
        for out_name, params in FOSTA_CONFIGS.items():
            model = FoSTA(n_components=N_COMPONENTS, random_state=seed, **params)
            embedding = model.fit_transform(
                pair["x_source"],
                pair["x_target"],
                pair["labels_source_model"],
                pair["labels_target_model"],
            )
            return out_name, embedding

    model = SUPERVISED_CLASSES[method_name](n_components=N_COMPONENTS, random_state=seed)
    embedding = model.fit_transform(
        pair["x_source"],
        pair["x_target"],
        pair["labels_source_model"],
        pair["labels_target_model"],
    )
    return method_name, embedding


def make_plot_specs(pair):
    n_source = pair["x_source"].shape[0]
    domains = np.array(["photo"] * n_source + ["sketch"] * pair["x_target"].shape[0])
    labels = map_label_names(np.concatenate([pair["labels_source_true"], pair["labels_target_true"]]))

    return [
        ("domain", domains, "tab10", "Domain"),
        ("labels", labels, "colorblind", "Ground Truth Label"),
    ]


def benchmark_method(method_name, embedding, pair, output_dir, seed=None):
    emb = coerce_embedding_array(embedding)
    n_source = pair["x_source"].shape[0]
    n_target = pair["x_target"].shape[0]
    if emb.shape[0] != n_source + n_target:
        raise ValueError(f"Embedding has {emb.shape[0]} rows; expected {n_source + n_target}.")

    emb_source = emb[:n_source]
    emb_target = emb[n_source:]
    mask_missing = pair["labels_target_model"] == -1

    foscttm = float(np.nanmean(calc_domainAveraged_FOSCTTM(
        emb_source,
        emb_target,
        ids1=pair["object_ids_source"],
        ids2=pair["object_ids_target"],
    )))
    alignment_score = float(test_alignment_score(emb_source, emb_target, random_state=seed))
    label_transfer = np.nan
    if np.any(mask_missing):
        label_transfer = test_transfer_accuracy(
            data1=emb_target[mask_missing],
            data2=emb_source,
            type1=pair["labels_target_true"][mask_missing],
            type2=pair["labels_source_true"],
        )

    method_dir = output_dir / method_name
    save_embeddings(method_dir, method_name, emb, n_source)
    save_embedding_plots(method_dir, method_name, emb, make_plot_specs(pair))

    return {
        "method": method_name,
        "label_transfer": float(label_transfer) if not np.isnan(label_transfer) else np.nan,
        "alignment_score": alignment_score,
        "FOSCTTM": foscttm,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = RESULTS_ROOT / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_metadata(root_dir, timestamp)
    results_csv = root_dir / "sketchy_alignment_results.csv"

    all_rows = []

    for seed in SEEDS:
        print(f"\n### STARTING SEED: {seed} ###")
        pair = build_pair(seed)
        seed_dir = root_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_pair_metadata(seed_dir, pair)

        for method_name in MODELS_TO_RUN:
            start = time.perf_counter()
            out_name = next(iter(FOSTA_CONFIGS)) if method_name == "FoSTA" else method_name
            try:
                print(f"Running {method_name}...")
                out_name, embedding = run_method(method_name, pair, seed)

                row = benchmark_method(out_name, embedding, pair, seed_dir, seed=seed)
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
