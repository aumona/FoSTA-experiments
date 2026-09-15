import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
np.int = int

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.label_protocol import make_supervision_masks, validate_label_mask_perc
from utils.utils import dataprep
from utils.simulation_utils import (
    add_noise,
    random_rotate,
    random_feature_split,
    importance_split,
    alternating_importance_split,
    add_gaussian_noise_features_split,
)
from src.Pamona.eval import (
    test_transfer_accuracy,
    test_alignment_score,
    calc_domainAveraged_FOSCTTM,
)

from src.fosta import FoSTA as FoSTA
from src.fosta_icml import FoSTA as FoSTA_ICML
from src.mali import MALI
from src.pamona import Pamona
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

DATASETS_PATH = Path("data_uci")
RESULTS_DIR = Path("results_uci")

DATASETS = [
    "balance_scale",
    "breast_cancer",
    "crx",
    "diabetes",
    "ecoli_5",
    "flare1",
    "glass",
    "heart_disease",
    "heart_failure",
    "hepatitis",
    "ionosphere",
    "iris",
    "parkinsons",
    "seeds",
    "tic-tac-toe",
]

METHODS = [


    "FoSTA_gap_auto",  # Default


    "FoSTA_gap_t2",
    "FoSTA_kerf_auto",
    "FoSTA_umap",
    "FoSTA_dense",
    "FoSTA_et",
    "FoSTA_spectral",
    "FoSTA_dpt",
    "FoSTA_rotf",
    "FoSTA_no_propag",




    "MALI_auto",
    "MALI_t2",
    # "MALI_nodpt",

    "Pamona",

    "KEMAlin",
    "KEMArbf",
]

SPLITS = [
    "add_gaussian_noise_features",
    "random",
    "importance",
    "alternate_importance",
    "rotate",
    "distort",
]

SEEDS = list(range(5))

TRANSFORM = "standardize"
TEST_PERC = 0.2  # Fixed, shared held-out test pairs within each dataset/seed.

# Fraction masked within the remaining 80% training pool, not the full dataset.
# Masks are shared across modalities and nested across proportions. Test labels
# are always hidden, while all features remain available to alignment methods.
# LABEL_MASK_PERC = [0.2, 0.4, 0.6, 0.8]  # Reasonable range of masking levels to explore.
LABEL_MASK_PERC = [0]



NOISE_SIGMA = 0.5  # reasonable amount of noise
SIGNAL_TO_NOISE_RATIO = 0.1


# Shared parameters across models
N_COMPONENTS = 2
N_JOBS = -1
VERBOSE = 1


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_run_paths(results_root: Path):
    ensure_dir(results_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_root / f"results_{stamp}.csv"
    json_path = results_root / f"config_{stamp}.json"
    return csv_path, json_path


def save_config(path: Path, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def append_result(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["label_transfer", "alignment_score", "foscttm"]
    group_cols = ["dataset", "split", "mask_fraction", "method", "metric"]

    long_rows = []
    for metric in value_cols:
        tmp = df[["dataset", "split", "mask_fraction", "method", metric]].copy()
        tmp = tmp.rename(columns={metric: "value"})
        tmp["metric"] = metric
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    out = (
        long_df.groupby(group_cols, dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(group_cols)
    )
    out["summary"] = out.apply(
        lambda r: f"{r['mean']:.4f} ± {0.0 if pd.isna(r['std']) else r['std']:.4f}",
        axis=1,
    )
    return out


def sanitize_embedding(embedding: np.ndarray, tol: float = 1000) -> np.ndarray:
    embedding = np.asarray(embedding)

    if np.iscomplexobj(embedding):
        embedding = np.real_if_close(embedding, tol=tol)

        if np.iscomplexobj(embedding):
            max_imag = np.max(np.abs(np.imag(embedding)))
            if max_imag < 1e-6:
                embedding = np.real(embedding)
            else:
                raise ValueError(
                    f"Embedding contains non-negligible imaginary part "
                    f"(max |Im| = {max_imag:.3e})"
                )

    return np.asarray(embedding, dtype=float)


# =============================================================================
# METRICS
# =============================================================================

def compute_alignment_metrics(
    embedding: np.ndarray,
    y_source_true: np.ndarray,
    y_target_true: np.ndarray,
    visible_training: np.ndarray,
    test_mask: np.ndarray,
):
    n_source = len(y_source_true)
    n_target = len(y_target_true)

    if embedding.shape[0] != n_source + n_target:
        raise ValueError(
            f"Embedding shape mismatch: got {embedding.shape[0]} rows, expected {n_source + n_target}."
        )

    emb_source = np.asarray(embedding[:n_source], dtype=float)
    emb_target = np.asarray(embedding[n_source:], dtype=float)

    if np.any(visible_training) and np.any(test_mask):
        a_to_b = test_transfer_accuracy(
            data1=emb_target[test_mask],
            data2=emb_source[visible_training],
            type1=y_target_true[test_mask],
            type2=y_source_true[visible_training],
        )
        b_to_a = test_transfer_accuracy(
            data1=emb_source[test_mask],
            data2=emb_target[visible_training],
            type1=y_source_true[test_mask],
            type2=y_target_true[visible_training],
        )
        label_transfer = (a_to_b + b_to_a) / 2
    else:
        label_transfer = np.nan

    alignment_score = test_alignment_score(emb_source, emb_target)
    foscttm = float(np.mean(calc_domainAveraged_FOSCTTM(emb_source, emb_target)))

    return {
        "label_transfer": float(label_transfer) if not np.isnan(label_transfer) else np.nan,
        "alignment_score": float(alignment_score),
        "foscttm": float(foscttm),
    }


# =============================================================================
# DATA
# =============================================================================

def load_dataset_frame(datasets_path: Path, data_name: str):
    df = pd.read_csv(datasets_path / f"{data_name}.csv")
    df, labels = dataprep(df, transform=TRANSFORM)
    labels = np.asarray(labels).astype(int)
    return df, labels


def build_domains(df, labels, split, seed):
    split = split.lower()

    if split == "random":
        df1, df2 = random_feature_split(df.copy(), random_state=seed)

    elif split == "importance":
        df1, df2 = importance_split(df.copy(), labels)

    elif split in {"alternate_importance", "alternating_importance"}:
        df1, df2 = alternating_importance_split(df.copy(), labels)

    elif split == "add_gaussian_noise_features":
        df1, df2 = add_gaussian_noise_features_split(
            df.copy(),
            signal_to_noise_ratio=SIGNAL_TO_NOISE_RATIO,
            sigma=1.0,
            random_state=seed,
        )

    elif split == "distort":
        df1 = df.copy()
        df2 = add_noise(df.copy(), sigma=NOISE_SIGMA, random_state=seed)

    elif split == "rotate":
        df1 = df.copy()
        df2 = random_rotate(df.copy(), random_state=seed)

    else:
        raise ValueError(f"Unknown split type: {split}")

    x_source = np.array(df2)
    x_target = np.array(df1)
    return x_source, x_target


def mask_pair_labels(y_true, mask_fraction, seed):
    """Hide test labels and jointly mask a nested subset of training pairs."""
    visible, test_mask = make_supervision_masks(y_true, mask_fraction, seed, TEST_PERC)
    observed = np.asarray(y_true, dtype=int).copy()
    observed[~visible] = -1
    return observed.copy(), observed.copy(), visible, test_mask


# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_model(method: str, seed: int):
    m = method.lower()

    if m == "fosta_icml_t2":
        return FoSTA_ICML(
            t=2,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    if m == "fosta_icml_auto":
        return FoSTA_ICML(
            n_components=N_COMPONENTS,
            t='auto',
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

    if m == "fosta_gap_t2":
        return FoSTA(
            n_components=N_COMPONENTS,
            kernel_method='gap',
            t=2,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    
    if m == "fosta_gap_auto":
        return FoSTA(
            n_components=N_COMPONENTS,
            kernel_method='gap',
            t='auto',
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    if m == "fosta_kerf_t2":
        return FoSTA(
            n_components=N_COMPONENTS,
            kernel_method='kerf',
            t=2,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    
    if m == "fosta_kerf_auto":
        return FoSTA(
            n_components=N_COMPONENTS,
            kernel_method='kerf',
            t='auto',
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    
    if m == 'fosta_umap':
        return FoSTA(
            embedder='UMAP',
            n_components=N_COMPONENTS,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    
    if m == 'fosta_dense':
        return FoSTA(
            n_components=N_COMPONENTS,
            ot_solver='dense',
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
    
    if m == 'fosta_et':
        return FoSTA(
            n_components=N_COMPONENTS,
            model_type='et',
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

    if m == "fosta_spectral":
        return FoSTA(
            n_components=N_COMPONENTS,
            embedder="spectral",
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

    if m == "fosta_dpt":
        return FoSTA(
            n_components=N_COMPONENTS,
            dpt=True,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

    if m == "fosta_rotf":
        return FoSTA(
            n_components=N_COMPONENTS,
            model_type="rotf",
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

    if m == "fosta_no_propag":
        return FoSTA(
            n_components=N_COMPONENTS,
            propagate=False,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )
   
    if m in {"mali_auto", "mali_t2"}:
        return MALI(
            t="auto" if m == "mali_auto" else 2,
            n_components=N_COMPONENTS,
            random_state=seed,
            verbose=VERBOSE,
        )

    if m == "mali_nodpt":
        return MALI(
            n_components=N_COMPONENTS,
            distances="noDPT",
            random_state=seed,
            verbose=VERBOSE,
        )

    if m == "pamona":
        return Pamona(
            n_components=N_COMPONENTS,
            random_state=seed,
        )

    if m == "kemalin":
        return KEMAlin(
            n_components=N_COMPONENTS,
        )

    if m == "kemarbf":
        return KEMArbf(
            n_components=N_COMPONENTS,
        )

    raise ValueError(f"Unknown method '{method}'.")


def fit_transform_model(model, x_source, x_target, y_source, y_target):
    if hasattr(model, "fit_transform"):
        return model.fit_transform(x_source, x_target, y_source, y_target)

    model.fit(x_source, x_target, y_source, y_target)
    if hasattr(model, "embedding_"):
        return model.embedding_

    raise RuntimeError(f"Model {type(model).__name__} has no fit_transform and no embedding_.")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    validate_label_mask_perc(LABEL_MASK_PERC)
    validate_label_mask_perc([TEST_PERC])
    if not 0 < TEST_PERC < 1:
        raise ValueError("TEST_PERC must be strictly between 0 and 1.")
    ensure_dir(RESULTS_DIR)
    results_csv, config_json = make_run_paths(RESULTS_DIR)

    config = {
        "datasets_path": str(DATASETS_PATH),
        "results_dir": str(RESULTS_DIR),
        "datasets": DATASETS,
        "methods": METHODS,
        "splits": SPLITS,
        "seeds": SEEDS,
        "mask_fractions": LABEL_MASK_PERC,
        "test_perc": TEST_PERC,
        "test_split": "seeded stratified shared pairs, fixed across masking levels and domain splits within a seed",
        "mask_count": "floor(p * number of training pairs), with the same mask shared across domains",
        "training_masks": "nested prefixes of one seeded stratified ordering",
        "test_labels": "always hidden in both domains; test features remain available for alignment",
        "label_transfer": "average of A-labeled-training to B-test and B-labeled-training to A-test",
        "alignment_metrics": "Alignment Score and FOSCTTM on all embedded observations",
        "noise_sigma": NOISE_SIGMA,
        "signal_to_noise_ratio": SIGNAL_TO_NOISE_RATIO,
        "n_components": N_COMPONENTS,
        "n_jobs": N_JOBS,
        "verbose": VERBOSE,
    }
    save_config(config_json, config)

    print(f"Results will be written to: {results_csv}")
    print(f"Config saved to: {config_json}")

    for data_name in DATASETS:
        print(f"\n=== Dataset: {data_name} ===")
        df, labels = load_dataset_frame(DATASETS_PATH, data_name)

        for seed in SEEDS:
            print(f"  Seed: {seed}")
        
            for split in SPLITS:
                print(f"    Split: {split}")
        
                try:
                    x_source, x_target = build_domains(df, labels, split, seed)
                    y_source_true = labels.copy()
                    y_target_true = labels.copy()
        
                except Exception as e:
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": data_name,
                        "seed": seed,
                        "split": split,
                        "mask_fraction": np.nan,
                        "method": None,
                        "label_transfer": np.nan,
                        "alignment_score": np.nan,
                        "foscttm": np.nan,
                        "status": "data_error",
                        "error": repr(e),
                    }
                    append_result(results_csv, row)
                    print(f"      Data error: {e}")
                    continue
        
                for mask_fraction in LABEL_MASK_PERC:
                    print(f"      Mask fraction: {mask_fraction}")
        
                    try:
                        y_source, y_target, visible_training, test_mask = mask_pair_labels(
                            y_true=y_target_true,
                            mask_fraction=mask_fraction,
                            seed=seed,
                        )
        
                    except Exception as e:
                        row = {
                            "timestamp": datetime.now().isoformat(),
                            "dataset": data_name,
                            "seed": seed,
                            "split": split,
                            "mask_fraction": mask_fraction,
                            "method": None,
                            "label_transfer": np.nan,
                            "alignment_score": np.nan,
                            "foscttm": np.nan,
                            "status": "mask_error",
                            "error": repr(e),
                        }
                        append_result(results_csv, row)
                        print(f"        Mask error: {e}")
                        continue
        
                    for method in METHODS:
                        print(f"        Method: {method}")
        
                        try:
                            model = build_model(method, seed)
                            embedding = fit_transform_model(
                                model=model,
                                x_source=x_source,
                                x_target=x_target,
                                y_source=y_source,
                                y_target=y_target,
                            )
                            embedding = sanitize_embedding(embedding)
        
                        except Exception as e:
                            row = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset": data_name,
                                "seed": seed,
                                "split": split,
                                "mask_fraction": mask_fraction,
                                "method": method,
                                "label_transfer": np.nan,
                                "alignment_score": np.nan,
                                "foscttm": np.nan,
                                "status": "fit_error",
                                "error": repr(e),
                            }
                            append_result(results_csv, row)
                            print(f"          Fit error: {e}")
                            continue
        
                        try:
                            metrics = compute_alignment_metrics(
                                embedding=embedding,
                                y_source_true=y_source_true,
                                y_target_true=y_target_true,
                                visible_training=visible_training,
                                test_mask=test_mask,
                            )
        
                            row = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset": data_name,
                                "seed": seed,
                                "split": split,
                                "mask_fraction": mask_fraction,
                                "method": method,
                                **metrics,
                                "status": "ok",
                                "error": "",
                            }
                            append_result(results_csv, row)
        
                            print(
                                f"          label_transfer={metrics['label_transfer']:.4f}, "
                                f"alignment_score={metrics['alignment_score']:.4f}, "
                                f"foscttm={metrics['foscttm']:.4f}"
                            )
        
                        except Exception as e:
                            row = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset": data_name,
                                "seed": seed,
                                "split": split,
                                "mask_fraction": mask_fraction,
                                "method": method,
                                "label_transfer": np.nan,
                                "alignment_score": np.nan,
                                "foscttm": np.nan,
                                "status": "metric_error",
                                "error": repr(e),
                            }
                            append_result(results_csv, row)
                            print(f"          Metric error: {e}")

    raw_df = pd.read_csv(results_csv)
    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    if len(ok_df) == 0:
        print("\nNo successful runs. Raw results only were saved.")
        return

    summary_df = aggregate_results(ok_df)
    summary_path = results_csv.with_name(results_csv.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary ===")
    print(summary_df[["dataset", "split", "mask_fraction", "method", "metric", "summary"]].to_string(index=False))
    print(f"\nRaw results saved to: {results_csv}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
