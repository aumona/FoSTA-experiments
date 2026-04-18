import numpy as np
np.int = int

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Robust imports
# -----------------------------------------------------------------------------
from utils.utils import dataprep

from utils.simulation_utils import (
    add_noise,
    random_rotate,
    random_feature_split,
    importance_split,
    alternating_importance_split,
    add_gaussian_noise_features_split,
    mask_labels_stratified
)
    
from src.Pamona.eval import (
    test_transfer_accuracy,
    test_alignment_score,
    calc_domainAveraged_FOSCTTM,
)

# Alignment models
from src.fosta import FoSTA
from src.rfmali import RFMALI
from src.mali import MALI
from src.pamona import Pamona
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_list_arg(value: str, cast=str) -> List:
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def parse_t_value(value):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "t must be either 'auto' or an integer, e.g. 2"
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_run_paths(results_root: Path) -> Tuple[Path, Path]:
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
    group_cols = ["dataset", "split", "method", "metric"]

    long_rows = []
    for metric in value_cols:
        tmp = df[["dataset", "split", "method", metric]].copy()
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


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_alignment_metrics(
    embedding: np.ndarray,
    y_source: np.ndarray,
    y_target_true: np.ndarray,
    mask_missing_target: np.ndarray,
) -> Dict[str, float]:
    n_source = len(y_source)
    n_target = len(y_target_true)

    if embedding.shape[0] != n_source + n_target:
        raise ValueError(
            f"Embedding shape mismatch: got {embedding.shape[0]} rows, expected {n_source + n_target}."
        )

    emb_source = np.asarray(embedding[:n_source], dtype=float)
    emb_target = np.asarray(embedding[n_source:], dtype=float)

    if np.any(mask_missing_target):
        label_transfer = test_transfer_accuracy(
            data1=emb_target[mask_missing_target],
            data2=emb_source,
            type1=y_target_true[mask_missing_target],
            type2=y_source,
        )
    else:
        label_transfer = np.nan

    alignment_score = test_alignment_score(emb_source, emb_target)
    foscttm = float(np.mean(calc_domainAveraged_FOSCTTM(emb_source, emb_target)))

    return {
        "label_transfer": float(label_transfer) if not np.isnan(label_transfer) else np.nan,
        "alignment_score": float(alignment_score),
        "foscttm": float(foscttm),
    }


# -----------------------------------------------------------------------------
# Data loading and exact old-style preprocessing
# -----------------------------------------------------------------------------

def load_dataset_frame(datasets_path: Path, data_name: str, transform='standardize') -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(datasets_path / f"{data_name}.csv")
    df, labels = dataprep(df, transform=transform)
    labels = np.asarray(labels).astype(int)
    return df, labels


def build_domains_exact_old_style(
    df: pd.DataFrame,
    labels: np.ndarray,
    split: str,
    seed: int,
    noise_sigma: float,
    signal_to_noise_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = split.lower()

    if split == "random":
        df1, df2 = random_feature_split(df.copy(), random_state=seed)

    elif split == "importance":
        df1, df2 = importance_split(df.copy(), labels)

    elif split in {"alternate_importance", "alternating_importance"}:
        df1, df2 = alternating_importance_split(df.copy(), labels)

    elif split in {"add_gaussian_noise_features"}:
        df1, df2 = add_gaussian_noise_features_split(
            df.copy(),
            signal_to_noise_ratio=signal_to_noise_ratio,
            random_state=seed,
        )

    elif split == "distort":
        df1 = df.copy()
        df2 = add_noise(df.copy(), sigma=noise_sigma, random_state=seed)


    elif split == "rotate":
        df1 = df.copy()
        df2 = random_rotate(df.copy(), random_state=seed)

    else:
        raise ValueError(f"Unknown split type: {split}")

    x_source = np.array(df2)
    x_target = np.array(df1)
    return x_source, x_target


def mask_target_labels(
    y_true: np.ndarray,
    mask_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_masked = mask_labels_stratified(
        y_true.copy(),
        mask_fraction=mask_fraction,
        random_state=seed,
    )
    mask_missing = y_masked == -1
    return y_masked.astype(int), mask_missing.astype(bool)


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def build_model(method: str, seed: int, cfg: argparse.Namespace):
    method = method.lower()

    if method == "rfmali":
        return RFMALI(
            embedder=cfg.embedder,
            n_components=cfg.n_components,
            n_estimators=cfg.n_estimators,
            t=cfg.t,
            beta=cfg.beta,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )

    if method == "fosta":
        return FoSTA(
            embedder=cfg.embedder,
            mu=cfg.mu,
            n_components=cfg.n_components,
            semantic_norm=cfg.semantic_norm,
            euclidean_mode=cfg.euclidean_mode,
            prior_correct=True,
            dpt=cfg.dpt,
            kernel_method=cfg.kernel_method,
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            t=cfg.t,
            beta=cfg.beta,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )
    
    if method == "fosta_dpt":
        return FoSTA(
            embedder=cfg.embedder,
            mu=cfg.mu,
            n_components=cfg.n_components,
            semantic_norm=cfg.semantic_norm,
            euclidean_mode=cfg.euclidean_mode,
            prior_correct=True,
            dpt=True,
            kernel_method=cfg.kernel_method,
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            t=cfg.t,
            beta=cfg.beta,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )

    if method == "fosta_no_prior":
        return FoSTA(
            embedder=cfg.embedder,
            mu=cfg.mu,
            n_components=cfg.n_components,
            semantic_norm=cfg.semantic_norm,
            euclidean_mode=cfg.euclidean_mode,
            prior_correct=False,
            dpt=cfg.dpt,
            kernel_method=cfg.kernel_method,
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            t=cfg.t,
            beta=cfg.beta,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )

    if method == "fosta_euclidean":
        return FoSTA(
            embedder=cfg.embedder,
            mu=cfg.mu,
            n_components=cfg.n_components,
            semantic_norm=cfg.semantic_norm,
            euclidean_mode=True,
            prior_correct=True,
            dpt=cfg.dpt,
            kernel_method=cfg.kernel_method,
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            t=cfg.t,
            beta=cfg.beta,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )

    if method == "mali":
        return MALI(
            embedder=cfg.embedder,
            n_components=cfg.n_components,
            t=cfg.t,
            distances='DPT',
            random_state=seed,
            verbose=cfg.verbose,
        )
    if method == "mali_nodpt":
        return MALI(
            embedder=cfg.embedder,
            n_components=cfg.n_components,
            t=cfg.t,
            distances='noDPT',
            random_state=seed,
            verbose=cfg.verbose,
        )

    if method == "pamona":
        return Pamona(
            n_components=cfg.n_components,
            embedder=cfg.embedder,
            gamma=cfg.gamma,
            random_state=seed,
        )

    if method == "kemalin":
        return KEMAlin(
            n_components=cfg.n_components,
            mu=cfg.mu,
        )

    if method == "kemarbf":
        return KEMArbf(
            n_components=cfg.n_components,
            mu=cfg.mu,
        )

    raise ValueError(f"Unknown method '{method}'.")


def fit_transform_model(model, x_source, x_target, y_source, y_target):
    if hasattr(model, "fit_transform"):
        return model.fit_transform(x_source, x_target, y_source, y_target)
    model.fit(x_source, x_target, y_source, y_target)
    if hasattr(model, "embedding_"):
        return model.embedding_
    raise RuntimeError(f"Model {type(model).__name__} has no fit_transform and no embedding_.")


# -----------------------------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------------------------

def run_experiment(cfg: argparse.Namespace) -> None:
    datasets_path = Path(cfg.datasets_path)
    results_root = Path(cfg.results_dir)
    ensure_dir(results_root)

    results_csv, config_json = make_run_paths(results_root)
    save_config(config_json, vars(cfg))

    datasets = parse_list_arg(cfg.datasets)
    methods = parse_list_arg(cfg.methods)
    splits = parse_list_arg(cfg.splits)
    seeds = parse_list_arg(cfg.seeds, cast=int)

    print(f"Results will be written to: {results_csv}")
    print(f"Config saved to: {config_json}")

    for data_name in datasets:
        print(f"\n=== Dataset: {data_name} ===")
        df, labels = load_dataset_frame(datasets_path, data_name)

        for seed in seeds:
            print(f"  Seed: {seed}")

            for split in splits:
                print(f"    Split: {split}")

                try:
                    x_source, x_target = build_domains_exact_old_style(
                        df=df,
                        labels=labels,
                        split=split,
                        seed=seed,
                        noise_sigma=cfg.noise_sigma,
                        signal_to_noise_ratio=cfg.signal_to_noise_ratio,
                    )

                    y_source = np.array(labels)
                    y_target_true = labels.copy()
                    y_target, mask_missing_target = mask_target_labels(
                        y_true=y_target_true,
                        mask_fraction=cfg.mask_fraction,
                        seed=seed,
                    )

                except Exception as e:
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": data_name,
                        "seed": seed,
                        "split": split,
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

                for method in methods:
                    print(f"      Method: {method}")

                    try:
                        model = build_model(method, seed, cfg)
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
                            "method": method,
                            "label_transfer": np.nan,
                            "alignment_score": np.nan,
                            "foscttm": np.nan,
                            "status": "fit_error",
                            "error": repr(e),
                        }
                        append_result(results_csv, row)
                        print(f"        Fit error: {e}")
                        continue

                    try:
                        metrics = compute_alignment_metrics(
                            embedding=embedding,
                            y_source=y_source,
                            y_target_true=y_target_true,
                            mask_missing_target=mask_missing_target,
                        )

                        row = {
                            "timestamp": datetime.now().isoformat(),
                            "dataset": data_name,
                            "seed": seed,
                            "split": split,
                            "method": method,
                            **metrics,
                            "status": "ok",
                            "error": "",
                        }
                        append_result(results_csv, row)
                        print(
                            f"        label_transfer={metrics['label_transfer']:.4f}, "
                            f"alignment_score={metrics['alignment_score']:.4f}, "
                            f"foscttm={metrics['foscttm']:.4f}"
                        )

                    except Exception as e:
                        row = {
                            "timestamp": datetime.now().isoformat(),
                            "dataset": data_name,
                            "seed": seed,
                            "split": split,
                            "method": method,
                            "label_transfer": np.nan,
                            "alignment_score": np.nan,
                            "foscttm": np.nan,
                            "status": "metric_error",
                            "error": repr(e),
                        }
                        append_result(results_csv, row)
                        print(f"        Metric error: {e}")

    raw_df = pd.read_csv(results_csv)
    ok_df = raw_df[raw_df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        print("\nNo successful runs. Raw results only were saved.")
        return

    summary_df = aggregate_results(ok_df)
    summary_path = results_csv.with_name(results_csv.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary ===")
    print(summary_df[["dataset", "split", "method", "metric", "summary"]].to_string(index=False))
    print(f"\nRaw results saved to: {results_csv}")
    print(f"Summary saved to: {summary_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UCI domain-alignment benchmark with transformed targets.")

    p.add_argument("--datasets-path", type=str, default="data_uci")
    p.add_argument("--results-dir", type=str, default="results_uci")

    p.add_argument(
        "--datasets",
        type=str,
        default="balance_scale,breast_cancer,crx,diabetes,ecoli_5,flare1,glass,heart_disease,heart_failure,hepatitis,ionosphere,iris,parkinsons,seeds,tic-tac-toe",
        help="Comma-separated dataset names without .csv",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="FoSTA,FoSTA_dpt,FoSTA_no_prior,FoSTA_euclidean,RFMALI,MALI,MALI_nodpt,Pamona,KEMAlin,KEMArbf",

    )
    p.add_argument(
        "--splits",
        type=str,
        default="add_gaussian_noise_features,random,importance,alternate_importance,rotate,distort",
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    p.add_argument("--mask-fraction", type=float, default=0.5)
    p.add_argument("--noise-sigma", type=float, default=0.5)
    p.add_argument("--signal-to-noise-ratio", type=float, default=0.1)

    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--embedder", type=str, default="PHATE")
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--semantic-norm", type=str, default="l2")
    p.add_argument("--euclidean-mode", action="store_true", default=False)
    p.add_argument("--prior-correct", action="store_true", default=True)
    p.add_argument("--dpt", action="store_true", default=False)
    p.add_argument("--kernel-method", type=str, default="gap")
    p.add_argument("--model-type", type=str, default="rf")
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--t", type=parse_t_value, default=2)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--verbose", type=int, default=1)

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_experiment(args)