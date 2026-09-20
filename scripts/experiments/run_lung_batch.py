import os

# Configure XLA before imports that can initialize JAX (including scvi).
# JAX 0.6.2's Triton GEMM compiler can fail on HiRef's matrix shapes with
# "Too small divisible part of the contracting dimension" (openxla/xla#33157).
# Keep GPU execution, but use the non-Triton GEMM path. Honor explicit overrides.
if "--xla_gpu_enable_triton_gemm" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
    ).strip()


import scanpy as sc
import numpy as np
import pandas as pd
import sys
import json
import platform
import atexit
import logging
import re
import threading
from importlib.metadata import PackageNotFoundError, version
import warnings
import scanorama
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

from scib_metrics.benchmark import Benchmarker
import scvi
import pyliger
import torch

warnings.filterwarnings("ignore")

# Resolve the repository root when this script is launched by filename.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
from src.fosta import FoSTA
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf
from src.mali import MALI
from src.pamona import Pamona

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = os.path.join(PROJECT_ROOT, "data_sc", "lung_batches.h5ad")
BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, "results_sc_experiments")

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"
# BATCH_LIST = ["A1", "A2", "A3", "A4", "A5", "A6"]
# BATCH_LIST = ['B1', 'B2', 'B3', 'B4']
BATCH_LIST = ['1', '2', '3', '4', '5', '6']

SEEDS = [39041, 56089, 79121, 444, 777] 
MASK = True
MASK_FRACTION = 0.50 
MIN_CELLS_FOR_MASKING = 10 

PCA_COMPONENTS = 30
# Apply PCA inputs to FoSTA, KEMAlin, KEMArbf, MALI, and Pamona.
# False uses the selected highly variable gene expression directly.
USE_PCA_FOR_SUPERVISED = False
N_DIM = 2

# Prefer Apple Metal, with CPU as the fallback.
if torch.backends.mps.is_available():
    TRAINING_ACCELERATOR = "mps"
else:
    TRAINING_ACCELERATOR = "cpu"

# Set to [] to run Unintegrated only
MODELS_TO_RUN = [
    "scVI",
    "scANVI", 
    "LIGER", 
    "Scanorama", 
    "FoSTA", 
    "KEMAlin", 
    "KEMArbf", 
    "MALI", 
    "Pamona"
]

FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "t": 2,
    }
}

SUPERVISED_CLASSES = {
    "FoSTA": FoSTA,
    "KEMAlin": KEMAlin,
    "KEMArbf": KEMArbf,
    "MALI": MALI,
    "Pamona": Pamona
}

# =============================================================================
# HELPERS
# =============================================================================
class TimestampedTee:
    """Mirror a Python output stream into a shared, line-oriented run log."""

    def __init__(self, terminal, logfile, lock):
        self.terminal = terminal
        self.logfile = logfile
        self.lock = lock
        self.pending = ""

    def _log_line(self, line):
        # Progress bars use carriage returns and ANSI cursor/color sequences.
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        if line.strip():
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            self.logfile.write(f"[{stamp}] {line}\n")
            self.logfile.flush()

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.pending += text
            lines = re.split(r"[\r\n]", self.pending)
            self.pending = lines.pop()
            for line in lines:
                self._log_line(line)
        return len(text)

    def flush(self):
        with self.lock:
            if self.pending:
                self._log_line(self.pending)
                self.pending = ""
            self.terminal.flush()
            self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def start_run_logging(result_dir):
    path = os.path.join(result_dir, "run.log")
    logfile = open(path, "a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TimestampedTee(original_stdout, logfile, lock)
    sys.stderr = TimestampedTee(original_stderr, logfile, lock)
    # Libraries may have installed logging handlers before the tee was enabled.
    loggers = [logging.getLogger()] + [
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is original_stdout:
                    handler.setStream(sys.stdout)
                elif handler.stream is original_stderr:
                    handler.setStream(sys.stderr)
    # Keep the file open through interpreter shutdown so late errors are saved.
    atexit.register(sys.stdout.flush)
    atexit.register(sys.stderr.flush)
    print(f"Logging stdout and stderr to: {path}")
    return path


def save_experiment_metadata(result_dir, adata, seed=None, batches=None):
    package_versions = {}
    for package in ("numpy", "pandas", "scanpy", "anndata", "scvi-tools",
                    "scib-metrics", "torch", "scikit-learn", "scanorama", "pyliger"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": os.path.abspath(__file__),
        "data_path": DATA_PATH,
        "result_dir": os.path.abspath(result_dir),
        "log_file": RUN_LOG_PATH,
        "seeds": SEEDS,
        "batches": BATCH_LIST,
        "batch_pairs": list(combinations(BATCH_LIST, 2)),
        "batch_key": BATCH_KEY,
        "label_key": LABEL_KEY,
        "masking": {
            "enabled": MASK,
            "fraction": MASK_FRACTION,
            "min_cells_for_masking": MIN_CELLS_FOR_MASKING,
            "eligibility": "Within-domain class count >= min_cells_for_masking / fraction",
            "sampling": "Class-stratified independently within each unpaired domain",
            "random_state": "CURRENT_SEED for each domain",
            "unlabeled_category": "Unknown",
        },
        "preprocessing": {
            "highly_variable_genes": 2000,
            "hvg_flavor": "cell_ranger",
            "hvg_batch_key": BATCH_KEY,
            "computed_pca_components": 30,
            "use_pca_for_supervised": USE_PCA_FOR_SUPERVISED,
            "supervised_pca_components": PCA_COMPONENTS,
            "supervised_input": "PCA" if USE_PCA_FOR_SUPERVISED else "Highly variable gene expression",
        },
        "models": {
            "requested": MODELS_TO_RUN,
            "always_evaluated": ["Unintegrated"],
            "n_components": N_DIM,
            "scanorama_dimred": N_DIM,
            "scanorama_approx": False,
            "fosta_configs": FOSTA_CONFIGS,
            "scvi_training": {"accelerator": TRAINING_ACCELERATOR, "devices": 1, "max_epochs": "scvi default"},
            "scanvi_training": {
                "max_epochs": 20, "n_samples_per_label": 100,
                "accelerator": TRAINING_ACCELERATOR, "devices": 1, "drop_last": True,
            },
        },
        "evaluation": {
            "embedding_dimensions": N_DIM,
            "scope": "All cells in the batch pair",
            "label_key": "ground_truth_labels",
            "ground_truth_labels_modified": False,
            "min_max_scale": False,
            "aggregate_weights": {"bio_conservation": 0.6, "batch_correction": 0.4},
        },
        "input_shape": {"cells": int(adata.n_obs), "genes": int(adata.n_vars)},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": package_versions},
    }
    if seed is not None:
        metadata["current_seed"] = seed
        metadata["current_batch_pair"] = list(batches)
        metadata["masking_counts"] = [
            {"batch": str(batch), "cell_type": str(label), "cells": len(group),
             "masked_cells": int(group["is_masked"].sum())}
            for (batch, label), group in adata.obs.groupby(
                [BATCH_KEY, "ground_truth_labels"], observed=True
            )
        ]
    path = os.path.join(result_dir, "experiment_metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")


def stratified_sample_mask(obs_df, label_col, frac, min_cells, seed):
    counts = obs_df[label_col].value_counts()
    eligible_labels = counts[counts >= (min_cells / frac)].index
    mask_indices = []
    for label in eligible_labels:
        label_idx = obs_df[obs_df[label_col] == label].index
        _, masked = train_test_split(label_idx, test_size=frac, random_state=seed)
        mask_indices.extend(masked)
    return mask_indices

def add_custom_aggregates(df_res):
    df = df_res.copy()
    if "Metric Type" not in df.index: return df
    metric_type = df.loc["Metric Type"]
    score_df = df.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    bio_cols = [c for c in metric_type.index[metric_type == "Bio conservation"] if c not in {"Bio conservation", "Total"}]
    batch_cols = [c for c in metric_type.index[metric_type == "Batch correction"] if c not in {"Batch correction", "Total"}]
    score_df["Bio conservation mean"] = score_df[bio_cols].mean(axis=1)
    score_df["Batch correction mean"] = score_df[batch_cols].mean(axis=1)
    score_df["Total mean"] = (0.6 * score_df["Bio conservation mean"] + 0.4 * score_df["Batch correction mean"])
    metric_type_ext = metric_type.copy()
    metric_type_ext["Bio conservation mean"] = "Aggregate score"; metric_type_ext["Batch correction mean"] = "Aggregate score"; metric_type_ext["Total mean"] = "Aggregate score"
    return pd.concat([score_df, metric_type_ext.to_frame().T.rename(index={0: "Metric Type"})])

def benchmark_method_and_update_csv(adata, method_key, metrics_csv, seed):
    # Label masking affects training; evaluation always includes all cells.
    embedding = np.asarray(adata.obsm[method_key])
    expected_shape = (adata.n_obs, N_DIM)
    if embedding.shape != expected_shape:
        raise ValueError(
            f"{method_key}: expected full-cell embedding {expected_shape}, "
            f"got {embedding.shape}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{method_key}: embedding contains non-finite values.")
    print(f"Benchmarking {method_key} (Seed {seed}) - FULL DATASET...")
    
    bm = Benchmarker(adata, batch_key=BATCH_KEY, label_key="ground_truth_labels", 
                    embedding_obsm_keys=[method_key], n_jobs=-1)
    bm.benchmark()
    df_res = add_custom_aggregates(bm.get_results(min_max_scale=False))
    metric_df = df_res.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    current = metric_df.loc[[method_key]].copy()
    current["Seed"] = seed 
    current.to_csv(metrics_csv, mode='a', header=not os.path.exists(metrics_csv))
    return current

def save_method_plot(adata, method_key, result_dir):
    # Standard plotting using first 2 dims of the specified key
    adata.obsm["X_2d_viz"] = adata.obsm[method_key][:, :2]
    
    for col, suffix in [(LABEL_KEY, "masked"), ("ground_truth_labels", "truth")]:
        fig = sc.pl.embedding(adata, basis="X_2d_viz", color=[BATCH_KEY, col],
                             show=False, return_fig=True, title=[f"{method_key} Batch", f"{method_key} {suffix}"])
        fig.savefig(os.path.join(result_dir, f"plot_{method_key}_{suffix}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

def prepare_fosta_labels(series):
    s = series.astype(str)
    labels = s.values.copy()
    mask_missing = (s == "nan") | (s == "Unknown") | (s == "None")
    labels[mask_missing] = "-1"
    return labels

# =============================================================================
# MAIN LOOP
# =============================================================================
ROOT_RESULT_DIR = os.path.join(BASE_RESULT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(ROOT_RESULT_DIR, exist_ok=True)
RUN_LOG_PATH = start_run_logging(ROOT_RESULT_DIR)
print(f"Training accelerator: {TRAINING_ACCELERATOR}")
print(f"Loading data: {DATA_PATH}")
full_adata_orig = sc.read(DATA_PATH)
save_experiment_metadata(ROOT_RESULT_DIR, full_adata_orig)

for CURRENT_SEED in SEEDS:
    print(f"\n### STARTING SEED: {CURRENT_SEED} ###")
    np.random.seed(CURRENT_SEED)
    scvi.settings.seed = CURRENT_SEED

    for BATCH_1, BATCH_2 in combinations(BATCH_LIST, 2):
        PAIR_NAME = f"{BATCH_1}_vs_{BATCH_2}"
        RESULT_DIR = os.path.join(ROOT_RESULT_DIR, f"seed_{CURRENT_SEED}", PAIR_NAME)
        os.makedirs(RESULT_DIR, exist_ok=True)
        METRICS_CSV = os.path.join(RESULT_DIR, f"benchmark_metrics_seed{CURRENT_SEED}.csv")
        
        print(f"\nRUNNING PAIR: {PAIR_NAME}")
        adata = full_adata_orig[full_adata_orig.obs[BATCH_KEY].isin([BATCH_1, BATCH_2])].copy()
        adata.obs["ground_truth_labels"] = adata.obs[LABEL_KEY].copy()
        adata.obs["is_masked"] = False
        
        if MASK:
            if "Unknown" not in adata.obs[LABEL_KEY].cat.categories:
                adata.obs[LABEL_KEY] = adata.obs[LABEL_KEY].cat.add_categories(["Unknown"])
            # Mask each unpaired domain independently, including class eligibility.
            mask_idx = []
            for batch in (BATCH_1, BATCH_2):
                batch_obs = adata.obs.loc[adata.obs[BATCH_KEY] == batch]
                mask_idx.extend(stratified_sample_mask(
                    batch_obs, "ground_truth_labels", MASK_FRACTION,
                    MIN_CELLS_FOR_MASKING, CURRENT_SEED,
                ))
            adata.obs.loc[mask_idx, LABEL_KEY] = "Unknown"
            adata.obs.loc[mask_idx, "is_masked"] = True

        save_experiment_metadata(RESULT_DIR, adata, CURRENT_SEED, (BATCH_1, BATCH_2))
        
        adata.layers["counts"] = adata.X.copy()
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key=BATCH_KEY)
        adata = adata[:, adata.var.highly_variable].copy()
        
        # Shared 30D PCA input for supervised models
        sc.tl.pca(adata, n_comps=30)
        
        # 0. Unintegrated
        # Evaluate in N_DIM dimensions to match the integrated embeddings
        adata.obsm["Unintegrated"] = adata.obsm["X_pca"][:, :N_DIM].copy()
        benchmark_method_and_update_csv(adata, "Unintegrated", METRICS_CSV, CURRENT_SEED)
        save_method_plot(adata, "Unintegrated", RESULT_DIR)

        # Efficiency: Skip model training if no models are requested
        if not MODELS_TO_RUN:
            print("MODELS_TO_RUN is empty. Saving Unintegrated result and continuing...")
            adata.write(os.path.join(RESULT_DIR, "adata_final.h5ad"))
            continue

        # 1. scVI / scANVI
        if any(m in MODELS_TO_RUN for m in ["scVI", "scANVI"]):
            scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
            vae = scvi.model.SCVI(adata, n_latent=N_DIM)
            vae.train(accelerator=TRAINING_ACCELERATOR, devices=1)
            for m in ["scVI", "scANVI"]:
                if m in MODELS_TO_RUN:
                    if m == "scVI":
                        adata.obsm[m] = vae.get_latent_representation()
                    else:
                        lvae = scvi.model.SCANVI.from_scvi_model(vae, adata=adata, labels_key=LABEL_KEY, unlabeled_category="Unknown")
                        # The label-subsampled loader can end in a single cell,
                        # which BatchNorm cannot process during training.
                        # Only training minibatches are dropped; evaluation uses all cells.
                        lvae.train(
                            max_epochs=20, n_samples_per_label=100,
                            accelerator=TRAINING_ACCELERATOR, devices=1,
                            datasplitter_kwargs={"drop_last": True},
                        )
                        adata.obsm[m] = lvae.get_latent_representation()
                    benchmark_method_and_update_csv(adata, m, METRICS_CSV, CURRENT_SEED)
                    save_method_plot(adata, m, RESULT_DIR)

        # 2. LIGER
        if "LIGER" in MODELS_TO_RUN:
            bdata = adata.copy()
            bdata.X = bdata.layers["counts"]
            batch_cats = list(bdata.obs[BATCH_KEY].cat.categories)
            adata_list = [bdata[bdata.obs[BATCH_KEY] == b].copy() for b in batch_cats]
            for i, ad in enumerate(adata_list):
                ad.uns["sample_name"], ad.uns["var_gene_idx"] = batch_cats[i], np.arange(bdata.n_vars)
            liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
            liger_data.var_genes = bdata.var_names
            pyliger.normalize(liger_data)
            with np.errstate(divide="ignore", invalid="ignore"): pyliger.scale_not_center(liger_data)
            pyliger.optimize_ALS(liger_data, k=N_DIM, rand_seed=CURRENT_SEED)
            pyliger.quantile_norm(liger_data)
            liger_res = np.zeros((adata.shape[0], N_DIM))
            for i, b in enumerate(batch_cats):
                liger_res[adata.obs[BATCH_KEY] == b] = liger_data.adata_list[i].obsm["H_norm"]
            adata.obsm["LIGER"] = liger_res
            benchmark_method_and_update_csv(adata, "LIGER", METRICS_CSV, CURRENT_SEED)
            save_method_plot(adata, "LIGER", RESULT_DIR)

        # 3. Scanorama
        if "Scanorama" in MODELS_TO_RUN:
            batch_cats = adata.obs[BATCH_KEY].cat.categories
            adata_list = [adata[adata.obs[BATCH_KEY] == b].copy() for b in batch_cats]
            for ad in adata_list:
                sc.pp.normalize_total(ad, target_sum=1e4)
                sc.pp.log1p(ad)
            # Integrate directly in the benchmark dimension, without truncating.
            # Exact search avoids the Annoy backend returning only one neighbour.
            scanorama.integrate_scanpy(adata_list, dimred=N_DIM, approx=False)
            adata.obsm["Scanorama"] = np.zeros((adata.shape[0], N_DIM))
            for i, b in enumerate(batch_cats):
                adata.obsm["Scanorama"][adata.obs[BATCH_KEY] == b] = adata_list[i].obsm["X_scanorama"]
            benchmark_method_and_update_csv(adata, "Scanorama", METRICS_CSV, CURRENT_SEED)
            save_method_plot(adata, "Scanorama", RESULT_DIR)

        # 4. Supervised Models
        active_supervised = [m for m in MODELS_TO_RUN if m in SUPERVISED_CLASSES]
        if active_supervised:
            idx_a, idx_b = (adata.obs[BATCH_KEY] == BATCH_1), (adata.obs[BATCH_KEY] == BATCH_2)
            if USE_PCA_FOR_SUPERVISED:
                x_a, x_b = adata[idx_a].obsm["X_pca"][:, :PCA_COMPONENTS], adata[idx_b].obsm["X_pca"][:, :PCA_COMPONENTS]
            else:
                x_a, x_b = adata[idx_a].X, adata[idx_b].X
                x_a = x_a.toarray() if issparse(x_a) else np.asarray(x_a)
                x_b = x_b.toarray() if issparse(x_b) else np.asarray(x_b)
            y_a_f, y_b_f = prepare_fosta_labels(adata[idx_a].obs[LABEL_KEY]), prepare_fosta_labels(adata[idx_b].obs[LABEL_KEY])

            for m_name in active_supervised:
                if m_name == "FoSTA":
                    for f_name, f_params in FOSTA_CONFIGS.items():
                        obj = SUPERVISED_CLASSES[m_name](n_components=N_DIM, random_state=CURRENT_SEED, **f_params)
                        emb = obj.fit_transform(x_a, x_b, y_a_f, y_b_f)
                        full_emb = np.zeros((adata.shape[0], N_DIM))
                        full_emb[idx_a], full_emb[idx_b] = emb[:x_a.shape[0]], emb[x_a.shape[0]:]
                        adata.obsm[f_name] = full_emb
                        benchmark_method_and_update_csv(adata, f_name, METRICS_CSV, CURRENT_SEED)
                        save_method_plot(adata, f_name, RESULT_DIR)
                else:
                    obj = SUPERVISED_CLASSES[m_name](n_components=N_DIM, random_state=CURRENT_SEED)
                    emb = obj.fit_transform(x_a, x_b, y_a_f, y_b_f)
                    full_emb = np.zeros((adata.shape[0], N_DIM))
                    full_emb[idx_a], full_emb[idx_b] = emb[:x_a.shape[0]], emb[x_a.shape[0]:]
                    adata.obsm[m_name] = full_emb
                    benchmark_method_and_update_csv(adata, m_name, METRICS_CSV, CURRENT_SEED)
                    save_method_plot(adata, m_name, RESULT_DIR)

        adata.write(os.path.join(RESULT_DIR, "adata_final.h5ad"))

print(f"\nFinished! Results in: {ROOT_RESULT_DIR}")
