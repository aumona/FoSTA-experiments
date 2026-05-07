import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split

from scib_metrics.benchmark import Benchmarker
import scvi
import pyliger

warnings.filterwarnings("ignore")

# Assuming FoSTA is in the parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.fosta import FoSTA

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "/Users/aumona/Projects/RF-MALI/data_sc/lung_batches.h5ad"
BASE_RESULT_DIR = "/Users/aumona/Projects/RF-MALI/result_sc_ad/"

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"
# BATCH_LIST = ['B2', 'B3', 'B4'] 
# BATCH_LIST = ['B1', 'B2', 'B3', 'B4']
BATCH_LIST = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']

# BATCH_LIST = ['A1', 'A2']

# --- SEED CONFIG ---
SEEDS = [42, 123, 999] 
# SEEDS = [42]  # For quick testing, use a single seed. Change to multiple for full runs.

# --- MASKING CONFIG ---
MASK = True           # Set to True to hide labels during training
MASK_FRACTION = 0.20  # 20% of labels will be hidden
MIN_CELLS_FOR_MASKING = 10  # Rare populations below this count won't be masked

# --- FoSTA SPECIFIC ---
FOSTA_USE_PCA = True  # If True, use top 30 PCs for FoSTA input instead of raw HVGs
FOSTA_PCA_COMPONENTS = 30

# MODELS_TO_RUN = ["scVI", "scANVI", "LIGER", "FoSTA"]
MODELS_TO_RUN = ["FoSTA"]

N_DIM = 2

FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "unlabeled_coupling": "predict_shared", 
        "t": 2,
        "class_weight": "balanced_subsample"
    }
}

# Combine into a single dictionary for reproducibility saving
FULL_CONFIG = {
    "data_path": DATA_PATH,
    "batch_key": BATCH_KEY,
    "label_key": LABEL_KEY,
    "batch_list": BATCH_LIST,
    "masking": MASK,
    "mask_fraction": MASK_FRACTION,
    "min_cells_mask": MIN_CELLS_FOR_MASKING,
    "fosta_use_pca": FOSTA_USE_PCA,
    "seeds": SEEDS,
    "models_run": MODELS_TO_RUN,
    "latent_dims": N_DIM,
    "fosta_configs": FOSTA_CONFIGS,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

# =============================================================================
# HELPERS
# =============================================================================
def stratified_sample_mask(obs_df, label_col, frac, min_cells, seed):
    """Creates a mask index that respects rare populations."""
    counts = obs_df[label_col].value_counts()
    eligible_labels = counts[counts >= (min_cells / frac)].index
    
    mask_indices = []
    for label in eligible_labels:
        label_idx = obs_df[obs_df[label_col] == label].index
        _, masked = train_test_split(label_idx, test_size=frac, random_state=seed)
        mask_indices.extend(masked)
    
    return mask_indices

def save_run_config(config, directory):
    config_path = os.path.join(directory, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {config_path}")

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
    metric_type_ext["Bio conservation mean"] = "Aggregate score"
    metric_type_ext["Batch correction mean"] = "Aggregate score"
    metric_type_ext["Total mean"] = "Aggregate score"
    return pd.concat([score_df, metric_type_ext.to_frame().T.rename(index={0: "Metric Type"})])

def benchmark_method_and_update_csv(adata, method_key, metrics_csv, seed):
    # Determine evaluation set: only masked cells if MASK is True and cells were actually masked
    if MASK and adata.obs["is_masked"].any():
        print(f"Benchmarking {method_key} (Seed {seed}) - MASKED CELLS ONLY...")
        eval_adata = adata[adata.obs["is_masked"]].copy()
    else:
        print(f"Benchmarking {method_key} (Seed {seed}) - FULL DATASET...")
        eval_adata = adata
    
    bm = Benchmarker(
        eval_adata, 
        batch_key=BATCH_KEY, 
        label_key="ground_truth_labels", 
        embedding_obsm_keys=[method_key],
        n_jobs=-1 
    )
    bm.benchmark()
    
    df_res = add_custom_aggregates(bm.get_results(min_max_scale=False))
    metric_df = df_res.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    metric_df.index.name = "Method"
    current = metric_df.loc[[method_key]].copy()
    current["Seed"] = seed 
    
    current.to_csv(metrics_csv, mode='a', header=not os.path.exists(metrics_csv))
    return current

def save_method_plot(adata, method_key, result_dir):
    adata.obsm["X_2d_viz"] = adata.obsm[method_key]
    fig_masked = sc.pl.embedding(adata, basis="X_2d_viz", color=[BATCH_KEY, LABEL_KEY],
                                 show=False, return_fig=True, title=[f"{method_key} Batch", f"{method_key} Masked Labels"])
    fig_masked.savefig(os.path.join(result_dir, f"plot_{method_key}_masked.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_masked)
    
    fig_truth = sc.pl.embedding(adata, basis="X_2d_viz", color=[BATCH_KEY, "ground_truth_labels"],
                                 show=False, return_fig=True, title=[f"{method_key} Batch", f"{method_key} Ground Truth"])
    fig_truth.savefig(os.path.join(result_dir, f"plot_{method_key}_truth.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_truth)

def prepare_fosta_labels(series):
    s = series.astype(str)
    labels = s.values.copy()
    mask_missing = (s == "nan") | (s == "Unknown") | (s == "None")
    labels[mask_missing] = "-1"
    return labels

# =============================================================================
# MAIN LOOP
# =============================================================================
full_adata_orig = sc.read(DATA_PATH)
ROOT_RESULT_DIR = os.path.join(BASE_RESULT_DIR, FULL_CONFIG["timestamp"])
os.makedirs(ROOT_RESULT_DIR, exist_ok=True)

save_run_config(FULL_CONFIG, ROOT_RESULT_DIR)

for CURRENT_SEED in SEEDS:
    print(f"\n\n### STARTING SEED: {CURRENT_SEED} ###")
    np.random.seed(CURRENT_SEED)
    scvi.settings.seed = CURRENT_SEED

    for BATCH_1, BATCH_2 in combinations(BATCH_LIST, 2):
        PAIR_NAME = f"{BATCH_1}_vs_{BATCH_2}"
        RESULT_DIR = os.path.join(ROOT_RESULT_DIR, f"seed_{CURRENT_SEED}", PAIR_NAME)
        os.makedirs(RESULT_DIR, exist_ok=True)
        METRICS_CSV = os.path.join(RESULT_DIR, f"benchmark_metrics_seed{CURRENT_SEED}.csv")
        
        print(f"\n{'='*60}\nRUNNING PAIR: {PAIR_NAME} | SEED: {CURRENT_SEED}\n{'='*60}")
        adata = full_adata_orig[full_adata_orig.obs[BATCH_KEY].isin([BATCH_1, BATCH_2])].copy()
        
        adata.obs["ground_truth_labels"] = adata.obs[LABEL_KEY].copy()
        adata.obs["is_masked"] = False
        
        if MASK:
            if "Unknown" not in adata.obs[LABEL_KEY].cat.categories:
                adata.obs[LABEL_KEY] = adata.obs[LABEL_KEY].cat.add_categories(["Unknown"])
            mask_idx = stratified_sample_mask(adata.obs, "ground_truth_labels", MASK_FRACTION, MIN_CELLS_FOR_MASKING, CURRENT_SEED)
            adata.obs.loc[mask_idx, LABEL_KEY] = "Unknown"
            adata.obs.loc[mask_idx, "is_masked"] = True
        
        adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].cat.remove_unused_categories()
        adata.layers["counts"] = adata.X.copy()

        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key=BATCH_KEY)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.tl.pca(adata, n_comps=30)
        
        # Unintegrated
        adata.obsm["Unintegrated"] = adata.obsm["X_pca"][:, :N_DIM]
        benchmark_method_and_update_csv(adata, "Unintegrated", METRICS_CSV, CURRENT_SEED)
        save_method_plot(adata, "Unintegrated", RESULT_DIR)

        # Standard Models
        if any(m in MODELS_TO_RUN for m in ["scVI", "scANVI"]):
            scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
            vae = scvi.model.SCVI(adata, n_latent=N_DIM)
            vae.train(accelerator="mps", devices=1) 
            
            for m in ["scVI", "scANVI"]:
                if m in MODELS_TO_RUN:
                    if m == "scVI":
                        adata.obsm[m] = vae.get_latent_representation()
                    else:
                        lvae = scvi.model.SCANVI.from_scvi_model(vae, adata=adata, labels_key=LABEL_KEY, unlabeled_category="Unknown")
                        lvae.train(max_epochs=20, n_samples_per_label=100, accelerator="mps", devices=1)
                        adata.obsm[m] = lvae.get_latent_representation()
                    
                    benchmark_method_and_update_csv(adata, m, METRICS_CSV, CURRENT_SEED)
                    save_method_plot(adata, m, RESULT_DIR)

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

        if "FoSTA" in MODELS_TO_RUN:
            idx_a, idx_b = (adata.obs[BATCH_KEY] == BATCH_1), (adata.obs[BATCH_KEY] == BATCH_2)
            
            if FOSTA_USE_PCA:
                x_a = adata[idx_a].obsm["X_pca"][:, :FOSTA_PCA_COMPONENTS]
                x_b = adata[idx_b].obsm["X_pca"][:, :FOSTA_PCA_COMPONENTS]
            else:
                x_a = adata[idx_a].X.toarray() if hasattr(adata[idx_a].X, "toarray") else adata[idx_a].X
                x_b = adata[idx_b].X.toarray() if hasattr(adata[idx_b].X, "toarray") else adata[idx_b].X
            
            y_a_f, y_b_f = prepare_fosta_labels(adata[idx_a].obs[LABEL_KEY]), prepare_fosta_labels(adata[idx_b].obs[LABEL_KEY])

            for f_name, f_params in FOSTA_CONFIGS.items():
                params = {"n_components": N_DIM, "random_state": CURRENT_SEED, "n_jobs": -1}
                params.update(f_params)
                fosta_obj = FoSTA(**params)
                emb = fosta_obj.fit_transform(x_a, x_b, y_a_f, y_b_f)
                full_emb = np.zeros((adata.shape[0], N_DIM))
                full_emb[idx_a], full_emb[idx_b] = emb[:x_a.shape[0]], emb[x_a.shape[0]:]
                adata.obsm[f_name] = full_emb
                benchmark_method_and_update_csv(adata, f_name, METRICS_CSV, CURRENT_SEED)
                save_method_plot(adata, f_name, RESULT_DIR)

        adata.write(os.path.join(RESULT_DIR, f"adata_final.h5ad"))

print(f"\nFinished! Results in: {ROOT_RESULT_DIR}")