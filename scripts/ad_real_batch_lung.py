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

from scib_metrics.benchmark import Benchmarker
import scvi
import pyliger

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.fosta import FoSTA

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "/Users/aumona/Projects/RF-MALI/data_sc/lung_batches.h5ad"
BASE_RESULT_DIR = "/Users/aumona/Projects/RF-MALI/result_sc_ad/"

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"
BATCH_LIST = ["A1", "A2", "A3", "A4", "A5", "A6"] 

MODELS_TO_RUN = ["scVI", "scANVI", "LIGER", "FoSTA"]
# MODELS_TO_RUN = ["FoSTA"]

N_DIM = 2
RANDOM_STATE = 42

# --- NEW: FoSTA Variants Dictionary ---
# Key: Name used in obsm and CSV
# Value: Dict of parameters passed to FoSTA()
FOSTA_CONFIGS = {
    "FoSTA_t2": {
        "unlabeled_coupling": "predict_shared", 
        "t": 2, 
        "class_weight": "balanced_subsample"
    }, # <--- Added comma
    # "FoSTA_auto": {
    #     "unlabeled_coupling": "predict_shared", 
    #     "t": "auto", 
    #     "class_weight": "balanced_subsample"
    # }, # <--- Added closing brace and comma
    "FoSTA_et_t2": {
        "unlabeled_coupling": "predict_shared", 
        "model_type": "et",
        "t": 2, 
        "class_weight": "balanced_subsample"
    }, # <--- Added comma
    # "FoSTA_et_auto": {
    #     "unlabeled_coupling": "predict_shared", 
    #     "model_type": "et",
    #     "t": "auto", 
    #     "class_weight": "balanced_subsample"
    # } 
}

np.random.seed(RANDOM_STATE)
scvi.settings.seed = RANDOM_STATE
ALL_RESULTS = []

# =============================================================================
# HELPERS
# =============================================================================
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

def benchmark_method_and_update_csv(adata, method_key, metrics_csv):
    print(f"Benchmarking {method_key}...")
    bm = Benchmarker(adata, batch_key=BATCH_KEY, label_key=LABEL_KEY, embedding_obsm_keys=[method_key])
    bm.benchmark()
    df_res = add_custom_aggregates(bm.get_results(min_max_scale=False))
    metric_df = df_res.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    metric_df.index.name = "Method"
    current = metric_df.loc[[method_key]]
    current.to_csv(metrics_csv, mode='a', header=not os.path.exists(metrics_csv))
    return current

def save_method_plot(adata, method_key, result_dir, b1, b2):
    adata.obsm["X_2d_viz"] = adata.obsm[method_key]
    # Fixed titles to prevent "title list shorter than panels" warning
    titles = [f"{method_key} Batch", f"{method_key} Cell Type"]
    fig = sc.pl.embedding(adata, basis="X_2d_viz", color=[BATCH_KEY, LABEL_KEY],
                          show=False, return_fig=True, title=titles)
    fig.savefig(os.path.join(result_dir, f"plot_{method_key}.png"), bbox_inches="tight", dpi=300)
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
full_adata = sc.read(DATA_PATH)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for BATCH_1, BATCH_2 in combinations(BATCH_LIST, 2):
    PAIR_NAME = f"{BATCH_1}_vs_{BATCH_2}"
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, timestamp, PAIR_NAME)
    os.makedirs(RESULT_DIR, exist_ok=True)
    METRICS_CSV = os.path.join(RESULT_DIR, "benchmark_metrics_progressive.csv")
    
    print(f"\n{'='*60}\nRUNNING PAIR: {PAIR_NAME}\n{'='*60}")
    adata = full_adata[full_adata.obs[BATCH_KEY].isin([BATCH_1, BATCH_2])].copy()
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].cat.remove_unused_categories()
    adata.layers["counts"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key=BATCH_KEY)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, n_comps=30)
    
    # Unintegrated baseline
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"][:, :N_DIM]
    benchmark_method_and_update_csv(adata, "Unintegrated", METRICS_CSV)
    save_method_plot(adata, "Unintegrated", RESULT_DIR, BATCH_1, BATCH_2)

    # Standard Models
    if any(m in MODELS_TO_RUN for m in ["scVI", "scANVI"]):
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
        vae = scvi.model.SCVI(adata, n_latent=N_DIM)
        vae.train()
        for m in ["scVI", "scANVI"]:
            if m in MODELS_TO_RUN:
                if m == "scVI":
                    adata.obsm[m] = vae.get_latent_representation()
                else:
                    lvae = scvi.model.SCANVI.from_scvi_model(vae, adata=adata, labels_key=LABEL_KEY, unlabeled_category="Unknown")
                    lvae.train(max_epochs=20)
                    adata.obsm[m] = lvae.get_latent_representation()
                benchmark_method_and_update_csv(adata, m, METRICS_CSV)
                save_method_plot(adata, m, RESULT_DIR, BATCH_1, BATCH_2)

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
        pyliger.optimize_ALS(liger_data, k=N_DIM, rand_seed=RANDOM_STATE)
        pyliger.quantile_norm(liger_data)
        liger_res = np.zeros((adata.shape[0], N_DIM))
        for i, b in enumerate(batch_cats):
            liger_res[adata.obs[BATCH_KEY] == b] = liger_data.adata_list[i].obsm["H_norm"]
        adata.obsm["LIGER"] = liger_res
        benchmark_method_and_update_csv(adata, "LIGER", METRICS_CSV)
        save_method_plot(adata, "LIGER", RESULT_DIR, BATCH_1, BATCH_2)

    # --- UPDATED: Multi-FoSTA Run ---
    if "FoSTA" in MODELS_TO_RUN:
        idx_a, idx_b = (adata.obs[BATCH_KEY] == BATCH_1), (adata.obs[BATCH_KEY] == BATCH_2)
        x_a = adata[idx_a].X.toarray() if hasattr(adata[idx_a].X, "toarray") else adata[idx_a].X
        x_b = adata[idx_b].X.toarray() if hasattr(adata[idx_b].X, "toarray") else adata[idx_b].X
        y_a_f, y_b_f = prepare_fosta_labels(adata[idx_a].obs[LABEL_KEY]), prepare_fosta_labels(adata[idx_b].obs[LABEL_KEY])

        for f_name, f_params in FOSTA_CONFIGS.items():
            print(f"\nRunning FoSTA variant: {f_name}")
            # Merge static params with config dict
            params = {
                "n_components": N_DIM, 
                "random_state": RANDOM_STATE, 
                "n_jobs": -1, 
                "verbose": 1
            }
            params.update(f_params)
            
            fosta_obj = FoSTA(**params)
            emb = fosta_obj.fit_transform(x_a, x_b, y_a_f, y_b_f)
            
            full_emb = np.zeros((adata.shape[0], N_DIM))
            full_emb[idx_a], full_emb[idx_b] = emb[:x_a.shape[0]], emb[x_a.shape[0]:]
            
            adata.obsm[f_name] = full_emb
            benchmark_method_and_update_csv(adata, f_name, METRICS_CSV)
            save_method_plot(adata, f_name, RESULT_DIR, BATCH_1, BATCH_2)

    adata.write(os.path.join(RESULT_DIR, f"adata_{PAIR_NAME}_final.h5ad"))

print(f"\nFinished! Results in: {os.path.join(BASE_RESULT_DIR, timestamp)}")