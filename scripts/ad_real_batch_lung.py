import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

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

BATCH_1 = "A1"
BATCH_2 = "A2"

# MODELS_TO_RUN = ["FoSTA"]
MODELS_TO_RUN = ["scVI", "scANVI", "LIGER", "FoSTA"]

N_DIM = 2
RANDOM_STATE = 0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PAIR_NAME = f"{BATCH_1}_vs_{BATCH_2}"
RESULT_DIR = os.path.join(BASE_RESULT_DIR, f"{timestamp}_{PAIR_NAME}")
os.makedirs(RESULT_DIR, exist_ok=True)

METRICS_CSV = os.path.join(RESULT_DIR, "benchmark_metrics_progressive.csv")


# =============================================================================
# HELPERS
# =============================================================================
ALL_RESULTS = []


def add_custom_aggregates(df_res):
    df = df_res.copy()

    if "Metric Type" not in df.index:
        return df

    metric_type = df.loc["Metric Type"]
    score_df = df.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")

    bio_cols = metric_type.index[metric_type == "Bio conservation"].tolist()
    batch_cols = metric_type.index[metric_type == "Batch correction"].tolist()

    exclude = {"Bio conservation", "Batch correction", "Total"}
    bio_cols = [c for c in bio_cols if c not in exclude]
    batch_cols = [c for c in batch_cols if c not in exclude]

    score_df["Bio conservation mean"] = score_df[bio_cols].mean(axis=1)
    score_df["Batch correction mean"] = score_df[batch_cols].mean(axis=1)
    score_df["Total mean"] = (
        0.6 * score_df["Bio conservation mean"]
        + 0.4 * score_df["Batch correction mean"]
    )

    metric_type_ext = metric_type.copy()
    metric_type_ext["Bio conservation mean"] = "Aggregate score"
    metric_type_ext["Batch correction mean"] = "Aggregate score"
    metric_type_ext["Total mean"] = "Aggregate score"

    return pd.concat([
        score_df,
        metric_type_ext.to_frame().T.rename(index={0: "Metric Type"}),
    ])


def benchmark_method_and_update_csv(adata, method_key):
    print(f"\nBenchmarking {method_key}...")

    bm = Benchmarker(
        adata,
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,
        embedding_obsm_keys=[method_key],
    )
    bm.benchmark()

    df_res = bm.get_results(min_max_scale=False)
    df_res = add_custom_aggregates(df_res)

    metric_df = df_res.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    metric_df.index.name = "Method"

    current = metric_df.loc[[method_key]]
    ALL_RESULTS.append(current)

    cumulative = pd.concat(ALL_RESULTS, axis=0)
    cumulative.to_csv(METRICS_CSV)

    print(f"Updated metrics: {METRICS_CSV}")

    return current


def save_method_plot(adata, method_key):
    adata.obsm["X_2d_viz"] = adata.obsm[method_key]

    fig = sc.pl.embedding(
        adata,
        basis="X_2d_viz",
        color=[BATCH_KEY, LABEL_KEY],
        show=False,
        return_fig=True,
        title=f"{method_key} ({BATCH_1} vs {BATCH_2})",
    )

    out_png = os.path.join(RESULT_DIR, f"plot_{method_key}.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved plot: {out_png}")


def save_outputs_for_method(adata, method_key):
    benchmark_method_and_update_csv(adata, method_key)
    save_method_plot(adata, method_key)


def prepare_fosta_labels(series):
    s = series.astype(str)
    labels = s.values.copy()
    mask_missing = (s == "nan") | (s == "Unknown") | (s == "None")
    labels[mask_missing] = "-1"
    return labels


# =============================================================================
# LOAD AND SUBSET
# =============================================================================
adata = sc.read(DATA_PATH)

print("Available batches:", list(adata.obs[BATCH_KEY].cat.categories))

adata = adata[adata.obs[BATCH_KEY].isin([BATCH_1, BATCH_2])].copy()
adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].cat.remove_unused_categories()

if adata.n_obs == 0:
    raise ValueError(f"No cells found for selected batches: {BATCH_1}, {BATCH_2}")

batch_cats = list(adata.obs[BATCH_KEY].cat.categories)

print("Selected batches:", batch_cats)
print(adata.obs[BATCH_KEY].value_counts())

adata.layers["counts"] = adata.X.copy()

config = {
    "data_path": DATA_PATH,
    "result_dir": RESULT_DIR,
    "batch_key": BATCH_KEY,
    "label_key": LABEL_KEY,
    "batch_1": BATCH_1,
    "batch_2": BATCH_2,
    "models_to_run": MODELS_TO_RUN,
    "n_dim": N_DIM,
    "random_state": RANDOM_STATE,
}

with open(os.path.join(RESULT_DIR, "run_config.json"), "w") as f:
    json.dump(config, f, indent=2)


# =============================================================================
# PREPROCESS
# =============================================================================
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="cell_ranger",
    batch_key=BATCH_KEY,
)

adata = adata[:, adata.var.highly_variable].copy()

sc.tl.pca(adata, n_comps=30)
adata.obsm["Unintegrated"] = adata.obsm["X_pca"][:, :N_DIM]

save_outputs_for_method(adata, "Unintegrated")


# =============================================================================
# RUN MODELS
# =============================================================================
if any(m in MODELS_TO_RUN for m in ["scVI", "scANVI"]):
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=BATCH_KEY,
    )

    vae = scvi.model.SCVI(
        adata,
        n_latent=N_DIM,
    )
    vae.train()

    if "scVI" in MODELS_TO_RUN:
        adata.obsm["scVI"] = vae.get_latent_representation()
        save_outputs_for_method(adata, "scVI")

    if "scANVI" in MODELS_TO_RUN:
        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata,
            labels_key=LABEL_KEY,
            unlabeled_category="Unknown",
        )
        lvae.train(max_epochs=20)

        adata.obsm["scANVI"] = lvae.get_latent_representation()
        save_outputs_for_method(adata, "scANVI")


if "LIGER" in MODELS_TO_RUN:
    bdata = adata.copy()
    bdata.X = bdata.layers["counts"]

    adata_list = [
        bdata[bdata.obs[BATCH_KEY] == b].copy()
        for b in batch_cats
    ]

    for i, ad in enumerate(adata_list):
        ad.uns["sample_name"] = batch_cats[i]
        ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)

    liger_data = pyliger.create_liger(
        adata_list,
        remove_missing=False,
        make_sparse=False,
    )
    liger_data.var_genes = bdata.var_names

    pyliger.normalize(liger_data)

    with np.errstate(divide="ignore", invalid="ignore"):
        pyliger.scale_not_center(liger_data)

    pyliger.optimize_ALS(liger_data, k=N_DIM)
    pyliger.quantile_norm(liger_data)

    liger_res = np.zeros((adata.shape[0], N_DIM))

    for i, b in enumerate(batch_cats):
        liger_res[adata.obs[BATCH_KEY] == b] = liger_data.adata_list[i].obsm["H_norm"]

    adata.obsm["LIGER"] = liger_res
    save_outputs_for_method(adata, "LIGER")


if "FoSTA" in MODELS_TO_RUN:
    idx_a = adata.obs[BATCH_KEY] == BATCH_1
    idx_b = adata.obs[BATCH_KEY] == BATCH_2

    x_a = adata[idx_a].X.toarray() if hasattr(adata[idx_a].X, "toarray") else adata[idx_a].X
    x_b = adata[idx_b].X.toarray() if hasattr(adata[idx_b].X, "toarray") else adata[idx_b].X

    y_a_fosta = prepare_fosta_labels(adata[idx_a].obs[LABEL_KEY])
    y_b_fosta = prepare_fosta_labels(adata[idx_b].obs[LABEL_KEY])

    fosta = FoSTA(
        ot_solver="hiref",
        unlabeled_coupling="predict_shared",
        t=2,
        class_weight="balanced_subsample",
        n_components=N_DIM,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    emb = fosta.fit_transform(
        x_a,
        x_b,
        y_a_fosta,
        y_b_fosta,
    )

    full_emb = np.zeros((adata.shape[0], N_DIM))
    full_emb[idx_a] = emb[:x_a.shape[0]]
    full_emb[idx_b] = emb[x_a.shape[0]:]

    adata.obsm["FoSTA"] = full_emb
    save_outputs_for_method(adata, "FoSTA")


# =============================================================================
# FINAL SAVE
# =============================================================================
adata.write(os.path.join(RESULT_DIR, f"adata_{PAIR_NAME}_with_embeddings.h5ad"))

print(f"\nAll results saved to: {RESULT_DIR}")
print(f"Progressive metrics saved to: {METRICS_CSV}")