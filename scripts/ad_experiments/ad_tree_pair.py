'''
Ground-truth tree = gen_tree(seed=seed, sigma=0)
Batch A = gen_tree(seed=seed, sigma=4) with p_a% dropout
Batch B = gen_tree(seed=seed+seed_offset, sigma=2) with p_b% dropout
For each batch, some % of labels are masked to simulate partial supervision.
We then run various methods to integrate Batch A and Batch B, and evaluate how well they recover
the structure of the ground-truth tree using DeMAP and FOSCTTM metrics.
'''
import importlib.util
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
np.int = int
import pandas as pd
import phate
import scanorama
import scanpy as sc
import scvi
import pyliger
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.fosta import FoSTA
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf
from src.mali import MALI
from src.Pamona.eval import test_alignment_score, test_transfer_accuracy, calc_domainAveraged_FOSCTTM
from src.pamona import Pamona
from utils.tree_utils import gen_tree
from ad_experiment_utils import (
    append_result_row,
    coerce_embedding_array,
    save_embedding_plots,
    save_embeddings,
    write_json,
)


def load_official_demap_metric():
    for sys_path in sys.path:
        candidate = Path(sys_path) / "demap" / "demap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("official_demap_metric", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.DEMaP
    raise ImportError("Could not locate the installed official demap.demap module.")


DEMaP = load_official_demap_metric()


# =============================================================================
# CONFIG
# =============================================================================
TREE_PARAMS = dict(
    n_branch=10,
    n_child=2,
    n_dim_per_branch=4,
    branch_length=100,
    merged_branch=False,
)

SEEDS = [11784, 39041, 56089, 79121, 4386721]

# SEEDS = [39041, 56089, 79121]
# SEEDS = [39041]

GROUND_TRUTH_SIGMA = 0
BATCH_A_SIGMA = 2
BATCH_A_DROPOUT_LEVEL = 0.20
BATCH_B_SIGMA = 5
BATCH_B_DROPOUT_LEVEL = 0.50
BATCH_B_SEED_OFFSET = 100_000
LABEL_MASKING_LEVEL_B = 0.50
N_COMPONENTS = 2
DEMAP_KNN = 30   # default value used in the original DeMAP code

MODELS_TO_RUN = [
    "scVI",
    "scANVI",
    "LIGER",
    "Scanorama",
    "FoSTA",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    "Pamona",
]
BASELINES_TO_RUN = {"Unintegrated", "Unintegrated_PHATE"}

FOSTA_CONFIGS = {
    "FoSTA_tauto": {
        "t": 'auto',
    }
}

SUPERVISED_CLASSES = {
    "FoSTA": FoSTA,
    "KEMAlin": KEMAlin,
    "KEMArbf": KEMArbf,
    "MALI": MALI,
    "Pamona": Pamona,
}

RESULTS_ROOT = PROJECT_ROOT / "results_tree"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================
def validate_probability(name, value):
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def validate_config():
    validate_probability("BATCH_A_DROPOUT_LEVEL", BATCH_A_DROPOUT_LEVEL)
    validate_probability("BATCH_B_DROPOUT_LEVEL", BATCH_B_DROPOUT_LEVEL)
    validate_probability("LABEL_MASKING_LEVEL_B", LABEL_MASKING_LEVEL_B)


def save_experiment_metadata(output_dir, timestamp):
    metadata = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp": timestamp,
        "tree_params": TREE_PARAMS,
        "seeds": SEEDS,
        "perturbation_settings": {
            "ground_truth_sigma": GROUND_TRUTH_SIGMA,
            "batch_a_sigma": BATCH_A_SIGMA,
            "batch_a_dropout_level": BATCH_A_DROPOUT_LEVEL,
            "batch_b_sigma": BATCH_B_SIGMA,
            "batch_b_dropout_level": BATCH_B_DROPOUT_LEVEL,
            "batch_b_seed_offset": BATCH_B_SEED_OFFSET,
            "label_masking_level_b": LABEL_MASKING_LEVEL_B,
        },
        "model_settings": {
            "n_components": N_COMPONENTS,
            "demap_knn": DEMAP_KNN,
            "baselines_to_run": sorted(BASELINES_TO_RUN),
            "models_to_run": MODELS_TO_RUN,
            "fosta_configs": FOSTA_CONFIGS,
            "supervised_methods": sorted(SUPERVISED_CLASSES),
        },
    }
    write_json(output_dir / "experiment_metadata.json", metadata)


def minmax_normalize(x):
    return MinMaxScaler().fit_transform(np.asarray(x, dtype=float))


def apply_dropout(x, dropout_prob, seed):
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(x.shape) >= dropout_prob
    zero_rows = np.where(~keep_mask.any(axis=1))[0]
    if zero_rows.size > 0:
        rescue_cols = rng.integers(0, x.shape[1], size=zero_rows.size)
        keep_mask[zero_rows, rescue_cols] = True
    return np.asarray(x, dtype=float) * keep_mask.astype(float)


def make_stratified_mask(labels, frac, seed):
    labels = np.asarray(labels).astype(str)
    if frac <= 0:
        return np.zeros(labels.shape[0], dtype=bool)

    rng = np.random.default_rng(seed)
    mask = np.zeros(labels.shape[0], dtype=bool)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        n_mask = int(np.floor(idx.size * frac))
        if n_mask <= 0:
            continue
        chosen = rng.choice(idx, size=n_mask, replace=False)
        mask[chosen] = True
    return mask


def build_pair(seed):
    ground_truth_tree, labels, _ = gen_tree(seed=seed, sigma=GROUND_TRUTH_SIGMA, **TREE_PARAMS)
    batch_a_tree, labels_a, _ = gen_tree(seed=seed, sigma=BATCH_A_SIGMA, **TREE_PARAMS)
    batch_b_tree, labels_b, _ = gen_tree(seed=seed + BATCH_B_SEED_OFFSET, sigma=BATCH_B_SIGMA, **TREE_PARAMS)

    ground_truth_tree = minmax_normalize(ground_truth_tree)
    batch_a_tree = apply_dropout(minmax_normalize(batch_a_tree), BATCH_A_DROPOUT_LEVEL, seed + 11)
    batch_b_tree = apply_dropout(minmax_normalize(batch_b_tree), BATCH_B_DROPOUT_LEVEL, seed + 23)

    if not np.array_equal(labels, labels_a) or not np.array_equal(labels, labels_b):
        raise ValueError("Generated batch labels do not match ground-truth labels.")

    target_mask = make_stratified_mask(labels, LABEL_MASKING_LEVEL_B, seed + 17)

    target_labels_obs = np.asarray(labels).astype(int).copy()
    target_labels_obs[target_mask] = -1

    scanvi_labels_a = np.asarray(labels).astype(str).copy()
    scanvi_labels_obs = np.asarray(labels).astype(str).copy()
    scanvi_labels_obs[target_mask] = "Unknown"

    return (
        ground_truth_tree,
        batch_a_tree,
        batch_b_tree,
        np.asarray(labels).astype(int),
        target_labels_obs,
        scanvi_labels_a,
        scanvi_labels_obs,
    )


def build_adata(clean_tree, noisy_tree, labels_a, labels_b_obs, scanvi_labels_a, scanvi_labels_b):
    x = np.vstack([clean_tree, noisy_tree])
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(
                ["A"] * len(clean_tree) + ["B"] * len(noisy_tree),
                categories=["A", "B"],
            ),
            "ground_truth_labels": np.concatenate([labels_a, labels_a]),
            "observed_labels": np.concatenate([labels_a, labels_b_obs]),
            "scanvi_labels": np.concatenate([scanvi_labels_a, scanvi_labels_b]),
            "pair_id": np.concatenate([np.arange(len(clean_tree)), np.arange(len(noisy_tree))]),
        }
    , index=[f"cell_{i}" for i in range(x.shape[0])]
    )
    var = pd.DataFrame(index=[f"dim_{i}" for i in range(x.shape[1])])
    obs.index.name = "cell"
    var.index.name = "gene"
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.obs_names = obs.index.astype(str)
    adata.var_names = var.index.astype(str)
    adata.layers["counts"] = x.copy()
    return adata

def make_plot_specs(labels_a, n_a, n_total):
    labels = np.concatenate([labels_a, labels_a]).astype(str)
    batches = np.array(["A"] * n_a + ["B"] * (n_total - n_a))
    return [
        ("labels", labels, "colorblind", "Ground Truth Label"),
        ("batch", batches, "tab10", "Batch"),
    ]


def benchmark_method(method_name, embedding, ground_truth_tree, labels_a, labels_b_obs, output_dir):
    emb = coerce_embedding_array(embedding)
    if emb.shape[1] > 2:
        emb = emb[:, :2]

    n_a = len(labels_a)
    emb_a = emb[:n_a]
    emb_b = emb[n_a:]

    duplicated_clean_tree = np.vstack([ground_truth_tree, ground_truth_tree])
    demap = DEMaP(duplicated_clean_tree, emb, knn=DEMAP_KNN)

    foscttm_vals = calc_domainAveraged_FOSCTTM(emb_a, emb_b)
    foscttm = float(np.mean(foscttm_vals))
    alignment_score = test_alignment_score(emb_a, emb_b)
    mask_missing_target = np.asarray(labels_b_obs) == -1
    if np.any(mask_missing_target):
        label_transfer = test_transfer_accuracy(
            data1=emb_b[mask_missing_target],
            data2=emb_a,
            type1=np.asarray(labels_a)[mask_missing_target],
            type2=np.asarray(labels_a),
        )
    else:
        label_transfer = np.nan

    method_dir = output_dir / method_name
    save_embeddings(method_dir, method_name, emb, n_a)
    save_embedding_plots(method_dir, method_name, emb, make_plot_specs(labels_a, n_a, emb.shape[0]))

    return {
        "method": method_name,
        "DeMAP": float(demap),
        "FOSCTTM": foscttm,
        "alignment_score": float(alignment_score),
        "label_transfer": float(label_transfer) if not np.isnan(label_transfer) else np.nan,
    }


def save_pair_inputs(output_dir, ground_truth_tree, batch_a_tree, batch_b_tree, labels_a, labels_b_obs):
    np.savez_compressed(
        output_dir / "paired_tree_inputs.npz",
        ground_truth_tree=ground_truth_tree,
        batch_a_tree=batch_a_tree,
        batch_b_tree=batch_b_tree,
        labels_a=labels_a,
        labels_b_obs=labels_b_obs,
    )


# =============================================================================
# METHODS
# =============================================================================
def run_unintegrated_pca(adata):
    sc.tl.pca(adata, n_comps=N_COMPONENTS)
    return adata.obsm["X_pca"].copy()


def run_unintegrated_phate(adata, seed):
    return phate.PHATE(n_components=N_COMPONENTS, random_state=seed).fit_transform(adata.X)


def run_scvi(adata, seed):
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
    model = scvi.model.SCVI(adata, n_latent=N_COMPONENTS)
    model.train(max_epochs=50)
    return model.get_latent_representation()


def run_scanvi(adata, seed):
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch", labels_key="scanvi_labels")
    scvi_model = scvi.model.SCVI(adata, n_latent=N_COMPONENTS)
    scvi_model.train(max_epochs=50)
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key="scanvi_labels",
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)
    return scanvi_model.get_latent_representation()


def run_liger(adata):
    bdata = adata.copy()
    bdata.X = sparse.csr_matrix(bdata.layers["counts"])
    batch_cats = list(bdata.obs["batch"].cat.categories)
    adata_list = [bdata[bdata.obs["batch"] == batch].copy() for batch in batch_cats]

    for i, sub_adata in enumerate(adata_list):
        sub_adata.uns["sample_name"] = batch_cats[i]
        sub_adata.uns["var_gene_idx"] = np.arange(bdata.n_vars)

        if not sparse.issparse(sub_adata.X):
            sub_adata.X = sparse.csr_matrix(sub_adata.X)

    liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
    liger_data.var_genes = bdata.var_names
    pyliger.normalize(liger_data)
    with np.errstate(divide="ignore", invalid="ignore"):
        pyliger.scale_not_center(liger_data)
    pyliger.optimize_ALS(liger_data, k=N_COMPONENTS)
    pyliger.quantile_norm(liger_data)

    result = np.zeros((adata.shape[0], N_COMPONENTS))
    for i, batch in enumerate(batch_cats):
        result[adata.obs["batch"] == batch] = liger_data.adata_list[i].obsm["H_norm"][:, :N_COMPONENTS]
    return result


def run_scanorama(adata):
    batch_cats = list(adata.obs["batch"].cat.categories)
    adata_list = [adata[adata.obs["batch"] == batch].copy() for batch in batch_cats]

    for sub_adata in adata_list:
        sc.pp.normalize_total(sub_adata, target_sum=1e4)
        sc.pp.log1p(sub_adata)

    scanorama.integrate_scanpy(adata_list)
    scan_dim = adata_list[0].obsm["X_scanorama"].shape[1]
    result = np.zeros((adata.shape[0], scan_dim))
    for i, batch in enumerate(batch_cats):
        result[adata.obs["batch"] == batch] = adata_list[i].obsm["X_scanorama"]
    return result[:, :N_COMPONENTS]


def run_supervised_method(method_name, x_a, x_b, y_a, y_b, seed):
    if method_name == "FoSTA":
        for f_name, f_params in FOSTA_CONFIGS.items():
            model = SUPERVISED_CLASSES[method_name](n_components=N_COMPONENTS, random_state=seed, **f_params)
            return f_name, model.fit_transform(x_a, x_b, y_a, y_b)

    model = SUPERVISED_CLASSES[method_name](n_components=N_COMPONENTS, random_state=seed)
    return method_name, model.fit_transform(x_a, x_b, y_a, y_b)


# =============================================================================
# MAIN
# =============================================================================
def main():
    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = RESULTS_ROOT / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_metadata(root_dir, timestamp)
    results_csv = root_dir / "tree_alignment_results.csv"

    all_rows = []

    for seed in SEEDS:
        print(f"\n### STARTING SEED: {seed} ###")
        (
            ground_truth_tree,
            batch_a_tree,
            batch_b_tree,
            labels_a,
            labels_b_obs,
            scanvi_labels_a,
            scanvi_labels_b,
        ) = build_pair(seed)

        seed_dir = root_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_pair_inputs(seed_dir, ground_truth_tree, batch_a_tree, batch_b_tree, labels_a, labels_b_obs)

        adata = build_adata(
            batch_a_tree,
            batch_b_tree,
            labels_a,
            labels_b_obs,
            scanvi_labels_a,
            scanvi_labels_b,
        )

        x_a = batch_a_tree
        x_b = batch_b_tree
        y_a = labels_a
        y_b = labels_b_obs

        method_runs = [
            ("Unintegrated", lambda: run_unintegrated_pca(adata.copy())),
            ("Unintegrated_PHATE", lambda: run_unintegrated_phate(adata.copy(), seed)),
            ("scVI", lambda: run_scvi(adata.copy(), seed)),
            ("scANVI", lambda: run_scanvi(adata.copy(), seed)),
            ("LIGER", lambda: run_liger(adata.copy())),
            ("Scanorama", lambda: run_scanorama(adata.copy())),
            ("FoSTA", lambda: run_supervised_method("FoSTA", x_a, x_b, y_a, y_b, seed)),
            ("KEMAlin", lambda: run_supervised_method("KEMAlin", x_a, x_b, y_a, y_b, seed)),
            ("KEMArbf", lambda: run_supervised_method("KEMArbf", x_a, x_b, y_a, y_b, seed)),
            ("MALI", lambda: run_supervised_method("MALI", x_a, x_b, y_a, y_b, seed)),
            ("Pamona", lambda: run_supervised_method("Pamona", x_a, x_b, y_a, y_b, seed)),
        ]

        for method_name, runner in method_runs:
            if method_name not in BASELINES_TO_RUN and method_name not in MODELS_TO_RUN:
                continue

            start = time.perf_counter()
            try:
                print(f"Running {method_name}...")
                result = runner()

                if method_name in SUPERVISED_CLASSES or method_name == "FoSTA":
                    out_name, embedding = result
                else:
                    out_name, embedding = method_name, result

                metrics = benchmark_method(out_name, embedding, ground_truth_tree, labels_a, labels_b_obs, seed_dir)
                metrics["seed"] = seed
                metrics["runtime_sec"] = float(time.perf_counter() - start)
                metrics["status"] = "ok"
                all_rows.append(metrics)
                append_result_row(results_csv, metrics)
                print(
                    f"  DeMAP={metrics['DeMAP']:.4f} | "
                    f"Acc={metrics['label_transfer']:.4f} | "
                    f"AS={metrics['alignment_score']:.4f} | "
                    f"FOSCTTM={metrics['FOSCTTM']:.4f} | "
                    f"{metrics['runtime_sec']:.1f}s"
                )
            except Exception as exc:
                metrics = {
                    "method": next(iter(FOSTA_CONFIGS)) if method_name == "FoSTA" else method_name,
                    "DeMAP": np.nan,
                    "label_transfer": np.nan,
                    "alignment_score": np.nan,
                    "FOSCTTM": np.nan,
                    "seed": seed,
                    "runtime_sec": float(time.perf_counter() - start),
                    "status": f"error: {exc}",
                }
                all_rows.append(metrics)
                append_result_row(results_csv, metrics)
                print(f"  FAILED: {exc}")

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(["seed", "method"], kind="stable")
    results_df.to_csv(results_csv, index=False)

    print(f"\nFinished. Results saved to: {root_dir}")
    print(f"Metrics CSV: {results_csv}")


if __name__ == "__main__":
    main()
