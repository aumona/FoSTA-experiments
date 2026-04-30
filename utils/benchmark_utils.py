import os
import tracemalloc
import random
import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, silhouette_score
from utils.metrics import label_transfer_accuracy, calc_domainAveraged_FOSCTTM
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc

# Alignment models
from src.mali import MALI
from src.rfmali import RFMALI
from src.fosta import FoSTA
from src.kemalin import KEMAlin
from src.kemarbf import KEMArbf
from src.pamona_joint import JPamona
from src.pamona import Pamona

import scvi
from scvi.model import SCVI
from scvi.model import SCANVI
import scanorama
import pyliger
import harmonypy

import time

import scib
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import pickle

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
   

def ablation_fosta( method_name = None,
                        params_dict = None,
                        x_source = None, x_target = None, y_source= None, y_target = None, 
                        seed=42,
                        embedder = "PHATE",
                        n_components = 2):

    method_name = params_dict.pop("method_name", "fosta")
    if method_name == "RFMALI":
        model = RFMALI(
            **params_dict,
            embedder=embedder,
            n_components=n_components,
            random_state=seed
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    else:
        model = FoSTA(**params_dict,
                    embedder=embedder,
                    n_components=n_components,
                    random_state=seed
                    )
    
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    
    return embedding

def run_our_models(model_name=None, x_source = None, x_target = None, y_source= None, y_target = None, 
                   embedder = "PHATE", 
                   seed=42, 
                   gamma = 0.5, 
                   mu = 0.5, 
                   t="auto", 
                   n_components = 2,
                   **fosta_params):
    # this function works for MALI, RF-MALI, KEMA, Pamona
    # input: source and target datasets (x_source, x_target) and their labels (y_source, y_target)
    # output: embedding of shape (n_source + n_target, n_components (set to 2))
    
    if model_name == "RFMALI":
        model = RFMALI(
            embedder=embedder,
            n_components=n_components,
            random_state=seed,
            n_jobs=-1,
            t = t,
            verbose=1
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    
    elif model_name == "RFMALI_WIP" or model_name == "FoSTA":
        model = FoSTA(
            embedder=embedder,
            n_components=n_components,
            random_state=seed,
            t=t,
            n_jobs=-1,
            verbose=1,
            **fosta_params
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    elif model_name == 'MALI':
        model = MALI(embedder=embedder,
                    n_components=n_components,
                    verbose=1,
                    t=t,
                    random_state=seed)
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
        
    elif model_name == 'KEMArbf':
        model = KEMArbf(
            n_components=n_components,
            mu=mu,
            random_state=seed,
            unlabeled_value=-1
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    elif model_name == 'KEMAlin':
        model = KEMAlin(
            n_components=n_components,
            mu=mu,
            random_state=seed,
            unlabeled_value=-1
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    elif model_name == 'Pamona':
        model = Pamona(
            n_components=n_components,
            embedder="UMAP",
            gamma=gamma,  # controls supervision strength, 1=fully supervised
            random_state=seed
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    elif model_name == 'JPamona':
        model = JPamona(
            n_components=n_components,
            embedder="UMAP",
            mu=mu,
            gamma=gamma,  # controls supervision strength, 1=fully supervised
            random_state=seed
        )
        print("\nStarting alignment...")
        embedding = model.fit_transform(x_source, x_target, y_source, y_target)
        print("Alignment complete.")
    
    else:
        # print(f"Model {model_name} not recognized.")
        raise ValueError(f"Model {model_name} not recognized.")
        
    return embedding


def get_inputs_from_adatas(adata1, adata2, label_key="cell_type", embedding_basis="X"):
    x0 = csr_matrix(adata1.obsm[embedding_basis])
    x1 = csr_matrix(adata2.obsm[embedding_basis])
    
    # x0 = np.array(adata1.obsm[embedding_basis])
    # x1 = np.array(adata2.obsm[embedding_basis])
    
    y0 = np.array(adata1.obs[label_key])
    y1 = np.array(adata2.obs[label_key])
                
    # keep index so we can reorder the results of our methods
    idx_d0 = adata1.obs.index
    idx_d1 = adata2.obs.index
    idxs_d = np.concatenate((idx_d0, idx_d1))
    return x0, x1, y0, y1, idxs_d
    
def split_adata(adata, batch_key="batch", **kwargs):
    batches_names = adata.obs[batch_key].unique().tolist()
    
    adata1 = adata[adata.obs[batch_key] == batches_names[0]]
    adata2 = adata[adata.obs[batch_key] == batches_names[1]]
    
    return get_inputs_from_adatas(adata1, adata2, **kwargs)



def run_our_models_from_adata(adata, model_name= None, batch_key = "batch", label_key = "cell_type", embedding_basis="X", seed = 42, **kwargs):
    # for some methods (MALI, RF-MALI, KEMA, Pamona), the expected input is two datasets (source and target) and two labels
    # this function allows to extract these from adata object given the batch_key and label_key
    # the embedding is then stored in adata.obsm[model_name]
    # return adata with embedding added

    x0, x1, y0, y1, idxs_d = split_adata(adata, batch_key=batch_key, label_key=label_key, embedding_basis=embedding_basis)
   
    embedding = try_run_our_models(x0=x0, x1=x1, y0=y0, y1=y1, model_name=model_name, seed=seed, **kwargs)
    
    # put embedding in adata (after reordering rows of the embedding to match adata.obs.index)
    emb = pd.DataFrame(embedding)
    emb = emb.set_index(idxs_d)  
    emb = emb.reindex(adata.obs.index)
    adata.obsm[model_name] = np.asarray(emb)
    return adata

def try_run_our_models(x0=None, x1=None, y0=None, y1=None, model_name=None, seed=None, **kwargs):
    try:
        embedding = run_our_models(model_name, x0, x1, y0, y1, seed = seed, **kwargs)
        
    except Exception as e:
        x0 = x0.toarray()
        x1 = x1.toarray()
        print("(!) Sparse matrix conversion to dense due to error:", e)
        
        # try:
        embedding = run_our_models(model_name, x0, x1, y0, y1, seed = seed, **kwargs)
        # except Exception as e2:
        #     print(f"(!) Model {model_name} failed even after sparse to dense conversion:", e2)
        #     embedding = np.zeros((x0.shape[0] + x1.shape[0],2))  # Dummy embedding to avoid crashes in downstream code
            
    return embedding


def run_our_models_from_adatas(adata1, adata2, label_key="cell_type", model_name=None, embedding_basis="X", seed=42, **kwargs):
    x0, x1, y0, y1, idxs_d = get_inputs_from_adatas(adata1, adata2, label_key=label_key, embedding_basis=embedding_basis)
    embedding = try_run_our_models(x0=x0, x1=x1, y0=y0, y1=y1, model_name=model_name, seed=seed, **kwargs)
    
    # how to get the 
    
    return embedding, y0, y1, idxs_d



def run_models_from_adata(adata, model_name, batch_key = "batch", label_key_ours = None, label_key = "cell_type", n_components = 30, seed = 42, embedding_basis="X", **kwargs):
    # kwargs are passed to our methods only
    # runs either our methods or other methods depending on model_name
    # returns adata with embedding in adata.obsm[model_name]
    # label_key_ours is used for our methods only (if different from label_key)
    
    set_seeds(seed) # reset seeds for reproducible results (so not affected by previous operations, each method starts fresh)

    start_time = time.time()
    tracemalloc.start()
    current_mem_start, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    
    if model_name.lower() in ["rfmali", "rfmali_wip", "mali", "pamona", "kemarbf", "kemalin", "fosta"]:
        label_key_to_use = label_key_ours if label_key_ours is not None else label_key
        adata = run_our_models_from_adata(adata, model_name= model_name, batch_key = batch_key, label_key = label_key_to_use, embedding_basis=embedding_basis, n_components = n_components, seed= seed, **kwargs)
    
    else: # OTHER METHODS
        
        # List of adata per batch
        batch_cats = adata.obs[batch_key].astype('category').cat.categories
        adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]
         
        # ------------------ run SCANORAMA --------------------------------
        if model_name.lower() == "scanorama":
            scanorama.integrate_scanpy(adata_list, dimred = n_components)

            adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
            for i, b in enumerate(batch_cats):
                adata.obsm["Scanorama"][adata.obs[batch_key] == b] = adata_list[i].obsm["X_scanorama"]
                
            
        # ------------------ run LIGER --------------------------------
        elif model_name.lower() == "liger": 
        
            bdata = adata.copy()
            # Pyliger normalizes by library size with a size factor of 1
            # So here we give it the count data
            bdata.X = bdata.layers["counts"]
            # List of adata per batch
            adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]
            for i, ad in enumerate(adata_list):
                ad.uns["sample_name"] = batch_cats[i]
                # Hack to make sure each method uses the same genes
                ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)
                
                if isinstance(adata_list[i].X, np.ndarray):
                    print(f"Converting dataset {i} to sparse...")
                    adata_list[i].X = csr_matrix(adata_list[i].X)
            
            liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
            # Hack to make sure each method uses the same genes
            liger_data.var_genes = bdata.var_names
            
            
            print(liger_data)
            
            pyliger.normalize(liger_data)
            pyliger.scale_not_center(liger_data)
            pyliger.optimize_ALS(liger_data, k=n_components)
            pyliger.quantile_norm(liger_data)


            adata.obsm["LIGER"] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
            for i, b in enumerate(batch_cats):
                adata.obsm["LIGER"][adata.obs.batch == b] = liger_data.adata_list[i].obsm["H_norm"]
                
        # ------------------ run harmony --------------------------------
        elif model_name.lower() == "harmony":
            # Run Harmony to correct for batch effects 
            harmony_out = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, vars_use=[batch_key])

            adata.obsm["Harmony"] = harmony_out.Z_corr

        # ------------------- run scVI --------------------------------------------------
        elif model_name.lower() == "scvi":
            scvi.settings.seed = seed
            SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
            vae = SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=n_components)
            vae.train()
            adata.obsm["scVI"] = vae.get_latent_representation()

        # ------------------ run scANVI ------------------------------------------------
        elif model_name.lower() == "scanvi":
            # torch.use_deterministic_algorithms(True)
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True 
            # # did not work:  
            #Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, 
            #but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. 
            #To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility  '''
            
            scvi.settings.seed = seed
            # if not('vae' in locals() and vae is not None): # if vae is not defined (from previously running scvi)
            # retrain scvi to avoid reproducibility issues
            SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
            vae = SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=n_components)
            vae.train()
            
            lvae = SCANVI.from_scvi_model(
                vae,
                adata=adata,
                labels_key=label_key,
                unlabeled_category="Unknown",
            )
            lvae.train(max_epochs=20, n_samples_per_label=100)
            adata.obsm["scANVI"] = lvae.get_latent_representation()
        
        else:
            print(f"Method {model_name} not recognized.")      
        
    time_taken = time.time() - start_time
    _, peak_mem_after = tracemalloc.get_traced_memory()
    peak_mem_delta_mb = max(0.0, (peak_mem_after - current_mem_start) / (1024 ** 2))
    tracemalloc.stop()
    return adata, time_taken, peak_mem_delta_mb

def setup_visualization(embedding, y_source, y_target):
     # 1. SETUP & SIZE CHECKS
    # ------------------------------------
    # Ensure we are working with flat arrays
    y_source_vis = np.array(y_source).flatten()
    y_target_vis = np.array(y_target).flatten()

    n_a = len(y_source_vis)
    n_b = len(y_target_vis)
    n_total_labels = n_a + n_b
    n_total_embed = embedding.shape[0]

    print(f"Data Check: Source={n_a}, Target={n_b} | Embedding shape={embedding.shape}")

    # Safety Fix: If embedding doesn't match labels (e.g. stale variables), we stop to prevent weird plots
    if n_total_embed != n_total_labels:
        print(f"(!) CRITICAL WARNING: Embedding has {n_total_embed} points but labels have {n_total_labels}.")
        print("    Adjusting logic to fit the SMALLER size to avoid crash.")
        min_len = min(n_total_embed, n_total_labels)
        # Truncate to safe limit for plotting purposes
        labels_combined = np.concatenate([y_source_vis, y_target_vis])[:min_len]
        embedding_vis = embedding[:min_len]
        
        # Re-calculate split point for domains
        # We assume Source is first. If Source is larger than min_len, Source gets cut.
        # Ideally, this shouldn't happen if code is run in order.
        real_n_a = min(n_a, min_len)
        real_n_b = min_len - real_n_a
    else:
        labels_combined = np.concatenate([y_source_vis, y_target_vis]).astype(int)
        embedding_vis = embedding
        real_n_a = n_a
        real_n_b = n_b

    # Create Domain Labels
    domains_combined = np.concatenate([
        np.full(real_n_a, "Domain A (Source)"),
        np.full(real_n_b, "Domain B (Target)")
    ])
    
    return embedding_vis, labels_combined, domains_combined, real_n_a, real_n_b

def visualization(embedding, y_source, y_target, seed=42, title= "", save_path = None, T_true = None, keep_idx = None):
    # if save_path is provided, saves the figure to that path
    # ------------------------------------
    # VISUALIZATION (Robust to Uneven Sizes)
    # ------------------------------------
    print("\n--- Visualization ---")

    embedding_vis, labels_combined, domains_combined, real_n_a, real_n_b = setup_visualization(embedding, y_source, y_target)

    # 2. GENERATE PLOTS
    # ------------------------------------
    plt.figure(figsize=(18, 8))
    plt.suptitle(title)
    
    # === PLOT 1: ALIGNMENT BY DOMAIN ===
    ax1 = plt.subplot(1, 2, 1)

    domain_colors = {"Domain A (Source)": "#377eb8", "Domain B (Target)": "#ff7f00"}
    for domain, color in domain_colors.items():
        idx = (domains_combined == domain)
        if np.sum(idx) > 0:
            ax1.scatter(
                embedding_vis[idx, 0], embedding_vis[idx, 1],
                c=color, label=domain, alpha=0.6, s=20, edgecolors='none'
            )

    # --- MATCHING LINES (ROBUST VERSION) ---
    try:
        if 'T_true' in locals() and T_true is not None:
            T_vis = T_true
            if sparse.issparse(T_vis): T_vis = T_vis.toarray()
            
            # KEY FIX: Adjust T_true to match removed samples
            # If we have 'keep_idx' (indices of targets kept), we must slice T_true columns
            if 'keep_idx' in locals() and keep_idx is not None and len(keep_idx) == real_n_b:
                if T_vis.shape[1] > len(keep_idx):
                    print(f"  Adjusting T_true columns using keep_idx...")
                    T_vis = T_vis[:, keep_idx]
            
            # Fallback: If shape still mismatches, slice strictly by dimensions
            if T_vis.shape != (real_n_a, real_n_b):
                print(f"  (!) T_true shape {T_vis.shape} != Data {(real_n_a, real_n_b)}. Slicing to fit.")
                T_vis = T_vis[:real_n_a, :real_n_b]

            # Draw Lines
            # Calculate row sums to see which Source samples actually match a REMAINING Target
            row_sums = T_vis.sum(axis=1)
            valid_sources = np.where(row_sums > 0.001)[0] # Tolerance for float
            
            # Limit to 50 lines
            if len(valid_sources) > 50:
                np.random.seed(seed)  # For reproducibility
                plot_indices = np.random.choice(valid_sources, 50, replace=False)
            else:
                plot_indices = valid_sources

            line_count = 0
            for i in plot_indices:
                j = np.argmax(T_vis[i, :]) # Best match index in the NEW target set
                
                # Final bounds check
                if i < real_n_a and j < real_n_b:
                    x1, y1 = embedding_vis[i]
                    x2, y2 = embedding_vis[real_n_a + j]
                    ax1.plot([x1, x2], [y1, y2], c='black', alpha=0.8, linewidth=5)
                    line_count += 1
                    
            print(f"  Plotted {line_count} matching lines.")

    except Exception as e:
        print(f"(!) Could not plot matching lines: {e}")

    ax1.set_title(f"Alignment by Domain ({real_n_a} vs {real_n_b})", fontweight="bold")
    ax1.legend()
    ax1.set_xticks([]); ax1.set_yticks([])


    # === PLOT 2: ALIGNMENT BY LABEL ===
    ax2 = plt.subplot(1, 2, 2)

    # Handle -1 (Unlabeled) vs Known Classes
    mask_labeled = labels_combined != -1
    mask_unlabeled = labels_combined == -1

    # Background: Unlabeled
    if np.any(mask_unlabeled):
        ax2.scatter(
            embedding_vis[mask_unlabeled, 0], embedding_vis[mask_unlabeled, 1],
            c='lightgrey', label='Unlabeled', alpha=0.3, s=10
        )

    # Foreground: Labeled
    if np.any(mask_labeled):
        unique_classes = np.unique(labels_combined[mask_labeled])
        cmap = plt.get_cmap("tab10")

        for i, lbl in enumerate(unique_classes):
            idx = (labels_combined == lbl)
            ax2.scatter(
                embedding_vis[idx, 0], embedding_vis[idx, 1],
                color=cmap(i % 10), label=f"Class {lbl}", alpha=0.7, s=20
            )

    ax2.set_title("Alignment by Class Label", fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def benchmark_from_adata(adata, methods, batch_key = "batch", label_key = "cell_type", pre_integrated_embedding_obsm_key = "Unintegrated", save_path= None, seed=42):
    # Benchmarking with scIB_metrics (faster)
    # adds the missing metrics from scib (trajectory preservation and cell cycle conservation)
    # saves to save_path if provided, otherwise not saved
    # seed: Random seed for reproducible metrics (controls Leiden, KMeans, KNN operations)
    
    set_seeds(seed)
    # reset seeds for reproducible results
  
    
    # only keep the methods that are present in adata.obsm (to avoid crashes)
    methods = [m for m in methods if m in adata.obsm.keys()]
    
    
    # BENCHMARK
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys= [pre_integrated_embedding_obsm_key] + methods,
        pre_integrated_embedding_obsm_key= pre_integrated_embedding_obsm_key,
    #     n_jobs=1, #int(os.environ.get("NUMBA_NUM_THREADS", "Not Set")),
        batch_correction_metrics = BatchCorrection(ilisi_knn=True, 
                                                   kbet_per_label=True, 
                                                   graph_connectivity=True, 
                                                   pcr_comparison=True),
        bio_conservation_metrics = BioConservation(isolated_labels=True, 
                                                   nmi_ari_cluster_labels_leiden=True,
                                                   nmi_ari_cluster_labels_kmeans=True, 
                                                   silhouette_label=True, 
                                                   clisi_knn=True)
    )

    bm.benchmark()

    if save_path is not None:
        # save bm object so it can be reused to plot more things later.
        bm_save_file= save_path+"/bm.pkl" #+str(MALI.m) + ".pkl"
        pickle_operator = open(bm_save_file, 'wb')
        pickle.dump(bm, pickle_operator)
        pickle_operator.close()
        
    df = bm.get_results()

    # add label-free metrics from scib
    trajectory_ = True if 'dpt_pseudotime' in adata.obs else False
   
    
    for method in methods:
    # only cell cycle and trajectory preservation, if the data has pseudotime information (dpt_pseudotime in adata.obs)
        if trajectory_:
            sc.pp.neighbors(adata, use_rep=method) 
            # recompute the connectivities on the integrated embedding
        
        set_seeds(seed)
        # reset seeds for reproducible results

        scib_results = scib.me.metrics(
            adata,
            adata,
            embed=method,
            verbose=True,
            hvg_score_=False,
            cluster_nmi=False,
            batch_key=batch_key,
            label_key=label_key,
            silhouette_=False,
            type_="embed",
            nmi_=False,
            nmi_method=None,
            nmi_dir=None,
            ari_=False,
            pcr_=False,
            cell_cycle_=True, # cell cycle
            organism="human",
            isolated_labels_=False,
            n_isolated=None,
            graph_conn_=False,
            kBET_=False,
            # lisi_=lisi_,
            lisi_graph_=False,
            trajectory_= trajectory_, # trajectory preservation (only if pseudotime exists in the data). could compute it but requires root cell identification and more preprocessing, so we skip it for now
        )
    
        df.loc[df.index== method, "cell_cycle_score"] = scib_results.loc["cell_cycle_conservation"].values[0]
        df.loc[df.index == "Metric Type", "cell_cycle_score"] = "Bio Conservation (label-free)"
        if trajectory_:
            df.loc[df.index == method, "trajectory_score"] = scib_results.loc["trajectory"].values[0]
            df.loc[df.index == "Metric Type", "trajectory_score"] = "Bio Conservation (label-free)"
 

    # visualize
    bm.plot_results_table(min_max_scale=False, save_dir = save_path)

    return df



def benchmark_from_adata_scib(adata, methods, batch_key = "batch", label_key = "cell_type", pre_integrated_embedding_obsm_key = "Unintegrated", save_path= None):
    # Benchmarking with scIB_metrics
    # saves to save_path if provided, otherwise not saved
    # kind of a hack to follow this code: https://github.com/theislab/scib-pipeline/blob/main/scripts/metrics/metrics.py
    # only keep the methods that are present in adata.obsm (to avoid crashes)
    methods = [m for m in methods if m in adata.obsm.keys()]
    
    silhouette_ = True
    nmi_ = True
    ari_ = True
    pcr_ = True
    cell_cycle_ = True
    isolated_labels_ = True
    hvg_score_ = False # because we only have embeddings, not transformed counts
    graph_conn_ = True
    # kBET_ = True
    kBET_ = False # because we need rpy2 which requires an R installation
    # lisi_ = True
    # lisi_graph_ = True
    lisi_graph_ = False # was creating bugs. might be due to rare cell types (1 cell)
    
    
    n_hvgs = None
    recompute_neighbors = True
    precompute_pca = True
    embed = "X_emb"
    organism='human' # or mouse if using another dataset. it's for the cell cycle metric only, so it doesn't affect the other metrics.
    
    output = f"{save_path}/scib_metrics_results.csv"
    
    # check if pseudotime data exists in original data
    if 'dpt_pseudotime' in adata.obs:
        trajectory_ = True
    else:
        trajectory_ = False
    
    
    # create cluster NMI output file
    file_stump = os.path.splitext(output)[0]
    cluster_nmi = f'{file_stump}_nmi.txt'
    
     
    for method in methods:
        print(f"Running scIB benchmark for method {method}...")
        adata_int = adata.copy()
        adata_int.obsm["X_emb"] = adata.obsm[method]
        
        
        scib.preprocessing.reduce_data(
                adata_int,
                n_top_genes=n_hvgs,
                neighbors=recompute_neighbors,
                use_rep=embed,
                pca=precompute_pca,
                umap=False
            )
        
        scib.preprocessing.reduce_data( # need to rerun it because the neighbors need to be recomputed on the benchmark adata (subset from the whole thing with hidden labels)
                adata,
                overwrite_hvg = False, # skip HVG selection (should already be done from the original preprocessing)
                neighbors=True,
                use_rep='X_pca',
                pca=True, # should already be done
                umap=False
            )
        
        
        results = scib.me.metrics(
            adata,
            adata_int,
            verbose=True,
            hvg_score_=hvg_score_,
            cluster_nmi=cluster_nmi,
            batch_key=batch_key,
            label_key=label_key,
            silhouette_=silhouette_,
            embed=embed,
            type_="embed",
            nmi_=nmi_,
            nmi_method='arithmetic',
            nmi_dir=None,
            ari_=ari_,
            pcr_=pcr_,
            cell_cycle_=cell_cycle_,
            organism=organism,
            isolated_labels_=isolated_labels_,
            n_isolated=None,
            graph_conn_=graph_conn_,
            kBET_=kBET_,
            # lisi_=lisi_,
            lisi_graph_=lisi_graph_,
            trajectory_=trajectory_,
        )
        results.rename(columns={results.columns[0]: method}, inplace=True)

        
        results.insert(0, "method", method)
        if 'results_df' in locals():
            results_df = pd.concat([results_df, results], ignore_index=True)
        else:
            results_df = results
    
    results_df.to_csv(output)

    return results_df




def run_metrics_from_adata(adata, method_name, batch_key = "batch", label_key = "cell_type", masked_label_key = "labels_masked", **kwargs):
    # assumes an adata with embedding stored in adata.obsm[method_name]
    x0, x1, y0, y1, idxs_d = split_adata(adata, batch_key=batch_key, label_key=label_key, **kwargs)
    
    # here the labels are the true labels (before masking)
    y_source = y0
    y_target_true = y1 
    
    # get the masked labels
    idxs_1 = idxs_d[x0.shape[0]:] # indices of target batch
    y_target = adata.obs[masked_label_key][idxs_1].values
    
    mask_missing_target_full = y_target == -1  # boolean mask of missing labels in target
    
    return run_metrics(adata.obsm[method_name], y0, y_target, y_target_true, mask_missing_target_full)


def run_metrics(embedding, y_source, y_target, y_target_true, mask_missing_target_full):
    # assumes embedding is of shape (n_source + n_target, d), i.e a concatenation of source and target embeddings
    embedding_vis, labels_combined, domains_combined, real_n_a, real_n_b = setup_visualization(embedding, y_source, y_target)
    
    # METRICS (Label transfer)
    # ------------------------------------
    try:

        # after alignment + embedding
        n_a = len(y_source)          # source was not removed in your snippet
        n_b = len(y_target)          # target after removal
        
        res = label_transfer_accuracy(
            embedding=embedding,                 # shape (n_a + n_b, d)
            n_a=n_a,
            y_source=y_source,                   # shape (n_a,)
            y_target_obs=y_target,               # shape (n_b,) with -1 for masked
            y_target_true=y_target_true,         # shape (n_b,) ground truth (post-removal)
            mask_missing_target=mask_missing_target_full,  # shape (n_b,) post-removal
            n_neighbors=5,
        )
        
        print(f"Transfer acc on MASKED target labels: {res['acc_missing']}")
        print(f"(Optional) acc on VISIBLE target labels: {res['acc_visible']}")


        # Domain mixing unchanged
        sil_dom = silhouette_score(embedding_vis, domains_combined)
        print(f"Domain Mixing (Lower is better): {sil_dom:.4f}")



        # FOSCTTM --------------------------------------------------------------------
        x1_mat = embedding[0:n_a]
        x2_mat = embedding[n_a:]

        fracs = calc_domainAveraged_FOSCTTM(x1_mat, x2_mat, true_pairs_1to2=None, true_pairs_2to1=None)
        foscttm = fracs.mean()
        print(f"Average FOSCTTM (Lower is better): {foscttm:.4f}")



        # alignment score from pamona 
        from src.Pamona.eval import test_alignment_score
        alignment_score = test_alignment_score(x1_mat, x2_mat)
        
        
    except Exception as e:
        print(f"Metrics error: {e}")
        
        
    return res, sil_dom, foscttm, alignment_score
        


