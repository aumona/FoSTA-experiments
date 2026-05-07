import os

import matplotlib.pyplot as plt
import numpy as np
from phate import phate
import rfphate
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
import seaborn as sns
import scanpy as sc
import pandas as pd

from metrics import label_transfer_accuracy
from matplotlib.legend_handler import HandlerTuple

def visualization(y_source, y_target, embedding, T_true=None, seed=42, keep_idx=None, mask_missing_target_full=None):
    # directly copied from the experiment.ipynb 
    
    
    # ------------------------------------
    # VISUALIZATION (Robust to Uneven Sizes)
    # ------------------------------------
    print("\n--- Visualization ---")

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

    # 2. GENERATE PLOTS
    # ------------------------------------
    plt.figure(figsize=(18, 8))

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
            if 'keep_idx' in locals() and len(keep_idx) == real_n_b:
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
    plt.show()


    # ------------------------------------
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

    except Exception as e:
        print(f"Metrics error: {e}")
        
        
        
        
# Function to format the top 3 values: bold, underline, italic (adjusted for lower is better columns)
def format_top_3(df):
    formatted_df = df.copy()
    
    # List of columns where lower values are better
    lower_is_better_columns = ['FOSCTTM', 'Silhouette domain']
    
    
    for column in df.columns:  # Start from the first numerical column
        if column in lower_is_better_columns:
            # Find the smallest 3 values (lower is better)
            top_3 = df[column].nsmallest(3).values
        else:
            # Find the largest 3 values (higher is better)
            top_3 = df[column].nlargest(3).values
        
        if len(top_3) >= 3:
            # Apply formatting: bold for best, underline for second, italic for third
            formatted_df.loc[df.index, column] = df[column].apply(
                lambda x: f"\\textbf{{{x:.3f}}}" if x == top_3[0] else 
                            (f"\\underline{{{x:.3f}}}" if x == top_3[1] else 
                            (f"\\textit{{{x:.3f}}}" if x == top_3[2] else f"{x:.3f}"))
            )
    
    return formatted_df


def plot_metric_grouped_by(results_df, groupby_cols=["noise_std", "dropout_prob"], methods=None, n_components=None, annotate=True, fig_len_factor = 1):
    
    if methods is not None:
        results_df = results_df[results_df["method"].isin(methods)]
    
    if n_components is not None:
        results_df = results_df[results_df["n_components"] == n_components]
        
    cols_to_plot = ["Batch correction", "Bio conservation", "Total"]

    if len(groupby_cols) == 1:
        plot_cols = groupby_cols + ["method"]  # add method as hue if only one grouping variable
    else:
        plot_cols = groupby_cols
   
    for col in cols_to_plot:
        fig, ax = plt.subplots(figsize=(18*fig_len_factor, 7), layout="constrained")

        means = (
            results_df.groupby(groupby_cols, as_index=False)[col]
            .mean()
        )
        
        # means = means.sort_values(by=groupby_cols, ascending=False)
        # print(means)

        sns.barplot(
            data=means,
            x=plot_cols[0],
            y=col,
            hue=plot_cols[1],
            palette="tab20",
            ax=ax
        )

        # get categorical positions used by seaborn
        x_levels = means[plot_cols[0]].unique()
        hue_levels = means[plot_cols[1]].unique()
        n_hue = len(hue_levels)

        # width of a single bar (seaborn default)
        bar_width = 0.8 / n_hue

        # annotate each bar explicitly
        if annotate:
            for i, (_, r) in enumerate(means.iterrows()):
                x_idx = list(x_levels).index(r[plot_cols[0]])
                h_idx = list(hue_levels).index(r[plot_cols[1]])
                x = x_idx - 0.4 + bar_width / 2 + h_idx * bar_width
                y = r[col]

                ax.text(
                    x,
                    y/2,                      # centered vertically in bar
                    f"{r[plot_cols[1]]} = {y:.3f}",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9
                )

        ax.legend(
            title=plot_cols[1],
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        ax.set_title(f"{col} by {plot_cols[0]}")
        ax.tick_params(axis="x", rotation=90)

        plt.show()


def plot_bio_vs_batch_correction(results_df, title = "Bio conservation vs Batch correction", save_path=None):
    # 1. Calculate Mean and Standard Deviation
    mean_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].mean()
    std_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].std()
    std_df = std_df.fillna(0)

    # 2. Calculate Ranks
    mean_df['Bio Rank'] = mean_df['Bio conservation'].rank(ascending=False).astype(int)
    mean_df['Batch Rank'] = mean_df['Batch correction'].rank(ascending=False).astype(int)
    
    # 3. Calculate Average Rank and Sort
    mean_df['Avg Rank'] = (mean_df['Bio Rank'] + mean_df['Batch Rank']) / 2
    mean_df = mean_df.sort_values('Avg Rank')

    # --- ADJUSTMENT: Smaller Figure Size ---
    plt.figure(figsize=(5, 5)) 
    
    unique_methods = results_df['method'].unique()
    palette_list = sns.color_palette("tab20", n_colors=len(unique_methods))
    color_map = dict(zip(unique_methods, palette_list))
    
    for method in mean_df.index:
        x = mean_df.loc[method, "Batch correction"]
        y = mean_df.loc[method, "Bio conservation"]
        x_err = std_df.loc[method, "Batch correction"]
        y_err = std_df.loc[method, "Bio conservation"]
        
        bio_rank = mean_df.loc[method, 'Bio Rank']
        batch_rank = mean_df.loc[method, 'Batch Rank']
        legend_label = f"{method} (Bio #{bio_rank}, Batch #{batch_rank})"

        # Plot Error Bars
        plt.errorbar(
            x, y, 
            xerr=x_err, 
            yerr=y_err, 
            fmt='none',
            ecolor=color_map[method],
            elinewidth=1.5,
            capsize=5,
            alpha=0.4, 
            label=None 
        )

        # Plot Marker
        plt.scatter(
            x, y, 
            s=100, 
            color=color_map[method], 
            label=legend_label, 
            zorder=3
        )

    # --- ADJUSTMENT: Legend Styling ---
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        title="Method (Ordered by Avg Rank)",
        frameon=True,
        fontsize='small',  # Make text smaller to fit the smaller figure height
        title_fontsize='medium'
    )
    
    plt.title(title)
    plt.xlabel("Batch correction")
    plt.ylabel("Bio conservation")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path is not None:
        # 'bbox_inches="tight"' ensures the external legend is not cut off when saving
        plt.savefig(f"{save_path}/bio_vs_batch_correction.pdf", format='pdf', bbox_inches="tight")
        plt.savefig(f"{save_path}/bio_vs_batch_correction.png", format='png', bbox_inches="tight")
    plt.show()
    return


def rename_reorder_methods(results_df, right_fosta_name):
    if right_fosta_name not in results_df["method"].values:
        print(f"(!) Warning: '{right_fosta_name}' not found in 'method' column. please try with a different name.")
        return results_df
    
    results_df["method"] = results_df["method"].replace({right_fosta_name: "FoSTA"})

    methods_to_plot = define_methods_to_plot()
    results_df = results_df[results_df['method'].isin(methods_to_plot)] # subset to methods to keep

   
    methods_to_plot.remove("FoSTA")
    methods_to_plot.insert(0, "FoSTA")  # FoSTA second
    methods_to_plot.remove("Unintegrated")
    methods_to_plot.append("Unintegrated") # Unintegrated last

    results_df["method"] = pd.Categorical(results_df["method"], ordered=True, categories = methods_to_plot)

    return results_df

def define_methods_to_plot():
    methods_list = ['Unintegrated', 
                    'FoSTA',
                    'LIGER', 
                    'Scanorama', 
                    'scANVI', 
                    'scVI',
                    'KEMAlin', 
                    'KEMArbf', 
                    'MALI', 
                    'Pamona', 
                    # 'Harmony',
                    # below to remove after testing
                    # 'FoSTA_t2',
                    # 'FoSTA_t2_balanced', 
                    # 'FoSTA_t2_et', 
                    # 'FoSTA_tauto_kerf',
                    # 'FoSTA_tauto_gap',
                    # 'MALI_unshared_labels',
                    # 'Pamona_unshared_labels',
                    # 'KEMArbf_unshared_labels',
                    # 'KEMAlin_unshared_labels'
                    ]

    return methods_list


# def make_symbol_map(methods):
    
#     return 

def plot_bio_vs_batch_correction_markers(results_df, title="Bio conservation vs Batch correction", methods_to_plot=None, save_path=None):
    # subset to the methods to plot
    if methods_to_plot is not None:
        results_df = results_df[results_df['method'].isin(methods_to_plot)]
    else:
        methods_to_plot = results_df['method'].unique()
        
    # 1. Calculate Mean and Standard Deviation
    mean_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].mean()
    std_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].std()
    std_df = std_df.fillna(0)

    # 2. Calculate Ranks
    mean_df['Bio Rank'] = mean_df['Bio conservation'].rank(ascending=False).astype(int)
    mean_df['Batch Rank'] = mean_df['Batch correction'].rank(ascending=False).astype(int)
    
    # 3. Average Rank and Sort
    mean_df['Avg Rank'] = (mean_df['Bio Rank'] + mean_df['Batch Rank']) / 2
    mean_df = mean_df.sort_values('Avg Rank')

    plt.figure(figsize=(6, 6)) 
    
    symbols = [
        r"$\mathit{0}$", r"$\mathit{f}$", r"$\mathit{\kappa}$", r"$\mathit{\alpha}$", r"$\mathit{\beta}$", r"$\mathit{\gamma}$", 
        r"$\mathit{\delta}$", r"$\mathit{\epsilon}$", r"$\mathit{\zeta}$", 
        r"$\mathit{\eta}$", r"$\mathit{\theta}$",
        r"$\mathit{\lambda}$", r"$\mathit{\mu}$", r"$\mathit{\nu}$", 
        r"$\mathit{\xi}$", r"$\mathit{\pi}$", r"$\mathit{\rho}$", 
        r"$\mathit{\sigma}$", r"$\mathit{\tau}$", r"$\mathit{\phi}$", 
        r"$\mathit{\chi}$", r"$\mathit{\psi}$", r"$\mathit{\omega}$",
        r"$\clubsuit$", r"$\spadesuit$", r"$\heartsuit$", r"$\diamondsuit$",
        r"$\star$", r"$\dagger$", r"$\ddagger$", r"$\S$", r"$\P$"
    ]
    palette_list = sns.color_palette("tab20", n_colors=len(methods_to_plot))
    color_map = dict(zip(methods_to_plot, palette_list))
    symbol_map = dict(zip(methods_to_plot, symbols[:len(methods_to_plot)]))
    
    legend_handles = []
    legend_labels = []

    for method in mean_df.index:
        if method not in methods_to_plot:
            continue
        x = mean_df.loc[method, "Batch correction"]
        y = mean_df.loc[method, "Bio conservation"]
        x_err = std_df.loc[method, "Batch correction"]
        y_err = std_df.loc[method, "Bio conservation"]
        
        bio_rank = mean_df.loc[method, 'Bio Rank']
        batch_rank = mean_df.loc[method, 'Batch Rank']
        label = f"{method} (Bio #{bio_rank}, Batch #{batch_rank})"

        # 1. Error Bars
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                     ecolor=color_map[method], elinewidth=1.5, capsize=5, alpha=0.3)

        # 2. Circle (Background)
        c = plt.scatter(x, y, s=180, color=color_map[method], marker='o', zorder=3)
        
        # 3. Symbol (Foreground)
        s = plt.scatter(x, y, s=50, color="white", marker=symbol_map[method], zorder=4)

        legend_handles.append((c, s))
        legend_labels.append(label)

    # --- UPDATED LEGEND LOGIC ---
    # ndivide=None tells the legend to plot both items at the same center point
    # handlelength=1.5 ensures there is enough room for the circular icon to be centered
    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=-2)},
        bbox_to_anchor=(1.05, 1), loc='upper left', 
        title="Method (Ordered by Avg Rank)",
        frameon=True, 
        fontsize='small',
        handlelength=2, 
        handletextpad=0.5
    )
    
    plt.title(title)
    plt.xlabel("Batch correction")
    plt.ylabel("Bio conservation")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path is not None:
        plt.savefig(f"{save_path}/bio_vs_batch_correction.png", bbox_inches="tight")
        plt.savefig(f"{save_path}/bio_vs_batch_correction.pdf", bbox_inches="tight")
    plt.show()
    
    

def plot_bio_vs_batch_correction_markers_order_by_mean(results_df, title="Bio conservation vs Batch correction", methods_to_plot=None, save_path=None):
    # subset to the methods to plot
    if methods_to_plot is not None:
        results_df = results_df[results_df['method'].isin(methods_to_plot)]
    else:
        methods_to_plot = results_df['method'].unique()
        
    # 1. Calculate Mean and Standard Deviation
    mean_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction", "Total"]].mean()
    std_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction", "Total"]].std()
    std_df = std_df.fillna(0)

    # # 2. Calculate Ranks
    # mean_df['Bio Rank'] = mean_df['Bio conservation'].rank(ascending=False).astype(int)
    # mean_df['Batch Rank'] = mean_df['Batch correction'].rank(ascending=False).astype(int)
    
    # # 3. Average Rank and Sort
    # mean_df['Avg Rank'] = (mean_df['Bio Rank'] + mean_df['Batch Rank']) / 2
    # mean_df = mean_df.sort_values('Avg Rank')
    mean_df = mean_df.sort_values('Total', ascending=False)

    plt.figure(figsize=(6, 6)) 
    
    symbols = [
        r"$\mathit{0}$", r"$\mathit{f}$", r"$\mathit{\kappa}$", r"$\mathit{\alpha}$", r"$\mathit{\beta}$", r"$\mathit{\gamma}$", 
        r"$\mathit{\delta}$", r"$\mathit{\epsilon}$", r"$\mathit{\zeta}$", 
        r"$\mathit{\eta}$", r"$\mathit{\theta}$",
        r"$\mathit{\lambda}$", r"$\mathit{\mu}$", r"$\mathit{\nu}$", 
        r"$\mathit{\xi}$", r"$\mathit{\pi}$", r"$\mathit{\rho}$", 
        r"$\mathit{\sigma}$", r"$\mathit{\tau}$", r"$\mathit{\phi}$", 
        r"$\mathit{\chi}$", r"$\mathit{\psi}$", r"$\mathit{\omega}$",
        r"$\clubsuit$", r"$\spadesuit$", r"$\heartsuit$", r"$\diamondsuit$",
        r"$\star$", r"$\dagger$", r"$\ddagger$", r"$\S$", r"$\P$"
    ]
    palette_list = sns.color_palette("tab20", n_colors=len(methods_to_plot))
    color_map = dict(zip(methods_to_plot, palette_list))
    symbol_map = dict(zip(methods_to_plot, symbols[:len(methods_to_plot)]))
    
    legend_handles = []
    legend_labels = []

    for method in mean_df.index:
        if method not in methods_to_plot:
            continue
        x = mean_df.loc[method, "Batch correction"]
        y = mean_df.loc[method, "Bio conservation"]
        x_err = std_df.loc[method, "Batch correction"]
        y_err = std_df.loc[method, "Bio conservation"]
        
        # bio_rank = mean_df.loc[method, 'Bio Rank']
        # batch_rank = mean_df.loc[method, 'Batch Rank']
        # label = f"{method} (Bio #{bio_rank}, Batch #{batch_rank})"
        label = f"{method} (Total={mean_df.loc[method, 'Total']:.3f})"

        # 1. Error Bars
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                     ecolor=color_map[method], elinewidth=1.5, capsize=5, alpha=0.3)

        # 2. Circle (Background)
        c = plt.scatter(x, y, s=180, color=color_map[method], marker='o', zorder=3)
        
        # 3. Symbol (Foreground)
        s = plt.scatter(x, y, s=50, color="white", marker=symbol_map[method], zorder=4)

        legend_handles.append((c, s))
        legend_labels.append(label)

    # --- UPDATED LEGEND LOGIC ---
    # ndivide=None tells the legend to plot both items at the same center point
    # handlelength=1.5 ensures there is enough room for the circular icon to be centered
    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=-2)},
        bbox_to_anchor=(1.05, 1), loc='upper left', 
        title="Method (Ordered by Total)",
        frameon=True, 
        fontsize='small',
        handlelength=2, 
        handletextpad=0.5
    )
    
    plt.title(title)
    plt.xlabel("Batch correction")
    plt.ylabel("Bio conservation")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path is not None:
        plt.savefig(f"{save_path}/bio_vs_batch_correction.png", bbox_inches="tight")
        plt.savefig(f"{save_path}/bio_vs_batch_correction.pdf", bbox_inches="tight")
    plt.show()
    
def clean_cell_types(adata, threshold=10):
    adata.obs["original_cell_type"] = adata.obs["cell_type"]
    
    value_counts = adata.obs["cell_type"].value_counts()
    small_types = value_counts.index[value_counts < threshold]

    # replace cell_type "Type 2" with "Alveolar Type 2" 
    adata.obs.replace({"cell_type": {"Type 2": "Alveolar Type 2"}}, inplace=True)

    # replace cell_type "Type 2" with "Alveolar Type 2" 
    adata.obs.replace({"cell_type": {"Type 2": "Alveolar Type 2"}}, inplace=True)
    # replace cell_type "Endothelium" with "Endothelium (Blood Endothelial Cell)" 
    adata.obs.replace({"cell_type": {"Endothelium": "Blood Endothelial Cell"}}, inplace=True)
    # replace cell_type "Lymphatic" with "Lymphatic (Lymphatic Endothelial Cell)" 
    adata.obs.replace({"cell_type": {"Lymphatic": "Lymphatic Endothelial Cell"}}, inplace=True)

    for small_type in small_types:
        adata.obs.replace({"cell_type": {small_type: "Other"}}, inplace=True)


    # reorder the categories so that "Other" is last
    categories = [cat for cat in adata.obs["cell_type"].cat.categories if cat != "Other"]
    categories.append("Other")
    adata.obs["cell_type"] = adata.obs["cell_type"].cat.reorder_categories(categories)

    return adata


def load_adata(save_path, noise_level=0.5, dropout_level=0.5, seed=39041):
    # format noise and dropout to match folder names (1 decimal place)
    noise_level = f"{noise_level:.1f}"
    dropout_level = f"{dropout_level:.1f}"
    path = f"{save_path}/noise_{noise_level}_dropout_{dropout_level}/2_components/seed_{seed}"
    adata = sc.read_h5ad(f"{path}/adata_intermediate.h5ad")

    adata = clean_cell_types(adata)
    return adata, path


    
def add_rfphate_phate_embeddings(adata = None, save_path = None, label_key = "cell_type", batch_key = "simulated_batch", noise_level=0.5, dropout_level=0.5, seed=39041, embedding_basis = "X_pca", t=2):
    if adata is None:
        adata, path = load_adata(save_path, noise_level=noise_level, dropout_level=dropout_level, seed=seed)
    else:
        path = None
    phate_operator = phate.PHATE(n_components=2, t=t, random_state=42)
    rfphate_operator = rfphate.RFPHATE(n_components=2, t=t, random_state=42)

    phate_embeddings = phate_operator.fit_transform(adata.obsm[embedding_basis])
    rfphate_embeddings = rfphate_operator.fit_transform(adata.obsm[embedding_basis], adata.obs[label_key])

    adata.obsm[f"PHATE_t{t}"] = phate_embeddings
    adata.obsm[f"RFPHATE_t{t}"] = rfphate_embeddings
    
    return adata, path


def plot_embeddings(adata = None, methods = None, noise_level=0.5, dropout_level=0.5, save_path = None, rename_dict = None, figsize=None, seed= None, right_fosta_key="FoSTA", marker_test="*", size_test=20, **kwargs):
    label_key = "cell_type"
    batch_key = "simulated_batch"

    if adata is None:
        adata, save_path = load_adata(save_path=save_path, noise_level=noise_level, dropout_level=dropout_level, seed= seed)
     
    if right_fosta_key not in adata.obsm.keys():
        right_fosta_key = "FoSTA"
    else:
        # rename FoSTA_PHATE_t2 to FoSTA in the adata
        adata.obsm["FoSTA"] = adata.obsm[right_fosta_key]

    if methods is None:
        methods = list(adata.obsm.keys())
    n = len(methods)
    
    if figsize is None:
        figsize = (6*n, 8)
        
    fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=n)


    # ---- TOP ROW (batch_key) ----
    for i, method in enumerate(methods):
        if rename_dict is None:
            method_name = method
        else:
            method_name = rename_dict[method]
            
        sc.pl.embedding(
            adata[adata.obs['mask_indices'] == 0],
            ax=axes[0, i],
            basis=method,
            color=batch_key,
            title=f"{method_name}", # : {batch_key}
            show=False,
            legend_loc="right" if i == 0 else None,  # create legend ONCE
            marker='.',
            alpha=0.3,
            **kwargs
        )
        
        sc.pl.embedding(
            adata[adata.obs['mask_indices'] == 1], # test set have different markers
            ax=axes[0, i],
            basis=method,
            color=batch_key,
            title=f"{method_name}", # {label_key}
            show=False,
            legend_loc= None,
            marker=marker_test,
            size=size_test,
            **kwargs
        )

    # ---- BOTTOM ROW (label_key) ----
    for i, method in enumerate(methods):
        if rename_dict is None:
            method_name = method
        else:
            method_name = rename_dict[method]
            
        sc.pl.embedding(
            adata[adata.obs['mask_indices'] == 0],
            ax=axes[1, i],
            basis=method,
            color=label_key,
            title=f"{method_name}", # {label_key}
            show=False,
            legend_loc="right" if i == 0 else None,
            marker='.',
            alpha=0.3,
            # size=20,
            **kwargs
        )
        sc.pl.embedding(
            adata[adata.obs['mask_indices'] == 1], # test set have different markers
            ax=axes[1, i],
            basis=method,
            color=label_key,
            title=f"{method_name}", # {label_key}
            show=False,
            legend_loc= None,
            marker=marker_test,
            size=size_test,
            **kwargs
        )

    from matplotlib.lines import Line2D   
    # 1. Create the custom "Proxy" markers
    # We use color='black' or 'gray' to keep them neutral
    train_handle = Line2D([0], [0], marker='.', color='none', 
                        markerfacecolor='gray', alpha=0.3, 
                        label='Training Points', markersize=8)

    test_handle = Line2D([0], [0], marker='*', color='none', 
                        markerfacecolor='gray',  # or 'none' if you want edge only
                        markeredgecolor='gray',
                        label='Test Points', markersize=10)

    # ---- EXTRACT + MOVE LEGENDS ----
    leg_top = axes[0, 0].get_legend()
    leg_bottom = axes[1, 0].get_legend()

    # 2. Update the handles (This adds the new markers to the bottom of the lists)
    # You can do this for one or both legends
    handles_top, labels_top = leg_top.legend_handles, [t.get_text() for t in leg_top.get_texts()]
    handles_top.extend([train_handle, test_handle])
    labels_top.extend(["Training", "Test"])
    handles_bottom, labels_bottom = leg_bottom.legend_handles, [t.get_text() for t in leg_bottom.get_texts()]
    handles_bottom.extend([train_handle, test_handle])
    labels_bottom.extend(["Training", "Test"])

    # Re-initialize the legend with the combined list
    leg_top = fig.legend(handles_top, labels_top)
    leg_bottom = fig.legend(handles_bottom, labels_bottom)

    # remove them from axes
    axes[0, 0].legend_.remove()
    axes[1, 0].legend_.remove()

    # IMPORTANT: anchor legends in FIGURE coordinates
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    leg_top.set_bbox_to_anchor((0.88, 0.75), transform=fig.transFigure)
    leg_bottom.set_bbox_to_anchor((0.88, 0.25), transform=fig.transFigure)
    leg_top.set_loc("center left")
    leg_bottom.set_loc("center left")

    # add them back to the figure
    fig.add_artist(leg_top)
    fig.add_artist(leg_bottom)

    # plt.show()

    # plt.title(f"Noise level: {noise_level}, Dropout level: {dropout_level}")
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f"{save_path}/all_methods_embeddings.png", bbox_inches='tight', format="png")
        plt.savefig(f"{save_path}/all_methods_embeddings.pdf", bbox_inches='tight', format="pdf")
        
    return adata