import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
import seaborn as sns

from metrics import label_transfer_accuracy


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


# def plot_bio_vs_batch_correction(results_df, save_path=None):
#     ratio_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].mean()

#     sns.scatterplot(data=ratio_df, x="Batch correction", y="Bio conservation", hue=ratio_df.index, palette="tab20")
#     # plt.ylim(0, 1)
#     # plt.xlim(0, 1)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.title("Bio conservation vs Batch correction")
#     if save_path is not None:
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/bio_vs_batch_correction_ncomp_{n_components}.pdf", format='pdf')
#         plt.savefig(f"{save_path}/bio_vs_batch_correction_ncomp_{n_components}.png", format='png')
#     plt.show()



# def plot_bio_vs_batch_correction(results_df, save_path=None):
#     # 1. Calculate Mean and Standard Deviation
#     mean_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].mean()
#     std_df = results_df.groupby(['method'])[["Bio conservation", "Batch correction"]].std()
#     std_df = std_df.fillna(0)

#     # 2. Calculate Ranks (Assuming higher score is better -> ascending=False)
#     mean_df['Bio Rank'] = mean_df['Bio conservation'].rank(ascending=False).astype(int)
#     mean_df['Batch Rank'] = mean_df['Batch correction'].rank(ascending=False).astype(int)
    
#     # 3. Calculate Average Rank and Sort
#     mean_df['Avg Rank'] = (mean_df['Bio Rank'] + mean_df['Batch Rank']) / 2
#     mean_df = mean_df.sort_values('Avg Rank')

#     plt.figure(figsize=(10, 8)) 
    
#     # Use a fixed color map so colors stay consistent regardless of sorting
#     unique_methods = results_df['method'].unique()
#     # Create palette
#     palette_list = sns.color_palette("tab20", n_colors=len(unique_methods))
#     color_map = dict(zip(unique_methods, palette_list))
    
#     # Iterate through the SORTED methods
#     for method in mean_df.index:
#         x = mean_df.loc[method, "Batch correction"]
#         y = mean_df.loc[method, "Bio conservation"]
#         x_err = std_df.loc[method, "Batch correction"]
#         y_err = std_df.loc[method, "Bio conservation"]
        
#         bio_rank = mean_df.loc[method, 'Bio Rank']
#         batch_rank = mean_df.loc[method, 'Batch Rank']
#         avg_rank = mean_df.loc[method, 'Avg Rank']

#         # Legend Label: Includes Bio and Batch ranks
#         legend_label = f"{method} (Bio #{bio_rank}, Batch #{batch_rank})"

#         # 4. Plot Error Bars
#         plt.errorbar(
#             x, y, 
#             xerr=x_err, 
#             yerr=y_err, 
#             fmt='none',
#             ecolor=color_map[method],
#             elinewidth=1.5,
#             capsize=5,
#             alpha=0.4, 
#             label=None 
#         )

#         # 5. Plot the Marker (No annotation on plot)
#         plt.scatter(
#             x, y, 
#             s=100, 
#             color=color_map[method], 
#             label=legend_label, 
#             zorder=3
#         )

#     # 6. Final Formatting
#     plt.legend(
#         bbox_to_anchor=(1.05, 1), 
#         loc='upper left', 
#         title="Method (Ordered by Avg Rank)",
#         frameon=True
#     )
    
#     plt.title("Bio conservation vs Batch correction")
#     plt.xlabel("Batch correction")
#     plt.ylabel("Bio conservation")
#     plt.grid(True, linestyle='--', alpha=0.5)

#     if save_path is not None:
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/bio_vs_batch_correction.pdf", format='pdf')
#         plt.savefig(f"{save_path}/bio_vs_batch_correction.png", format='png')
#     plt.show()

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