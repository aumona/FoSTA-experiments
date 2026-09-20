import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

# --- CONFIGURATION ---
SELECTED_TIMESTAMPS = [
    "./results_sc_experiments/20260919_185500_1to6_20",
    "./results_sc_experiments/20260920_105524_B1toB4_20",

    # "./results_sc_experiments/20260920_033154_1to6_50",
    # "./results_sc_experiments/20260920_100839_B1toB4_50"

]

EXACT_SYMBOL_MAP = {
    "scANVI": r"$\beta$", "scVI": r"$\gamma$", "KEMArbf": r"$\epsilon$",
    "FoSTA": r"$f$", "Scanorama": r"$\alpha$",
    "Unintegrated": r"$0$", "MALI": r"$\zeta$", "KEMAlin": r"$\delta$",
    "LIGER": r"$K$", "Pamona": r"$\eta$"
}

EXPECTED_COLORS = {
    "scANVI": "#31a354", "scVI": "#a1d99b", "KEMArbf": "#fb9a99",
    "FoSTA": "#a6cee3", "Scanorama": "#fdbf6f",
    "Unintegrated": "#1f78b4", "MALI": "#9467bd", "KEMAlin": "#e31a1c",
    "LIGER": "#ff7f00", "Pamona": "#cab2d6"
}

def get_full_script():
    df_list = []
    
    for base_dir in SELECTED_TIMESTAMPS:
        if not os.path.exists(base_dir): continue
        
        found_files = glob.glob(os.path.join(base_dir, "**/benchmark_metrics_seed*.csv"), recursive=True)
        
        for f in found_files:
            try:
                # 1. Load CSV
                temp_df = pd.read_csv(f, index_col=0)
                
                # 2. Clean columns and index
                cols_to_drop = [c for c in temp_df.columns if "mean" in c]
                temp_df = temp_df.drop(columns=cols_to_drop)
                temp_df.index = temp_df.index.str.strip()
                
                # 3. RENAMING STEP: Change FoSTA_t2 to FoSTA
                temp_df.index = temp_df.index.map(lambda x: "FoSTA" if x == "FoSTA_t2" else x)
                
                temp_df = temp_df.reset_index().rename(columns={'index': 'Method'})
                
                # Ensure metrics are numeric
                temp_df["Bio conservation"] = pd.to_numeric(temp_df["Bio conservation"], errors='coerce')
                temp_df["Batch correction"] = pd.to_numeric(temp_df["Batch correction"], errors='coerce')
                
                df_list.append(temp_df[["Method", "Bio conservation", "Batch correction"]])
            except Exception as e:
                print(f"Skipping {f}: {e}")

    if not df_list:
        print("No data found.")
        return

    # Every loaded result contributes equally, including repeated runs.
    full_df = pd.concat(df_list, ignore_index=True)

    # --- AGGREGATION ---
    mean_df = full_df.groupby("Method")[["Bio conservation", "Batch correction"]].mean()
    std_df = full_df.groupby("Method")[["Bio conservation", "Batch correction"]].std().fillna(0)

    # --- RANKING BY AVERAGE RANK ---
    mean_df['Bio_Rank'] = mean_df['Bio conservation'].rank(ascending=False, method='min').astype(int)
    mean_df['Batch_Rank'] = mean_df['Batch correction'].rank(ascending=False, method='min').astype(int)
    
    mean_df['Avg_Rank'] = (mean_df['Bio_Rank'] + mean_df['Batch_Rank']) / 2
    mean_df = mean_df.sort_values(['Avg_Rank', 'Bio_Rank'], ascending=True)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(7, 7))
    legend_handles, legend_labels = [], []

    for method in mean_df.index:
        x, y = mean_df.loc[method, "Batch correction"], mean_df.loc[method, "Bio conservation"]
        x_err, y_err = std_df.loc[method, "Batch correction"], std_df.loc[method, "Bio conservation"]
        
        color = EXPECTED_COLORS.get(method, "#7f7f7f") 
        symbol = EXACT_SYMBOL_MAP.get(method, r"$\star$")
        label = f"{method} (Bio #{int(mean_df.loc[method, 'Bio_Rank'])}, Batch #{int(mean_df.loc[method, 'Batch_Rank'])})"

        # Background Fade Error Bars
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', ecolor=color, elinewidth=1.5, capsize=4, alpha=0.3, zorder=1)
        
        # Method Markers
        c = ax.scatter(x, y, s=300, color=color, marker='o', zorder=3)
        s = ax.scatter(x, y, s=100, color="white", marker=symbol, zorder=4)

        legend_handles.append((c, s))
        legend_labels.append(label)

    # Square Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Bio conservation vs Batch correction (Lung 2D)", fontsize=14)
    ax.set_xlabel("Batch correction", fontsize=12)
    ax.set_ylabel("Bio conservation", fontsize=12)

    # Symmetrical limits
    all_vals = np.concatenate([mean_df["Batch correction"].values, mean_df["Bio conservation"].values])
    limit_min, limit_max = all_vals.min() - 0.05, all_vals.max() + 0.05
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)

    lgd = ax.legend(handles=legend_handles, labels=legend_labels,
                    title="Ordered by average Bio/Batch rank",
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=-2)},
                    bbox_to_anchor=(1.05, 0.9), loc='upper left', frameon=True)

    # Save as PDF
    save_path = os.path.join(SELECTED_TIMESTAMPS[-1], "formatted_tradeoff_plot.pdf")
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    print(f"Plot saved with average Bio/Batch rank sorting: {save_path}")
    plt.show()

if __name__ == "__main__":
    get_full_script()
