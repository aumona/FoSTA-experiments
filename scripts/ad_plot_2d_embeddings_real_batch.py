import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

# --- CONFIGURATION ---
BASE_DIR = "/Users/aumona/Projects/RF-MALI/results_sc_ad/20260506_233231"
BATCH_PAIR = "B2_vs_B3"
SEED = "56089"
SELECTED_MODELS = ["FoSTA_t2", "KEMArbf", "scANVI", "scVI", "MALI", "Unintegrated"] 

# --- GLOBAL PLOT CONTROL ---
PLOT_POINT_SIZE = 5    
LEGEND_MARKER_SCALE = 5.0 
LEGEND_Y_OFFSET = 0.08   
CELL_TYPE_X_POS = 0.5    

# --- NEW FONT CONTROL ---
METHOD_FONT_SIZE = 16    # Base size for method names
LEGEND_FONT_SIZE = 12    # Font size for legend labels and titles
# ------------------------

# --- GRID TIGHTENING ---
GRID_WSPACE = 0.02       
GRID_HSPACE = 0.25       
# ----------------------------

def plot_embedding_grid_optimized():
    run_path = os.path.join(BASE_DIR, f"seed_{SEED}", BATCH_PAIR)
    h5ad_path = os.path.join(run_path, "adata_final.h5ad")
    
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return

    adata = sc.read_h5ad(h5ad_path)

    num_models = len(SELECTED_MODELS)
    num_rows = (num_models + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4.5 * num_rows), 
                             gridspec_kw={'wspace': GRID_WSPACE, 'hspace': GRID_HSPACE})
    axes = axes.flatten()

    batch_handles, cell_handles = [], []
    batch_labels, cell_labels = [], []

    for i, model_key in enumerate(SELECTED_MODELS):
        display_name = "FoSTA" if model_key == "FoSTA_t2" else model_key
        
        if model_key not in adata.obsm:
            continue

        coords = adata.obsm[model_key]

        # --- Plot A: Batch ---
        ax_batch = axes[i * 2]
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=adata.obs['batch'], 
                        s=PLOT_POINT_SIZE, palette="tab10", ax=ax_batch, 
                        edgecolor=None, rasterized=True)
        
        # --- Plot B: Cell Type ---
        ax_cell = axes[i * 2 + 1]
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=adata.obs['cell_type'], 
                        s=PLOT_POINT_SIZE, palette="tab20", ax=ax_cell, 
                        edgecolor=None, rasterized=True)

        for ax in [ax_batch, ax_cell]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
            for spine in ax.spines.values(): spine.set_visible(False)
            
            h, l = ax.get_legend_handles_labels()
            if i == 0:
                if ax == ax_batch: batch_handles, batch_labels = h, l
                if ax == ax_cell: cell_handles, cell_labels = h, l
            if ax.get_legend(): ax.get_legend().remove()

        # --- Conditional Bolding & Font Scaling ---
        is_fosta = (display_name == "FoSTA")
        weight = 'bold' if is_fosta else 'normal'
        # FoSTA stays slightly larger than the base method font size
        current_size = METHOD_FONT_SIZE + 2 if is_fosta else METHOD_FONT_SIZE

        pos_batch = ax_batch.get_position()
        pos_cell = ax_cell.get_position()
        center_x = (pos_batch.x0 + pos_cell.x1) / 2
        
        fig.text(center_x, pos_batch.y1 + 0.01, 
                 display_name, ha='center', va='bottom', 
                 fontsize=current_size, fontweight=weight)

    for j in range(len(SELECTED_MODELS) * 2, len(axes)):
        axes[j].axis('off')

    # Shared Legends with Font Control
    # 'prop' controls label size; 'title_fontsize' controls the title size
    if batch_handles:
        fig.legend(batch_handles, batch_labels, loc='upper center', 
                   bbox_to_anchor=(0.25, LEGEND_Y_OFFSET), title="Batches", ncol=5, 
                   frameon=False, markerscale=LEGEND_MARKER_SCALE,
                   prop={'size': LEGEND_FONT_SIZE}, title_fontsize=LEGEND_FONT_SIZE + 2)
                   
    if cell_handles:
        fig.legend(cell_handles, cell_labels, loc='upper left', 
                   bbox_to_anchor=(CELL_TYPE_X_POS, LEGEND_Y_OFFSET), title="Cell Types", ncol=4, 
                   frameon=False, markerscale=LEGEND_MARKER_SCALE,
                   prop={'size': LEGEND_FONT_SIZE}, title_fontsize=LEGEND_FONT_SIZE + 2)

    save_path = os.path.join(run_path, f"embedding_grid_scaled.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_embedding_grid_optimized()