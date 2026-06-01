import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# --- CONFIGURATION ---
BASE_DIR = Path("/Users/aumona/Projects/RF-MALI/results_tree/20260528_202057")
SEED = "56089"
SELECTED_MODELS = ["FoSTA_tauto", "KEMArbf", "scANVI", "scVI", "MALI", "Unintegrated_PHATE"]

# --- GLOBAL PLOT CONTROL ---
PLOT_POINT_SIZE = 5
LEGEND_MARKER_SCALE = 5.0
LEGEND_Y_OFFSET = 0.08
LABEL_X_POS = 0.5
METHOD_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12
GRID_WSPACE = 0.02
GRID_HSPACE = 0.25
SCORE_FONT_SIZE = 10

BRANCH_LABELS = [str(i) for i in range(1, 11)]
BRANCH_PALETTE = {
    "1": "#0072B2",
    "2": "#E69F00",
    "3": "#009E73",
    "4": "#D55E00",
    "5": "#CC79A7",
    "6": "#D69C62",
    "7": "#F0A3D8",
    "8": "#999999",
    "9": "#F0E442",
    "10": "#56B4E9",
}


def load_embedding(seed_path, model_key):
    emb_path = seed_path / model_key / f"{model_key}_embedding.npz"
    if not emb_path.exists():
        print(f"Missing embedding: {emb_path}")
        return None
    return np.load(emb_path)["embedding"]


def load_metrics(base_dir, seed):
    results_path = Path(base_dir) / "tree_alignment_results.csv"
    if not results_path.exists():
        print(f"Missing metrics CSV: {results_path}")
        return {}

    df = pd.read_csv(results_path)
    df = df[df["seed"].astype(str) == str(seed)]
    return {row["method"]: row for _, row in df.iterrows()}


def format_scores(metric_row):
    if metric_row is None:
        return "Scores unavailable"

    return (
        f"DeMAP={metric_row['DeMAP']:.3f} | "
        f"Acc={metric_row['label_transfer']:.3f} | "
        f"AS={metric_row['alignment_score']:.3f} | "
        f"FOSCTTM={metric_row['FOSCTTM']:.3f}"
    )


def plot_embedding_grid(base_dir=BASE_DIR, seed=SEED, selected_models=SELECTED_MODELS):
    seed_path = Path(base_dir) / f"seed_{seed}"
    inputs_path = seed_path / "paired_tree_inputs.npz"

    if not inputs_path.exists():
        print(f"Error: {inputs_path} not found.")
        return

    labels_a = (np.load(inputs_path)["labels_a"].astype(int) + 1).astype(str)
    labels = np.concatenate([labels_a, labels_a])
    batches = np.array(["A"] * len(labels_a) + ["B"] * len(labels_a))
    metrics_by_method = load_metrics(base_dir, seed)

    num_models = len(selected_models)
    num_rows = (num_models + 1) // 2
    fig, axes = plt.subplots(
        num_rows,
        4,
        figsize=(16, 4.5 * num_rows),
        gridspec_kw={"wspace": GRID_WSPACE, "hspace": GRID_HSPACE},
    )
    axes = np.atleast_1d(axes).flatten()

    batch_handles, batch_labels = [], []
    label_handles, label_labels = [], []

    for i, model_key in enumerate(selected_models):
        coords = load_embedding(seed_path, model_key)
        if coords is None:
            axes[i * 2].axis("off")
            axes[i * 2 + 1].axis("off")
            continue

        coords = coords[:, :2]
        display_name = "FoSTA" if model_key.startswith("FoSTA") else model_key

        ax_batch = axes[i * 2]
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=batches,
            s=PLOT_POINT_SIZE,
            palette="tab10",
            ax=ax_batch,
            edgecolor=None,
            rasterized=True,
        )

        ax_label = axes[i * 2 + 1]
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            s=PLOT_POINT_SIZE,
            palette=BRANCH_PALETTE,
            hue_order=BRANCH_LABELS,
            ax=ax_label,
            edgecolor=None,
            rasterized=True,
        )

        for ax in [ax_batch, ax_label]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            for spine in ax.spines.values():
                spine.set_visible(False)

            handles, legend_labels = ax.get_legend_handles_labels()
            if i == 0:
                if ax == ax_batch:
                    batch_handles, batch_labels = handles, legend_labels
                if ax == ax_label:
                    label_handles, label_labels = handles, legend_labels
            if ax.get_legend():
                ax.get_legend().remove()

        is_fosta = display_name == "FoSTA"
        pos_batch = ax_batch.get_position()
        pos_label = ax_label.get_position()
        center_x = (pos_batch.x0 + pos_label.x1) / 2
        fig.text(
            center_x,
            pos_batch.y1 + 0.028,
            display_name,
            ha="center",
            va="bottom",
            fontsize=METHOD_FONT_SIZE + 2 if is_fosta else METHOD_FONT_SIZE,
            fontweight="bold" if is_fosta else "normal",
        )
        fig.text(
            center_x,
            pos_batch.y1 + 0.009,
            format_scores(metrics_by_method.get(model_key)),
            ha="center",
            va="bottom",
            fontsize=SCORE_FONT_SIZE,
        )

    for j in range(len(selected_models) * 2, len(axes)):
        axes[j].axis("off")

    if batch_handles:
        fig.legend(
            batch_handles,
            batch_labels,
            loc="upper center",
            bbox_to_anchor=(0.25, LEGEND_Y_OFFSET),
            title="Batches",
            ncol=2,
            frameon=False,
            markerscale=LEGEND_MARKER_SCALE,
            prop={"size": LEGEND_FONT_SIZE},
            title_fontsize=LEGEND_FONT_SIZE + 2,
        )

    if label_handles:
        fig.legend(
            label_handles,
            label_labels,
            loc="upper left",
            bbox_to_anchor=(LABEL_X_POS, LEGEND_Y_OFFSET),
            title="Ground Truth",
            ncol=5,
            frameon=False,
            markerscale=LEGEND_MARKER_SCALE,
            prop={"size": LEGEND_FONT_SIZE},
            title_fontsize=LEGEND_FONT_SIZE + 2,
        )

    save_path = seed_path / "embedding_grid_scaled.pdf"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--models", nargs="+", default=SELECTED_MODELS)
    args = parser.parse_args()
    plot_embedding_grid(args.base_dir, args.seed, args.models)


if __name__ == "__main__":
    main()
