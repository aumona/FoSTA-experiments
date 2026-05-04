# compile_mask_fraction_ablation.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# =============================================================================
# CONFIG
# =============================================================================

# GLOBAL FONTSIZE OPTION
GLOBAL_FONTSIZE = 8
TICK_FONTSIZE = GLOBAL_FONTSIZE - 1

NEURIPS_TEXT_WIDTH_PT = 397.48499
PT_PER_INCH = 72.27
FIG_WIDTH = NEURIPS_TEXT_WIDTH_PT / PT_PER_INCH  # ≈ 5.50 inches

plt.rcParams.update({
    "font.size": GLOBAL_FONTSIZE,
    "axes.titlesize": GLOBAL_FONTSIZE,
    "axes.labelsize": GLOBAL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": GLOBAL_FONTSIZE,
})

RESULTS_CSV = Path("/NOBACKUP/aumona/projects/RF-MALI/results_uci/results_20260503_171832.csv")
OUT_DIR = Path("/NOBACKUP/aumona/projects/RF-MALI/results_uci/mask_fraction_plots")

SELECTED_METHODS = [
    "FoSTA_gap_auto",
    "MALI",
    "Pamona",
    "KEMAlin",
    "KEMArbf",
]

METHOD_DISPLAY_NAMES = {
    "FoSTA_gap_auto": "FoSTA",
    "MALI": "MALI",
    "Pamona": "Pamona",
    "KEMAlin": "KEMAlin",
    "KEMArbf": "KEMArbf",
}

METHOD_COLORS = {
    "FoSTA_gap_auto": "#E69F00",  # orange
    "MALI": "#7F7F7F",            # gray
}

METHOD_STYLES = {
    "FoSTA_gap_auto": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "MALI": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "Pamona": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "KEMAlin": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "KEMArbf": dict(linestyle="--", linewidth=1.9, alpha=0.70),
}

DEFAULT_STYLE = dict(linestyle="-", linewidth=2.0, alpha=0.8)

METRICS = {
    "label_transfer": "Accuracy",
    "alignment_score": "AS",
    "foscttm": "FOSCTTM",
}

REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "split",
    "mask_fraction",
    "method",
    "label_transfer",
    "alignment_score",
    "foscttm",
    "status",
}

SPLIT_DISPLAY_NAMES = {
    "add_gaussian_noise_features": "Noise",
    "alternate_importance": "Alt. Importance",
    "distort": "Distort",
    "importance": "Importance",
    "random": "Random",
    "rotate": "Rotate",
}


# =============================================================================
# DATA HELPERS (Unchanged logic)
# =============================================================================

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[(df["status"] == "ok") & (df["method"].isin(SELECTED_METHODS))].copy()
    df["mask_fraction"] = pd.to_numeric(df["mask_fraction"], errors="coerce")
    df = df.dropna(subset=["mask_fraction"])
    return df

def aggregate_by_mask_fraction(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    per_dataset = (
        df.groupby(["dataset", "method", "mask_fraction"], dropna=False)[metric]
        .mean()
        .reset_index(name="dataset_mean")
    )
    agg = (
        per_dataset.groupby(["method", "mask_fraction"], dropna=False)["dataset_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values(["method", "mask_fraction"])
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg

def save_compiled_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summaries = []
    for split, df_split in df.groupby("split", dropna=True):
        for metric, metric_name in METRICS.items():
            agg = aggregate_by_mask_fraction(df_split, metric)
            agg["split"] = split
            agg["metric"] = metric
            agg["metric_name"] = metric_name
            summaries.append(agg)
    summary = pd.concat(summaries, axis=0, ignore_index=True)
    summary_path = out_dir / "mask_fraction_compiled_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


# =============================================================================
# PLOTTING
# =============================================================================
def plot_metric_split_grid(df: pd.DataFrame, out_dir: Path) -> None:
    splits = sorted(df["split"].dropna().unique())
    metrics = list(METRICS.items())
    mask_ticks = np.sort(df["mask_fraction"].dropna().unique())

    fig, axes = plt.subplots(
        len(splits),
        len(metrics),
        figsize=(FIG_WIDTH, 8.8),
        sharex=True,
        squeeze=False,
    )

    for r, split in enumerate(splits):
        df_split = df[df["split"] == split]

        for c, (metric, metric_name) in enumerate(metrics):
            ax = axes[r, c]

            if df_split.empty:
                ax.axis("off")
                continue

            agg = aggregate_by_mask_fraction(df_split, metric)

            for method in SELECTED_METHODS:
                sub = agg[agg["method"] == method].sort_values("mask_fraction")
                if sub.empty:
                    continue

                ax.plot(
                    sub["mask_fraction"].to_numpy(float),
                    sub["mean"].to_numpy(float),
                    label=METHOD_DISPLAY_NAMES.get(method, method),
                    color=METHOD_COLORS.get(method),
                    **METHOD_STYLES.get(method, DEFAULT_STYLE),
                )

            # Titles only on the top row
            if r == 0:
                ax.set_title(metric_name, fontsize=GLOBAL_FONTSIZE)

            # Y-Labels only on the first column
            if c == 0:
                ax.set_ylabel(SPLIT_DISPLAY_NAMES.get(split, split), fontsize=GLOBAL_FONTSIZE)

            # X-Labels only on the bottom row
            if r == len(splits) - 1:
                ax.set_xlabel("% Masked Labels", fontsize=GLOBAL_FONTSIZE)

            ax.set_xticks(mask_ticks)
            ax.set_xticklabels([f"{x:g}" for x in mask_ticks])
            ax.grid(True, alpha=0.25)
            
            # Ticks explicitly use GLOBAL_FONTSIZE - 1
            ax.tick_params(axis="both", which="both", length=0, labelsize=TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        fontsize=GLOBAL_FONTSIZE,
        bbox_to_anchor=(0.5, 1.015),
        columnspacing=1.4,
        handlelength=2.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.99])

    for ext, kwargs in {
        "png": dict(dpi=300, bbox_inches="tight"),
        "pdf": dict(bbox_inches="tight"),
    }.items():
        out_path = out_dir / f"mask_fraction_metric_split_grid.{ext}"
        fig.savefig(out_path, **kwargs)
        print(f"Saved: {out_path}")

    plt.close(fig)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results(RESULTS_CSV)
    if df.empty:
        raise ValueError("No successful rows found for the selected methods.")
    save_compiled_summary(df, OUT_DIR)
    plot_metric_split_grid(df, OUT_DIR)

if __name__ == "__main__":
    main()