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

RESULTS_CSV = Path("./results_uci/results_20260916_141907_mask_ablation.csv")
OUT_DIR = Path("./results_uci/mask_fraction_plots")

SELECTED_METHODS = [
    "FoSTA_gap_t2",
    "MALI_t2",
    "Pamona",
    "KEMAlin",
    "KEMArbf",
]

METHOD_DISPLAY_NAMES = {
    "FoSTA_gap_t2": "FoSTA",
    "MALI_t2": "MALI",
    "Pamona": "Pamona",
    "KEMAlin": "KEMAlin",
    "KEMArbf": "KEMArbf",
}

METHOD_COLORS = {
    "FoSTA_gap_t2": "#E69F00",  # orange
    "MALI_t2": "#7F7F7F",
    "Pamona": "#4C78A8",            # gray
    "KEMAlin": "#5DA5DA",            # blue
    "KEMArbf": "#FA8072",            # red
}

METHOD_STYLES = {
    "FoSTA_gap_t2": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "MALI_t2": dict(linestyle=":", linewidth=1.9, alpha=0.85),
    "Pamona": dict(linestyle="-.", linewidth=1.9, alpha=0.70),
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

# =============================================================================
# DATA HELPERS
# =============================================================================

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Historical MALI runs used the automatic diffusion-time setting.
    df["method"] = df["method"].replace({"MALI": "MALI_auto", "mali_auto": "MALI_auto", "mali_t2": "MALI_t2"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[(df["status"] == "ok") & (df["method"].isin(SELECTED_METHODS))].copy()
    df["mask_fraction"] = pd.to_numeric(df["mask_fraction"], errors="coerce")
    df = df.dropna(subset=["mask_fraction"])
    return df

def aggregate_by_mask_fraction(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Average seeds within each split, then weight splits equally per dataset.
    per_split = df.groupby(
        ["dataset", "split", "method", "mask_fraction"], dropna=False
    )[metric].mean()
    per_dataset = (
        per_split.groupby(level=["dataset", "method", "mask_fraction"], dropna=False)
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
    for metric, metric_name in METRICS.items():
        agg = aggregate_by_mask_fraction(df, metric)
        agg["split"] = "average"
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
def plot_metric_averages(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS.items())
    mask_ticks = np.sort(df["mask_fraction"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(FIG_WIDTH, 2.5), sharex=True, squeeze=False,
    )
    for c, (metric, metric_name) in enumerate(metrics):
        ax = axes[0, c]
        ax.set_box_aspect(1)
        agg = aggregate_by_mask_fraction(df, metric)
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
        ax.set_title(metric_name, fontsize=GLOBAL_FONTSIZE)
        ax.set_xlabel("% Masked Labels", fontsize=GLOBAL_FONTSIZE)
        ax.set_xticks(mask_ticks)
        ax.set_xticklabels([f"{x:g}" for x in mask_ticks])
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="both", length=0, labelsize=TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_top = max(
        ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
        for ax in axes.flat
    )
    legend_gap = 5 / 72 / fig.get_figheight()  # Five points above the subplot titles.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
        fontsize=GLOBAL_FONTSIZE,
        bbox_to_anchor=(0.5, title_top + legend_gap),
        borderaxespad=0,
        columnspacing=1.4,
        handlelength=2.5,
    )

    for ext, kwargs in {
        "png": dict(dpi=300, bbox_inches="tight"),
        "pdf": dict(bbox_inches="tight"),
    }.items():
        out_path = out_dir / f"mask_fraction_metric_averages.{ext}"
        fig.savefig(out_path, **kwargs)
        print(f"Saved: {out_path}")

    plt.close(fig)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results(RESULTS_CSV)
    if df.empty:
        raise ValueError("No successful rows found for the selected methods.")
    save_compiled_summary(df, OUT_DIR)
    plot_metric_averages(df, OUT_DIR)

if __name__ == "__main__":
    main()
