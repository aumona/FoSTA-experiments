# compile_mask_fraction_ablation.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

RESULTS_CSV = Path("/NOBACKUP/aumona/projects/RF-MALI/results_uci/results_20260502_204412.csv")
OUT_DIR = Path("/NOBACKUP/aumona/projects/RF-MALI/results_uci/mask_fraction_plots")

SELECTED_METHODS = [
    "FoSTA_gap_t2",
    "FoSTA_gap_auto",
    "FoSTA_gap_mauto_t2",
    "FoSTA_gap_mauto_auto",
    "FoSTA_kerf_t2",
    "FoSTA_kerf_auto",
    "FoSTA_kerf_mauto_t2",
    "FoSTA_kerf_mauto_auto",
    "MALI",
    "MALI_nodpt",
    "Pamona",
    "KEMAlin",
    "KEMArbf",
]

METHOD_DISPLAY_NAMES = {
    "FoSTA_gap_t2": "FoSTA (Gap, t2)",
    "FoSTA_gap_auto": "FoSTA (Gap, auto)",
    "FoSTA_gap_mauto_t2": "FoSTA (Gap, mauto, t2)",
    "FoSTA_gap_mauto_auto": "FoSTA (Gap, mauto, auto)",
    "FoSTA_kerf_t2": "FoSTA (Kerf, t2)",
    "FoSTA_kerf_auto": "FoSTA (Kerf, auto)",
    "FoSTA_kerf_mauto_t2": "FoSTA (Kerf, mauto, t2)",
    "FoSTA_kerf_mauto_auto": "FoSTA (Kerf, mauto, auto)",
    "MALI": "MALI",
    "MALI_nodpt": "MALI w/o DPT",
    "Pamona": "Pamona",
    "KEMAlin": r"KEMA$_{\mathrm{lin}}$",
    "KEMArbf": r"KEMA$_{\mathrm{rbf}}$",
}


METHOD_COLORS = {
    "FoSTA_gap_t2": "#0072B2",   # blue
    "FoSTA_gap_auto": "#E69F00",    # orange
    "FoSTA_gap_mauto_t2": "#56B4E9",   # light blue
    "FoSTA_gap_mauto_auto": "#F0E442",   # yellow
    "FoSTA_kerf_t2": "#009E73",         # green
    "FoSTA_kerf_auto": "#D55E00",   # vermillion
    "FoSTA_kerf_mauto_t2": "#CC79A7",       # purple
    "FoSTA_kerf_mauto_auto": "#8C564B",      # brown
    "MALI": "#7F7F7F",      # gray
}

METHOD_STYLES = {
    "FoSTA_gap_t2": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_gap_auto": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_gap_mauto_t2": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_gap_mauto_auto": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_kerf_t2": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_kerf_auto": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_kerf_mauto_t2": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "FoSTA_kerf_mauto_auto": dict(linestyle="-", linewidth=3.2, alpha=1.0),
    "MALI": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "MALI_nodpt": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "Pamona": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "KEMAlin": dict(linestyle="--", linewidth=1.9, alpha=0.70),
    "KEMArbf": dict(linestyle="--", linewidth=1.9, alpha=0.70),
}

METRICS = {
    "label_transfer": "Accuracy",
    "alignment_score": "Alignment Score (AS)",
    "foscttm": "FOSCTTM",
}


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
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
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["status"] == "ok"].copy()
    df = df[df["method"].isin(SELECTED_METHODS)].copy()
    df["mask_fraction"] = pd.to_numeric(df["mask_fraction"], errors="coerce")
    df = df.dropna(subset=["mask_fraction"])

    return df


def aggregate_by_mask_fraction(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Average over seeds within each dataset/split/mask/method.
    In the grid figure, this function is called separately for each split,
    then averaged across datasets.
    """
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
    rows = []

    for split in sorted(df["split"].dropna().unique()):
        df_split = df[df["split"] == split].copy()

        for metric, metric_name in METRICS.items():
            agg = aggregate_by_mask_fraction(df_split, metric)
            agg["split"] = split
            agg["metric"] = metric
            agg["metric_name"] = metric_name
            rows.append(agg)

    summary = pd.concat(rows, axis=0, ignore_index=True)
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

    n_rows = len(metrics)
    n_cols = len(splits)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.15 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for r, (metric, metric_name) in enumerate(metrics):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            df_split = df[df["split"] == split].copy()

            if df_split.empty:
                ax.axis("off")
                continue

            agg = aggregate_by_mask_fraction(df_split, metric)

            for method in SELECTED_METHODS:
                sub = agg[agg["method"] == method].copy()
                if sub.empty:
                    continue

                x = sub["mask_fraction"].to_numpy(float)
                y = sub["mean"].to_numpy(float)

                order = np.argsort(x)
                x = x[order]
                y = y[order]

                label = METHOD_DISPLAY_NAMES.get(method, method)
                style = METHOD_STYLES.get(
                    method,
                    dict(linestyle="-", linewidth=2.0, alpha=0.8),
                )

                ax.plot(
                    x,
                    y,
                    label=label,
                    color=METHOD_COLORS.get(method, None),
                    **style,
                )

            if r == 0:
                ax.set_title(split, fontsize=11)

            if c == 0:
                ax.set_ylabel(metric_name, fontsize=11)

            if r == n_rows - 1:
                ax.set_xlabel("Masked target label fraction", fontsize=10)

            ax.set_xticks(mask_ticks)
            ax.set_xticklabels([f"{x:g}" for x in mask_ticks])

            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", which="both", length=0, labelsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.015),
        columnspacing=1.4,
        handlelength=2.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = out_dir / "mask_fraction_metric_split_grid.png"
    out_pdf = out_dir / "mask_fraction_metric_split_grid.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    df = load_results(RESULTS_CSV)
    if df.empty:
        raise ValueError("No successful rows found for the selected methods.")

    save_compiled_summary(df, OUT_DIR)
    plot_metric_split_grid(df, OUT_DIR)


if __name__ == "__main__":
    main()