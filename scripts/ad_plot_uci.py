import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================
# results_csv = "results_uci/results_20260401_190636.csv"   # change if needed
# results_csv = "results_uci/results_20260402_013729.csv"   # change if needed
results_csv = "results_uci/results_20260428_045940.csv"   # change if needed



out_png = "results_uci/multimodal_ablation_fosta.png"

metrics_to_plot = [
    ("label_transfer", "Label transfer"),
    ("alignment_score", "Alignment score"),
    ("foscttm", "FOSCTTM"),
]

lower_is_better = {"foscttm"}

split_order = [
    "add_gaussian_noise_features",
    "random",
    "importance",
    "alternate_importance",
    "rotate",
    "distort",
]

method_order = [

    "FoSTA_gap",
    "FoSTA_gap_tsem=auto",


    "FoSTA_gap_tsem=auto_avg",
    
    "FoSTA_kerf",
    "FoSTA_kerf_tsem=auto",


    "FoSTA_kerf_tsem=auto_avg",

    "MALI",
    "MALI_nodpt",
    # "Pamona",
    "KEMAlin",
    "KEMArbf",
]

method_display_map = {

    # "FoSTA_oob": "FoSTA oob",
    # "FoSTA_oob_dense": "FoSTA oob dense",
    # "FoSTA_orig": "FoSTA original",
    # "FoSTA_orig_dense": "FoSTA original dense",

    "FoSTA_gap": "FoSTA gap",
    "FoSTA_gap_tsem=auto": "FoSTA gap t_sem=auto",
    "FoSTA_gap_tsem=auto_avg": "FoSTA gap t_sem=auto avg",


    "FoSTA_kerf": "FoSTA kerf",
    "FoSTA_kerf_tsem=auto": "FoSTA kerf t_sem=auto",
    "FoSTA_kerf_tsem=auto_avg": "FoSTA kerf t_sem=auto avg",

    "MALI": "MALI",
    "MALI_nodpt": "MALI no dpt",
    "Pamona": "Pamona",
    "KEMAlin": "KEMA lin",
    "KEMArbf": "KEMA rbf",
}

# Same color for FoSTA family, distinguished by hatches
method_style = {
    # "FoSTA_oob": {
    #     "facecolor": "#4C78A8",
    #     "edgecolor": "black",
    #     "hatch": "",
    #     "linewidth": 1.0,
    # },
    # "FoSTA_oob_dense": {
    #     "facecolor": "#4C78A8",
    #     "edgecolor": "black",
    #     "hatch": "o",
    #     "linewidth": 1.0,
    # },
    # "FoSTA_orig": {
    #     "facecolor": "#4C78A8",
    #     "edgecolor": "black",
    #     "hatch": "//",
    #     "linewidth": 1.0,
    # },
    # "FoSTA_orig_dense": {
    #     "facecolor": "#4C78A8",
    #     "edgecolor": "black",
    #     "hatch": "\\",
    #     "linewidth": 1.0,
    # },


    "FoSTA_gap": {
        "facecolor": "#4C78A8",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
    "FoSTA_gap_tsem=auto": {
        "facecolor": "#4C78A8",
        "edgecolor": "black",
        "hatch": "//",
        "linewidth": 1.0,
    },
    "FoSTA_gap_tsem=auto_avg": {
        "facecolor": "#4C78A8",
        "edgecolor": "black",
        "hatch": "oo",
        "linewidth": 1.0,
    },



    "FoSTA_kerf": {
        "facecolor": "#A0F518",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
    "FoSTA_kerf_tsem=auto": {
        "facecolor": "#A0F518",
        "edgecolor": "black",
        "hatch": "//",
        "linewidth": 1.0,
    },
    "FoSTA_kerf_tsem=auto_avg": {
        "facecolor": "#A0F518",
        "edgecolor": "black",
        "hatch": "oo",
        "linewidth": 1.0,
    },




    "MALI": {
        "facecolor": "#F58518",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
    "MALI_nodpt": {
        "facecolor": "#F58518",
        "edgecolor": "black",
        "hatch": "--",
        "linewidth": 1.0,
    },
    "Pamona": {
        "facecolor": "#54A24B",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
    "KEMAlin": {
        "facecolor": "#E45756",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
    "KEMArbf": {
        "facecolor": "#B279A2",
        "edgecolor": "black",
        "hatch": "",
        "linewidth": 1.0,
    },
}


# =========================================================
# HELPERS
# =========================================================
def prettify_split_name(s: str) -> str:
    return s.replace("_", "\n")


def compute_fair_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Fair rule:
    for each (dataset, split, seed), keep the combo only if all methods
    have a non-missing score for this metric.
    """
    x = df[["dataset", "split", "seed", "method", metric]].copy()
    x = x[x["method"].isin(method_order)].copy()

    piv = x.pivot_table(
        index=["dataset", "split", "seed"],
        columns="method",
        values=metric,
        aggfunc="first",
    )

    required_methods = [m for m in method_order if m in piv.columns]
    piv = piv[required_methods]

    valid_idx = piv.index[piv.notna().all(axis=1)]
    x = x.set_index(["dataset", "split", "seed"]).loc[valid_idx].reset_index()

    out = (
        x.groupby(["split", "method"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return out


# =========================================================
# LOAD RESULTS
# =========================================================
df = pd.read_csv(results_csv)

if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()

df = df[df["method"].isin(method_order)].copy()
df = df[df["split"].isin(split_order)].copy()

if df.empty:
    raise ValueError("No usable rows found in the results file.")


# =========================================================
# SUMMARIES
# =========================================================
summary_by_metric = {}
for metric, _ in metrics_to_plot:
    summary_by_metric[metric] = compute_fair_summary(df, metric)

available_methods = [
    m for m in method_order
    if any((summary_by_metric[metric]["method"] == m).any() for metric, _ in metrics_to_plot)
]

available_splits = [
    s for s in split_order
    if any((summary_by_metric[metric]["split"] == s).any() for metric, _ in metrics_to_plot)
]

if len(available_methods) == 0 or len(available_splits) == 0:
    raise ValueError("No complete split/method combinations remained after fair filtering.")


# =========================================================
# PLOT
# =========================================================
n_metrics = len(metrics_to_plot)
fig, axes = plt.subplots(
    n_metrics,
    1,
    figsize=(16, 4.6 * n_metrics),
    sharex=True,
)

if n_metrics == 1:
    axes = [axes]

x_centers = np.arange(len(available_splits))
n_methods = len(available_methods)
group_width = 0.84
bar_width = group_width / n_methods

for ax, (metric, metric_title) in zip(axes, metrics_to_plot):
    summ = summary_by_metric[metric].copy()

    for i, method in enumerate(available_methods):
        sub = summ[summ["method"] == method].copy()
        sub = sub.set_index("split").reindex(available_splits).reset_index()

        xpos = x_centers - group_width / 2 + (i + 0.5) * bar_width
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std"].fillna(0.0).to_numpy(dtype=float)

        style = method_style[method]

        ax.bar(
            xpos,
            means,
            width=bar_width * 0.95,
            yerr=stds,
            capsize=3,
            label=method_display_map.get(method, method),
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=style["linewidth"],
        )

    ax.set_title(metric_title, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel("Lower" if metric in lower_is_better else "Higher")

axes[-1].set_xticks(x_centers)
axes[-1].set_xticklabels([prettify_split_name(s) for s in available_splits], fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=min(len(labels), 4),
    frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)

fig.suptitle("Alignment performance across simulated multimodal splits", fontsize=16, y=1.05)
plt.tight_layout(rect=[0, 0, 1, 0.96])

Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved PNG to: {out_png}")