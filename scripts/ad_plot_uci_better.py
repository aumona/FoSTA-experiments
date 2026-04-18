import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
from scipy.stats import wilcoxon


# =========================================================
# CONFIG
# =========================================================
results_csv = "results_uci/results_20260401_190636.csv"   # change if needed
out_png = "results_uci/fosta_ablation_clean.png"

anchor_method = "FoSTA"
method_map = {
    "FoSTA_no_prior": "No prior",
    "RFMALI": "Entropic OT",
}
ablation_methods = list(method_map.keys())

metrics = [
    ("label_transfer", "Label transfer"),
    ("alignment_score", "Alignment score"),
    ("foscttm", "FOSCTTM"),
]

split_order = [
    "add_noise",
    "random",
    "importance",
    "alternate_importance",
    "rotate",
    "distort",
]

split_display = {
    "add_noise": "Add noise",
    "random": "Random",
    "importance": "Importance",
    "alternate_importance": "Alternating\nimportance",
    "rotate": "Rotate",
    "distort": "Distort",
}

higher_is_better = {
    "label_transfer": True,
    "alignment_score": True,
    "foscttm": False,
}

alpha = 0.05


# =========================================================
# HELPERS
# =========================================================
def fair_filter_for_metric(df: pd.DataFrame, metric: str, methods_needed: list[str]) -> pd.DataFrame:
    sub = df[["dataset", "split", "seed", "method", metric]].copy()
    sub = sub[sub["method"].isin(methods_needed)].copy()

    piv = sub.pivot_table(
        index=["dataset", "split", "seed"],
        columns="method",
        values=metric,
        aggfunc="first",
    )

    present_cols = [m for m in methods_needed if m in piv.columns]
    piv = piv[present_cols]
    valid_idx = piv.index[piv.notna().all(axis=1)]

    return (
        sub.set_index(["dataset", "split", "seed"])
        .loc[valid_idx]
        .reset_index()
    )


def one_sided_wilcoxon(diffs: np.ndarray, direction: str) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    if len(diffs) < 2:
        return np.nan
    if np.allclose(diffs, 0):
        return 1.0

    try:
        res = wilcoxon(diffs, alternative=direction, zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return np.nan


def compute_raw_diff(sub_piv: pd.DataFrame, method: str) -> np.ndarray:
    return sub_piv[method].to_numpy(dtype=float) - sub_piv[anchor_method].to_numpy(dtype=float)


def favorable_unfavorable_pvalues(diffs: np.ndarray, metric: str):
    if higher_is_better[metric]:
        p_favorable = one_sided_wilcoxon(diffs, "greater")
        p_unfavorable = one_sided_wilcoxon(diffs, "less")
    else:
        p_favorable = one_sided_wilcoxon(diffs, "less")
        p_unfavorable = one_sided_wilcoxon(diffs, "greater")
    return p_favorable, p_unfavorable


# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(results_csv)
if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()

methods_needed = [anchor_method] + ablation_methods
df = df[df["method"].isin(methods_needed)].copy()
df = df[df["split"].isin(split_order)].copy()

if df.empty:
    raise ValueError("No usable FoSTA ablation rows found.")


# =========================================================
# BUILD SUMMARY
# =========================================================
rows = []

for metric, _ in metrics:
    fair_df = fair_filter_for_metric(df, metric, methods_needed)

    piv = fair_df.pivot_table(
        index=["dataset", "split", "seed"],
        columns="method",
        values=metric,
        aggfunc="first",
    )

    for method in ablation_methods:
        for split in split_order:
            sub = piv.reset_index()
            sub = sub[sub["split"] == split].copy()

            if sub.empty or method not in sub.columns or anchor_method not in sub.columns:
                rows.append({
                    "metric": metric,
                    "method": method,
                    "split": split,
                    "mean_diff": np.nan,
                    "state": "na",
                })
                continue

            diffs = compute_raw_diff(sub, method)
            diffs = diffs[np.isfinite(diffs)]

            if len(diffs) == 0:
                rows.append({
                    "metric": metric,
                    "method": method,
                    "split": split,
                    "mean_diff": np.nan,
                    "state": "na",
                })
                continue

            mean_diff = float(np.mean(diffs))
            p_favorable, p_unfavorable = favorable_unfavorable_pvalues(diffs, metric)

            if np.isfinite(p_favorable) and p_favorable < alpha:
                state = "favorable"
            elif np.isfinite(p_unfavorable) and p_unfavorable < alpha:
                state = "unfavorable"
            else:
                state = "ns"

            rows.append({
                "metric": metric,
                "method": method,
                "split": split,
                "mean_diff": mean_diff,
                "state": state,
            })

summary_df = pd.DataFrame(rows)


# =========================================================
# PLOT
# =========================================================
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(11.5, 7.6),
    gridspec_kw={"hspace": 0.7}
)

favorable_color = "#d9ead3"
unfavorable_color = "#f4cccc"
neutral_color = "white"
edge_color = "#333333"

for ax, (metric, panel_title) in zip(axes, metrics):
    panel = summary_df[summary_df["metric"] == metric].copy()

    ax.set_xlim(-0.5, len(split_order) - 0.5)
    ax.set_ylim(len(ablation_methods) - 0.5, -0.5)

    ax.set_xticks(np.arange(len(split_order)))
    ax.set_xticklabels([split_display[s] for s in split_order], fontsize=10)

    ax.set_yticks(np.arange(len(ablation_methods)))
    ax.set_yticklabels([method_map[m] for m in ablation_methods], fontsize=11)

    ax.set_title(panel_title, fontsize=12, pad=10)

    for i, method in enumerate(ablation_methods):
        for j, split in enumerate(split_order):
            row = panel[(panel["method"] == method) & (panel["split"] == split)].iloc[0]
            val = row["mean_diff"]
            state = row["state"]

            if state == "favorable":
                color = favorable_color
            elif state == "unfavorable":
                color = unfavorable_color
            else:
                color = neutral_color

            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=1.0,
            )
            ax.add_patch(rect)

            txt = "NA" if not np.isfinite(val) else f"{val:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Title block
fig.suptitle(
    "FoSTA ablation analysis relative to default FoSTA",
    fontsize=17,
    y=0.975,
)

fig.text(
    0.5,
    0.935,
    "Cell values report mean paired differences (ablation − FoSTA) across matched dataset-seed runs.",
    ha="center",
    va="center",
    fontsize=11,
)

fig.text(
    0.5,
    0.912,
    "Higher is better for Label transfer and Alignment score, whereas lower is better for FOSCTTM.",
    ha="center",
    va="center",
    fontsize=11,
)

# Legend block
legend_handles = [
    Patch(facecolor=favorable_color, edgecolor=edge_color, label="Significant in favorable direction"),
    Patch(facecolor=unfavorable_color, edgecolor=edge_color, label="Significant in unfavorable direction"),
    Patch(facecolor=neutral_color, edgecolor=edge_color, label="Not significant"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.06),   # lower than before
    fontsize=10.5,
)

fig.text(
    0.5,
    0.018,                        # lower footnote
    "One-sided Wilcoxon signed-rank test, α = 0.05.",
    ha="center",
    va="center",
    fontsize=10.5,
)

plt.subplots_adjust(
    top=0.84,
    bottom=0.22,                  # more room for bottom tick labels + legend
    left=0.18,
    right=0.98,
)

Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved PNG to: {out_png}")