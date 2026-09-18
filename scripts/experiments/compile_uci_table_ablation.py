import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================


# results_csv = "results_uci/results_20260503_015528_general.csv"

results_csv = "results_uci/results_20260915_143959_general_uci.csv"


# Set this to 1, 2, or 3 to control how many top ranks are colored
desired_top = 3

INCLUDE_STDS = True
TABLE_FONT_SIZE = r"\small"
METHOD_CELL_WIDTH = "1.5cm"

RANK_MACROS = {1: r"\gold", 2: r"\silver", 3: r"\bronze"}

metrics = {
    "label_transfer": "Acc",
    "alignment_score": "AS",
    "foscttm": "FOS"
}

lower_is_better = ["foscttm"]

split_order = [
    "add_gaussian_noise_features",
    "alternate_importance",
    "distort",
    "importance",
    "random",
    "rotate"
]

# split_order = ["distort"]


split_display_map = {
    "add_gaussian_noise_features": "Noise",
    "alternate_importance": "Alt.~Imp.",
    "distort": "Distort",
    "importance": "Imp.",
    "random": "Random",
    "rotate": "Rotate"
}

markdown_split_display_map = {
    "add_gaussian_noise_features": "Noise",
    "alternate_importance": "Alt. Imp.",
    "distort": "Distort",
    "importance": "Imp.",
    "random": "Random",
    "rotate": "Rotate",
}

# Exact list of columns, in display order. Comment out a method to hide it.
# Selected methods missing from the CSV appear as ---; no methods are auto-added.
method_order = [
    "FoSTA_gap_t2",  # default
    "FoSTA_rotf",
    "FoSTA_et",
    "FoSTA_kerf_t2",  # forest kernel
    # "FoSTA_kerf_auto",
    "FoSTA_dpt",
    "FoSTA_dense",
    "FoSTA_no_propag",
    "FoSTA_gap_auto",
    "FoSTA_umap",
    "FoSTA_spectral",
]
method_display_map = {
    "FoSTA_gap_t2": "Default", "FoSTA_gap_auto": "Auto. $t$",
    "FoSTA_kerf_t2": "KeRF", "FoSTA_kerf_auto": "KeRF-A",
    "FoSTA_umap": "UMAP", "FoSTA_dense": "Dense OT", "FoSTA_et": "ET",
    "FoSTA_spectral": "LE", "FoSTA_dpt": "DPT",
    "FoSTA_rotf": "RotF", "FoSTA_no_propag": "No Prop.",
}


def is_fosta(method):
    return str(method).lower().startswith("fosta")


def is_reference_baseline(method):
    name = str(method).lower()
    if name.startswith("mali"):
        return name == "mali_t2"
    return not is_fosta(method)


def select_variants(df):
    if not method_order:
        raise ValueError("Select at least one FoSTA variant in method_order.")
    if len(method_order) != len(set(method_order)):
        raise ValueError("method_order must not contain duplicate methods.")
    if any(not is_fosta(method) for method in method_order):
        raise ValueError("method_order must contain only FoSTA variants.")
    for method in method_order:
        method_display_map.setdefault(method, str(method).removeprefix("FoSTA_").replace("_", "-"))


# =========================================================
# HIGHLIGHTING LOGIC
# =========================================================
def get_rank(val, m_key, split_name, all_summaries):
    if pd.isna(val):
        return None
    summary = all_summaries[m_key]
    baselines = summary.loc[[is_reference_baseline(method) for method in summary.index], split_name].dropna()
    if baselines.empty:
        return None
    # Dense ranks: tied baseline scores count as one rank, as in the main UCI table.
    better = baselines[baselines < val] if m_key in lower_is_better else baselines[baselines > val]
    rank = better.nunique() + 1
    return rank if rank <= desired_top else None


def format_ranked_value(formatted: str, rank: int) -> str:
    rank_macro = RANK_MACROS[rank]
    highlighted = f"{rank_macro}{{{formatted}}}"
    if rank in (1, 2):
        # Share geometry instead of relying on differently sized external macros.
        fill = "[rgb]{1,0.95,0.8}" if rank == 1 else "[gray]{0.85}"
        highlighted = (
            f"\\colorbox{fill}{{"
            f"\\makebox[\\dimexpr {METHOD_CELL_WIDTH}-2\\fboxsep\\relax][c]"
            r"{\raisebox{0pt}[\ht\strutbox][\dp\strutbox]{"
            f"\\textcolor{{black}}{{\\textbf{{{formatted}}}}}}}}}}}"
        )
    return f"\\makebox[{METHOD_CELL_WIDTH}][c]{{{highlighted}}}"


def format_plain_value(formatted: str) -> str:
    return f"\\makebox[{METHOD_CELL_WIDTH}][c]{{{formatted}}}"


def format_mean_std(mean: float, std: float) -> str:
    if not INCLUDE_STDS:
        return f"{mean:.3f}"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}{{\\tiny $\\pm${std:.2f}}}"



def get_highlighted_value(val, m_key, split_name, all_summaries, std=0.0):
    if pd.isna(val):
        return "---"
    formatted = format_mean_std(val, std)
    rank = get_rank(val, m_key, split_name, all_summaries)
    if rank is not None:
        return format_ranked_value(formatted, rank)
    return format_plain_value(formatted)


def get_markdown_value(val, m_key, split_name, all_summaries):
    if pd.isna(val):
        return "---"
    formatted = f"{val:.3f}"
    rank = get_rank(val, m_key, split_name, all_summaries)
    if rank == 1:
        return f"***{formatted}***"
    if rank == 2:
        return f"**{formatted}**"
    if rank == 3:
        return f"*{formatted}*"
    return formatted


def markdown_method_name(method):
    return (
        method_display_map.get(method, method)
        .replace("$t=\\texttt{auto}$", "`t=auto`")
        .replace("$t=2$", "`t=2`")
        .replace("~", " ")
    )


# =========================================================
# PROCESSING
# =========================================================
def summarize_results(df):
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    # Preserve the ablation aggregation: average successful runs per method,
    # and report the mean within-dataset standard deviation across seeds.
    all_summaries, all_stds = {}, {}
    all_methods = list(dict.fromkeys([*method_order, *df["method"].dropna()]))
    for metric_key in metrics:
        all_summaries[metric_key] = (
            df.groupby(["split", "method"])[metric_key].mean().unstack(level=0)
            .reindex(index=all_methods, columns=split_order)
        )
        stds = df.groupby(["split", "method", "dataset"])[metric_key].std()
        all_stds[metric_key] = (
            stds.groupby(["split", "method"]).mean().unstack(level=0)
            .reindex(index=all_methods, columns=split_order)
        )
    return all_summaries, all_stds

# =========================================================
# TABLE GENERATION - METHODS AS COLUMNS
# =========================================================
def average_scores(all_summaries, all_stds):
    """Equally average displayed splits, requiring complete values for each method."""
    def average(values):
        return {
            key: summary.reindex(columns=split_order)
            .mean(axis=1, skipna=False).to_frame("average")
            for key, summary in values.items()
        }

    return average(all_summaries), average(all_stds)


def build_latex_table(all_summaries, all_stds):
    column_spec = "lr " + " ".join(["c"] * len(method_order))
    header = ["Split", "Metric"] + [
        format_plain_value(method_display_map[method]) for method in method_order
    ]
    lines = [
        r"% Requires \usepackage{booktabs,xcolor,adjustbox}",
        r"\begin{table}[t]",
        (
            r"\caption{FoSTA ablation performance over UCI datasets and seeds under "
            r"the selected distortion splits. Each cell reports mean $\pm$ the "
            r"average within-dataset standard deviation across seeds. "
            r"Results are shown for label transfer accuracy (Acc), alignment score "
            r"(AS), and correspondence recovery measured by FOSCTTM (FOS). Higher "
            r"is better for accuracy and AS, while lower is better for FOSCTTM. "
            + {
                1: "The best result for each metric is highlighted in gold (1st). ",
                2: "The top two results for each metric are highlighted in gold (1st) and silver (2nd). ",
                3: "The top three results for each metric are highlighted in gold (1st), silver (2nd), and bronze (3rd). ",
            }[desired_top]
            + "Each variant is ranked independently against non-FoSTA methods "
            "in the loaded CSV, using only MALI at $t=2$ among MALI variants "
            "and excluding other FoSTA variants. Ties share a rank; "
            "cells without baseline scores are not highlighted. "
            "The final three rows report average scores across the displayed splits, "
            "with standard deviations also averaged across splits.}"
        ),
        r"\label{tab:uci_ablation}",
        r"\centering",
        TABLE_FONT_SIZE,
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\fboxsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        "",
        r"\begin{adjustbox}{max width=\linewidth}",
        fr"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
    ]
    for split in split_order:
        lines.append(r"\midrule")
        for metric_key, metric_label in metrics.items():
            arrow = r"\downarrow" if metric_key in lower_is_better else r"\uparrow"
            row = [split_display_map[split] if metric_key == "alignment_score" else "",
                   f"{metric_label}${arrow}$"]
            for method in method_order:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                row.append(get_highlighted_value(mean, metric_key, split, all_summaries, std))
            lines.append(" & ".join(row) + r" \\")

    avg_means, avg_stds = average_scores(all_summaries, all_stds)
    lines.append(r"\midrule")
    for metric_key, metric_label in metrics.items():
        arrow = r"\downarrow" if metric_key in lower_is_better else r"\uparrow"
        row = ["Average score" if metric_key == "alignment_score" else "",
               f"{metric_label}${arrow}$"]
        row.extend(
            get_highlighted_value(
                avg_means[metric_key].loc[method, "average"],
                metric_key, "average", avg_means,
                avg_stds[metric_key].loc[method, "average"],
            )
            for method in method_order
        )
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def build_markdown_table(all_summaries, all_stds):
    header = ["Split", "Metric"] + [markdown_method_name(method) for method in method_order]
    lines = [
        (
            "FoSTA ablation results averaged over datasets and seeds. Each variant is "
            "ranked independently against non-FoSTA methods in the loaded CSV "
            "(only MALI at t=2 among MALI variants); "
            "other variants are excluded. Ties share a rank. Each cell "
            "reports mean ± average within-dataset standard deviation. First place "
            "is bold and italic, second place is bold, and third place is italic. "
            "The final rows report scores and standard deviations averaged across "
            "the displayed splits."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for split in split_order:
        for metric_key, metric_label in metrics.items():
            arrow = "↓" if metric_key in lower_is_better else "↑"
            row = [markdown_split_display_map[split], metric_label + arrow]
            for method in method_order:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                value = get_markdown_value(mean, metric_key, split, all_summaries)
                row.append(value + (f" ± {std:.2f}" if pd.notna(std) else ""))
            lines.append("| " + " | ".join(row) + " |")

    avg_means, avg_stds = average_scores(all_summaries, all_stds)
    for metric_key, metric_label in metrics.items():
        arrow = "↓" if metric_key in lower_is_better else "↑"
        row = ["Average score", metric_label + arrow]
        for method in method_order:
            mean = avg_means[metric_key].loc[method, "average"]
            std = avg_stds[metric_key].loc[method, "average"]
            value = get_markdown_value(mean, metric_key, "average", avg_means)
            row.append(value + (f" ± {std:.2f}" if pd.notna(mean) and pd.notna(std) else ""))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    results_path = Path(results_csv)
    df = pd.read_csv(results_path)
    # Some experiment exports use a lowercase FoSTA prefix.
    df["method"] = df["method"].str.replace(r"(?i)^fosta_", "FoSTA_", regex=True)
    # Historical MALI runs used the automatic diffusion-time setting.
    df["method"] = df["method"].replace({"MALI": "MALI_auto", "mali_auto": "MALI_auto", "mali_t2": "MALI_t2"})
    select_variants(df)
    all_summaries, all_stds = summarize_results(df)
    latex_table = build_latex_table(all_summaries, all_stds)
    markdown_table = build_markdown_table(all_summaries, all_stds)

    latex_output_path = results_path.with_name(f"{results_path.stem}_ablation_table.tex")
    markdown_output_path = results_path.with_name(f"{results_path.stem}_ablation_table.md")
    latex_output_path.write_text(latex_table, encoding="utf-8")
    markdown_output_path.write_text(markdown_table, encoding="utf-8")

    print(latex_table)
    print(f"Saved {latex_output_path}")
    print(f"Saved {markdown_output_path}")


if __name__ == "__main__":
    main()
