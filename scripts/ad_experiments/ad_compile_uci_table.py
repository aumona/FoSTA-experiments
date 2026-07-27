import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================


# results_csv = "results_uci/results_20260503_015528_general.csv"

results_csv = "results_uci/results_20260503_122708_general_distort05.csv"


# Set this to 1, 2, or 3 to control how many top ranks are colored
desired_top = 3

metrics = {
    "label_transfer": "Acc",
    "alignment_score": "AS",
    "foscttm": "FOS"
}

lower_is_better = ["foscttm"]

# split_order = [
#     "add_gaussian_noise_features",
#     "alternate_importance",
#     "distort",
#     "importance",
#     "random",
#     "rotate"
# ]

split_order = ["distort"]


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

method_order = [
    "FoSTA_gap_auto",

    # "FoSTA_gap_t2",
    # "FoSTA_gap_auto",
    # "FoSTA_kerf_t2",
    # "FoSTA_kerf_auto",



    "MALI", 
    # "MALI_nodpt",
    "Pamona", 
    "KEMAlin", 
    "KEMArbf"
]

method_display_map = {
    "FoSTA_ICML_auto": "FoSTA_ICML ($t=\\texttt{auto}$)",
    "FoSTA_ICML_t2": "FoSTA_ICML ($t=2$)",
    "FoSTA_gap_t2": "FoSTA ($t=2$)",
    "FoSTA_gap_auto": "FoSTA ($t=\\texttt{auto}$)",
    "FoSTA_gap_mauto_t2": "FoSTA (mauto, $t=2$)",
    "FoSTA_kerf_t2": "FoSTA-KeRF ($t=2$)",
    "FoSTA_kerf_auto": "FoSTA-KeRF ($t=\\texttt{auto}$)",
    "FoSTA_kerf_mauto_t2": "FoSTA-KeRF (mauto, $t=2$)",
    "MALI": "MALI", 
    "MALI_nodpt": "MALI (w/o DPT)",
    "Pamona": "Pamona", 
    "KEMAlin": "KEMAlin", 
    "KEMArbf": "KEMArbf"
}



# =========================================================
# HIGHLIGHTING LOGIC
# =========================================================
def get_rank(val, m_key, split_name, all_summaries):
    if pd.isna(val):
        return None
    all_scores = all_summaries[m_key][split_name].sort_values(ascending=(m_key in lower_is_better))
    unique_vals = all_scores.unique()
    for rank, ranked_value in enumerate(unique_vals[:desired_top], start=1):
        if val == ranked_value:
            return rank
    return None


def get_highlighted_value(val, m_key, split_name, all_summaries):
    if pd.isna(val):
        return "---"
    formatted = f"{val:.3f}"

    rank = get_rank(val, m_key, split_name, all_summaries)
    if rank == 1:
        return f"\\gold{{{formatted}}}" 
    if rank == 2:
        return f"\\silver{{{formatted}}}" 
    if rank == 3:
        return f"\\bronze{{{formatted}}}" 
    return formatted


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

    all_summaries = {}
    all_stds = {}
    for metric_key in metrics:
        selected = df[df["method"].isin(method_order)].copy()
        pivot = selected.pivot_table(
            index=["dataset", "split", "seed"],
            columns="method",
            values=metric_key,
        )
        valid_idx = pivot.index[pivot[method_order].notna().all(axis=1)]
        filtered = (
            selected.set_index(["dataset", "split", "seed"])
            .loc[valid_idx]
            .reset_index()
        )

        all_summaries[metric_key] = (
            filtered.groupby(["split", "method"])[metric_key]
            .mean()
            .unstack(level=0)
        )
        stds_per_dataset = filtered.groupby(
            ["split", "method", "dataset"]
        )[metric_key].std()
        all_stds[metric_key] = (
            stds_per_dataset.groupby(["split", "method"])
            .mean()
            .unstack(level=0)
        )

    return all_summaries, all_stds

# =========================================================
# TABLE GENERATION - TWO-ROW FORMAT
# =========================================================
def build_latex_table(all_summaries, all_stds):
    header_row = "Model "
    sub_header = " "
    for split in split_order:
        header_row += f"& \\multicolumn{{3}}{{c}}{{{split_display_map[split]}}} "
        sub_header += "& Acc & AS & FOS "

    lines = [
        header_row + r"\\",
        sub_header + r"\\",
        r"\midrule",
    ]
    for method in method_order:
        row_means = [method_display_map[method]]
        row_errors = [r"\scriptsize{$\pm$ std.}"]
        for split in split_order:
            for metric_key in metrics:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                row_means.append(
                    get_highlighted_value(
                        mean, metric_key, split, all_summaries
                    )
                )
                row_errors.append(
                    fr"\scriptsize{{$\pm${std:.2f}}}"
                    if not pd.isna(std)
                    else " "
                )
        lines.append(" & ".join(row_means) + r" \\")
        lines.append(" & ".join(row_errors) + r" \\[0.5ex]")

    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def build_markdown_table(all_summaries, all_stds):
    header = ["Model"]
    for split in split_order:
        split_name = markdown_split_display_map[split]
        header.extend(
            f"{split_name} {metric_label}"
            for metric_label in metrics.values()
        )

    lines = [
        (
            "UCI benchmark results averaged over datasets and seeds. Each "
            "model's second row reports the mean within-dataset standard "
            "deviation. First place is bold and italic, second place is bold, "
            "and third place is italic."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for method in method_order:
        row_means = [markdown_method_name(method)]
        row_errors = ["± std."]
        for split in split_order:
            for metric_key in metrics:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                row_means.append(
                    get_markdown_value(
                        mean, metric_key, split, all_summaries
                    )
                )
                row_errors.append(
                    f"± {std:.2f}" if not pd.isna(std) else ""
                )
        lines.append("| " + " | ".join(row_means) + " |")
        lines.append("| " + " | ".join(row_errors) + " |")

    return "\n".join(lines) + "\n"


def main():
    results_path = Path(results_csv)
    df = pd.read_csv(results_path)
    all_summaries, all_stds = summarize_results(df)
    latex_table = build_latex_table(all_summaries, all_stds)
    markdown_table = build_markdown_table(all_summaries, all_stds)

    latex_output_path = results_path.with_name(f"{results_path.stem}_table.tex")
    markdown_output_path = results_path.with_name(f"{results_path.stem}_table.md")
    latex_output_path.write_text(latex_table, encoding="utf-8")
    markdown_output_path.write_text(markdown_table, encoding="utf-8")

    print(latex_table)
    print(f"Saved {latex_output_path}")
    print(f"Saved {markdown_output_path}")


if __name__ == "__main__":
    main()
