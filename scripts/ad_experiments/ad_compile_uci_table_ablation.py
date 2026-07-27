import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

# results_csv = "results_uci/results_20260503_015528_general.csv"
# results_csv = "results_uci/results_20260503_122708_general_distort05.csv"
# results_csv = "results_uci/results_20260503_201720_other_ablation.csv"
results_csv = "results_uci/results_20260725_203910.csv"


desired_top = 3
metrics = {"label_transfer": "Acc", "alignment_score": "AS", "foscttm": "FOS"}
lower_is_better = ["foscttm"]

latex_caption = (
    "UCI ablation results averaged over datasets and seeds. First-, second-, "
    "and third-place scores are highlighted relative to the reference baseline "
    "results reported in Table~1."
)
markdown_caption = (
    "UCI ablation results averaged over datasets and seeds. Each variant's "
    "second row reports the mean within-dataset standard deviation. First-, "
    "second-, and third-place scores are highlighted relative to the reference "
    "baseline results reported in Table 1. First place is bold and italic, "
    "second place is bold, and third place is italic."
)

split_order = ["add_gaussian_noise_features", "alternate_importance", "distort", "importance", "random", "rotate"]

# split_order = ["rotate"]


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
    # "FoSTA_et",

    # "FoSTA_gap_t2",
    # "FoSTA_kerf_auto",

    "FoSTA_spectral",
    "FoSTA_dpt",
    "FoSTA_rotf",
    "FoSTA_no_propag",

    # "FoSTA_dense",

    # "FoSTA_umap",
]

method_display_map = {
    "FoSTA_gap_auto": "FoSTA (Default)",
    "FoSTA_gap_t2": "FoSTA ($t=2$)",
    "FoSTA_kerf_auto": "FoSTA-KeRF (auto)",
    "FoSTA_kerf_t2": "FoSTA-KeRF ($t=2$)",
    "FoSTA_umap": "FoSTA-UMAP",
    "FoSTA_dense": "FoSTA-Dense",
    "FoSTA_et": "FoSTA-et",
    "FoSTA_spectral": "Lap. Eig.",
    "FoSTA_dpt": "DPT",
    "FoSTA_rotf": "RotF",
    "FoSTA_no_propag": "No Prop.",
}

# =========================================================
# REFERENCE DATA
# =========================================================
ref_scores = {
    "label_transfer": {
        "add_gaussian_noise_features": [0.776, 0.760, 0.787, 0.776],
        "alternate_importance": [0.722, 0.721, 0.753, 0.767],
        "distort": [0.785, 0.811, 0.783, 0.794],
        "importance": [0.708, 0.710, 0.720, 0.782],
        "random": [0.682, 0.689, 0.732, 0.760],
        "rotate": [0.818, 0.834, 0.795, 0.808],
    },
    "alignment_score": {
        "add_gaussian_noise_features": [0.867, 0.865, 0.385, 0.430],
        "alternate_importance": [0.988, 0.523, 0.308, 0.568],
        "distort": [1.027, 0.939, 0.427, 0.568],
        "importance": [0.919, 0.471, 0.308, 0.500],
        "random": [0.982, 0.507, 0.312, 0.562],
        "rotate": [1.044, 1.017, 0.447, 0.597],
    },
    "foscttm": {
        "add_gaussian_noise_features": [0.303, 0.400, 0.301, 0.325],
        "alternate_importance": [0.364, 0.358, 0.373, 0.331],
        "distort": [0.204, 0.164, 0.310, 0.307],
        "importance": [0.374, 0.341, 0.385, 0.337],
        "random": [0.384, 0.350, 0.379, 0.336],
        "rotate": [0.098, 0.075, 0.233, 0.263],
    }
}


def get_highlighted_value(val, m_key, split_name):
    if pd.isna(val): return "---"
    baselines = ref_scores[m_key][split_name]
    competition = np.array(baselines + [val])
    is_lower = (m_key in lower_is_better)
    sorted_unique = sorted(np.unique(competition), reverse=not is_lower)
    
    formatted = f"{val:.3f}"
    
    if val == sorted_unique[0]:
        return f"\\gold{{{formatted}}}"
    elif desired_top >= 2 and len(sorted_unique) > 1 and val == sorted_unique[1]:
        return f"\\silver{{{formatted}}}"
    elif desired_top >= 3 and len(sorted_unique) > 2 and val == sorted_unique[2]:
        return f"\\bronze{{{formatted}}}"
    return formatted


def get_markdown_value(val, m_key, split_name):
    if pd.isna(val):
        return "---"

    baselines = ref_scores[m_key][split_name]
    competition = np.array(baselines + [val])
    is_lower = m_key in lower_is_better
    sorted_unique = sorted(np.unique(competition), reverse=not is_lower)
    formatted = f"{val:.3f}"

    if val == sorted_unique[0]:
        return f"***{formatted}***"
    if desired_top >= 2 and len(sorted_unique) > 1 and val == sorted_unique[1]:
        return f"**{formatted}**"
    if desired_top >= 3 and len(sorted_unique) > 2 and val == sorted_unique[2]:
        return f"*{formatted}*"
    return formatted


def markdown_method_name(method):
    return (
        method_display_map.get(method, method)
        .replace("$t=2$", "`t=2`")
        .replace("~", " ")
    )


# =========================================================
# PROCESSING
# =========================================================
def summarize_results(df):
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    filtered = df[df["method"].isin(method_order)].copy()
    all_summaries = {}
    all_stds = {}

    for metric_key in metrics:
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
# TABLE GENERATION
# =========================================================
def build_latex_table(all_summaries, all_stds):
    header = "FoSTA variant "
    sub_header = " "
    for split in split_order:
        header += fr"& \multicolumn{{3}}{{c}}{{{split_display_map[split]}}} "
        sub_header += "& Acc & AS & FOS "

    lines = [
        f"\\caption{{{latex_caption}}}",
        header + r" \\",
        sub_header + r" \\",
        r"\midrule",
    ]

    for method in method_order:
        if method not in all_summaries["label_transfer"].index:
            continue

        row_means = [method_display_map.get(method, method)]
        row_errors = [r"\scriptsize{$\pm$ std.}"]
        for split in split_order:
            for metric_key in metrics:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                row_means.append(
                    get_highlighted_value(mean, metric_key, split)
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
    header = ["FoSTA variant"]
    for split in split_order:
        split_name = markdown_split_display_map[split]
        header.extend(
            f"{split_name} {metric_label}"
            for metric_label in metrics.values()
        )

    lines = [
        markdown_caption,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for method in method_order:
        if method not in all_summaries["label_transfer"].index:
            continue

        row_means = [markdown_method_name(method)]
        row_errors = ["± std."]
        for split in split_order:
            for metric_key in metrics:
                mean = all_summaries[metric_key].loc[method, split]
                std = all_stds[metric_key].loc[method, split]
                row_means.append(
                    get_markdown_value(mean, metric_key, split)
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

    latex_output_path = results_path.with_name(
        f"{results_path.stem}_ablation_table.tex"
    )
    markdown_output_path = results_path.with_name(
        f"{results_path.stem}_ablation_table.md"
    )
    latex_output_path.write_text(latex_table, encoding="utf-8")
    markdown_output_path.write_text(markdown_table, encoding="utf-8")

    print(latex_table)
    print(f"Saved {latex_output_path}")
    print(f"Saved {markdown_output_path}")


if __name__ == "__main__":
    main()
