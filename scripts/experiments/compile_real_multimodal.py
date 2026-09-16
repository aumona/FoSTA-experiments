from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_multimodal"
TIMESTAMP = "20260915_150749"
OUTPUT_FILENAME = "results_multimodal_table.tex"
MARKDOWN_OUTPUT_FILENAME = "results_multimodal_table.md"

DESIRED_TOP = 3
COMPACT = True  # Hide top-k accuracy and average RGB-D/Sketchy encoder variants.
INCLUDE_STDS = True
TABLE_FONT_SIZE = r"\small"
METHOD_CELL_WIDTH = "1.5cm"

METHODS = [
    "FoSTA_t2",
    # "FoSTA_tauto",
    "KEMAlin",
    "KEMArbf",
    # "MALI_tauto",
    "MALI_t2",
    "Pamona",
]

METHOD_DISPLAY_MAP = {
    "FoSTA_t2": "FoSTA",
    "FoSTA_tauto": "FoSTA (auto $t$)",
    "KEMAlin": "KEMAlin",
    "KEMArbf": "KEMArbf",
    "MALI_t2": "MALI",
    "MALI_tauto": "MALI (auto $t$)",
    "Pamona": "Pamona",
    "Unintegrated": "Unintegrated",
    "Unintegrated_PHATE": "Unintegrated PHATE",
}

DATASET_DISPLAY_MAP = {
    "rgbd": (r"Image $\leftrightarrow$ Depth crop", "RGB-D"),
    "sketchy": (r"Image $\leftrightarrow$ Human sketch", "Sketchy"),
    "ave": (r"Audio $\leftrightarrow$ Video", "AVE"),
    "har": (r"Sensor 1 $\leftrightarrow$ Sensor 2", "HAR"),
    "rgbd_resnet18": (r"Image $\leftrightarrow$ Depth crop", "RGB-D ResNet18"),
    "rgbd_dinov2base": (r"Image $\leftrightarrow$ Depth crop", "RGB-D DINOv2-B"),
    "sketchy_resnet18": (r"Image $\leftrightarrow$ Human sketch", "Sketchy ResNet18"),
    "sketchy_dinov2base": (r"Image $\leftrightarrow$ Human sketch", "Sketchy DINOv2-B"),
}

MARKDOWN_DATASET_DISPLAY_MAP = {
    "rgbd": ("Image ↔ Depth crop", "RGB-D"),
    "sketchy": ("Image ↔ Human sketch", "Sketchy"),
    "ave": ("Audio ↔ Video", "AVE"),
    "har": ("Sensor 1 ↔ Sensor 2", "HAR"),
    "rgbd_resnet18": ("Image ↔ Depth crop", "RGB-D ResNet18"),
    "rgbd_dinov2base": ("Image ↔ Depth crop", "RGB-D DINOv2-B"),
    "sketchy_resnet18": ("Image ↔ Human sketch", "Sketchy ResNet18"),
    "sketchy_dinov2base": ("Image ↔ Human sketch", "Sketchy DINOv2-B"),
}

METRICS = [
    ("label_transfer_top1", "Acc$\\uparrow$", False),
    ("alignment_score", "AS$\\uparrow$", False),
    ("FOSCTTM", "FOS$\\downarrow$", True),
]
TOP5_METRIC = ("label_transfer_top5", "Acc@5$\\uparrow$", False)
TOP10_METRIC = ("label_transfer_top10", "Acc@10$\\uparrow$", False)
ALL_METRICS = [METRICS[0], TOP5_METRIC, TOP10_METRIC, *METRICS[1:]]
TOP5_DATASETS = ("ave",)
TOP10_DATASET_PREFIXES = ("rgbd_", "sketchy_")

CAPTION = (
    "Aggregated performance over real multimodal datasets and seeds. Results are "
    "reported for label transfer accuracy (Acc and, where applicable, Acc@5 or "
    "Acc@10), "
    "alignment score (AS), and "
    "correspondence recovery measured by FOSCTTM. Label transfer accuracy and "
    "top-k accuracy use photo-to-sketch (A to B) transfer only for Sketchy; "
    "for HAR, AVE, and RGB-D, they average both directions (A to B and B to A). "
    "Higher is better for accuracy "
    "and AS, while lower is better for FOSCTTM. The top three results for each "
    "metric are highlighted with filled gold (1st) and silver (2nd) boxes, "
    "and bronze text (3rd)."
    " Missing Pamona values correspond to runs with excessive runtimes."
)
MARKDOWN_CAPTION = (
    "Aggregated performance over real multimodal datasets and seeds. Results are "
    "reported for label transfer accuracy (Acc and, where applicable, Acc@5 or "
    "Acc@10), "
    "alignment score (AS), and "
    "correspondence recovery measured by FOSCTTM. Label transfer accuracy and "
    "top-k accuracy use photo-to-sketch (A to B) transfer only for Sketchy; "
    "for HAR, AVE, and RGB-D, they average both directions (A to B and B to A). "
    "Higher is better for accuracy "
    "and AS, while lower is better for FOSCTTM. The best result for each metric "
    "is bold and italic, and the second-best result is bold."
    " Missing Pamona values correspond to runs with excessive runtimes."
)
LABEL = "tab:real_multimodal"

RANK_MACROS = {
    1: r"\gold",
    2: r"\silver",
    3: r"\bronze",
}

def load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "results_multimodal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    required_cols = {
        "dataset",
        "method",
        "n_unique_classes",
        *[metric for metric, _, _ in ALL_METRICS],
    }
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df


def get_class_counts(df: pd.DataFrame) -> dict[str, int]:
    class_counts = {}
    for dataset, values in df.groupby("dataset", sort=False)["n_unique_classes"]:
        unique_values = values.dropna().unique()
        if len(unique_values) != 1:
            raise ValueError(
                f"Expected one n_unique_classes value for {dataset!r}, got {unique_values.tolist()}."
            )
        class_counts[dataset] = int(unique_values[0])
    return class_counts


def summarize(
    df: pd.DataFrame, methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if "label_mask_perc" in df and df["label_mask_perc"].nunique(dropna=False) > 1:
        raise ValueError("Summarize one label_mask_perc at a time; do not pool masking levels.")
    metric_cols = [metric for metric, _, _ in ALL_METRICS]
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to the requested methods.")

    means = df.groupby(["dataset", "method"], sort=False)[metric_cols].mean()
    stds = df.groupby(["dataset", "method"], sort=False)[metric_cols].std().fillna(0.0)
    datasets = sorted(dict.fromkeys(df["dataset"].tolist()), key=dataset_sort_key)
    present_methods = [method for method in methods if method in set(df["method"])]
    return means, stds, datasets, present_methods


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def display_method(method: str) -> str:
    return METHOD_DISPLAY_MAP.get(method, latex_escape(method))


def display_dataset(dataset: str, class_count: int) -> str:
    if dataset not in DATASET_DISPLAY_MAP:
        return f"{latex_escape(dataset)} ({class_count} classes)"

    alignment_task, dataset_name = DATASET_DISPLAY_MAP[dataset]
    return rf"\shortstack[l]{{{alignment_task} \\ ({dataset_name}, {class_count} classes)}}"


def markdown_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def display_method_markdown(method: str) -> str:
    display_name = METHOD_DISPLAY_MAP.get(method, method)
    return markdown_escape(
        display_name.replace("$t=\\texttt{auto}$", "`t=auto`")
    )


def display_dataset_markdown(dataset: str, class_count: int) -> str:
    if dataset not in MARKDOWN_DATASET_DISPLAY_MAP:
        return f"{markdown_escape(dataset)} ({class_count} classes)"

    alignment_task, dataset_name = MARKDOWN_DATASET_DISPLAY_MAP[dataset]
    return f"{alignment_task} ({dataset_name}, {class_count} classes)"


def dataset_sort_key(dataset: str) -> tuple[str, str]:
    if dataset not in DATASET_DISPLAY_MAP:
        return ("", dataset)

    alignment_task, dataset_name = DATASET_DISPLAY_MAP[dataset]
    return (alignment_task, dataset_name)


def metrics_for_dataset(dataset: str) -> list[tuple[str, str, bool]]:
    if COMPACT:
        return METRICS
    if dataset in TOP5_DATASETS:
        return [METRICS[0], TOP5_METRIC, *METRICS[1:]]
    if dataset.startswith(TOP10_DATASET_PREFIXES):
        return [METRICS[0], TOP10_METRIC, *METRICS[1:]]
    return METRICS


def compact_results(means, stds, datasets, methods, class_counts):
    """Give each encoder equal weight; require both variants for merged values."""
    means, stds = means.copy(), stds.copy()
    datasets, class_counts = list(datasets), dict(class_counts)
    for merged, variants in {
        "rgbd": ["rgbd_resnet18", "rgbd_dinov2base"],
        "sketchy": ["sketchy_resnet18", "sketchy_dinov2base"],
    }.items():
        present = [variant for variant in variants if variant in datasets]
        if not present:
            continue
        counts = {class_counts[variant] for variant in present}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent class counts for {merged}: {counts}")
        for frame in (means, stds):
            for method in methods:
                index = pd.MultiIndex.from_product(
                    [variants, [method]], names=frame.index.names
                )
                frame.loc[(merged, method), :] = frame.reindex(index).mean(
                    axis=0, skipna=False
                )
        means = means.drop(index=present, level="dataset")
        stds = stds.drop(index=present, level="dataset")
        datasets = [dataset for dataset in datasets if dataset not in present]
        datasets.append(merged)
        class_counts[merged] = counts.pop()
    return means, stds, sorted(datasets, key=dataset_sort_key), class_counts


def table_caption(markdown=False):
    caption = MARKDOWN_CAPTION if markdown else CAPTION
    if COMPACT:
        caption = caption.replace(
            "(Acc and, where applicable, Acc@5 or Acc@10)", "(Acc)"
        )
        caption += (
            " RGB-D and Sketchy scores and standard deviations are averaged "
            "equally over ResNet18 and DINOv2-B; both variants are required. "
            "The final average retains equal weighting of the original dataset "
            "variants, as in the full table."
        )
    return caption


def highlighted_value(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    dataset: str,
    method: str,
    metric: str,
    lower_is_better: bool,
) -> str:
    key = (dataset, method)
    if key not in means.index:
        return "---"

    value = means.loc[key, metric]
    if pd.isna(value):
        return "---"

    std = stds.loc[key, metric] if key in stds.index else 0.0
    dataset_scores = means.xs(dataset, level="dataset")[metric].dropna()
    unique_scores = sorted(dataset_scores.unique(), reverse=not lower_is_better)
    formatted = format_mean_std(value, std)

    if DESIRED_TOP >= 1 and value == unique_scores[0]:
        return format_ranked_value(formatted, 1)
    if DESIRED_TOP >= 2 and len(unique_scores) > 1 and value == unique_scores[1]:
        return format_ranked_value(formatted, 2)
    if DESIRED_TOP >= 3 and len(unique_scores) > 2 and value == unique_scores[2]:
        return format_ranked_value(formatted, 3)
    return format_plain_value(formatted)


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
    return f"{mean:.3f}{{\\tiny $\\pm${std:.2f}}}"


def format_mean_std_markdown(mean: float, std: float) -> str:
    if not INCLUDE_STDS:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.2f}"


def markdown_value(
    scores: pd.Series,
    value: float,
    std: float,
    lower_is_better: bool,
) -> str:
    if pd.isna(value):
        return "---"

    formatted = format_mean_std_markdown(value, std)
    unique_scores = sorted(scores.dropna().unique(), reverse=not lower_is_better)
    if unique_scores and value == unique_scores[0]:
        return f"***{formatted}***"
    if len(unique_scores) > 1 and value == unique_scores[1]:
        return f"**{formatted}**"
    return formatted


def make_column_spec(n_methods: int) -> str:
    return "lr " + " ".join(["c"] * n_methods)


def average_scores(
    means: pd.DataFrame, stds: pd.DataFrame, datasets: list[str], methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_means = {}
    avg_stds = {}
    for metric, _, _ in METRICS:
        avg_means[metric] = {}
        avg_stds[metric] = {}
        for method in methods:
            mean_values = [
                means.loc[(dataset, method), metric]
                for dataset in datasets
                if (dataset, method) in means.index and not pd.isna(means.loc[(dataset, method), metric])
            ]
            std_values = [
                stds.loc[(dataset, method), metric]
                for dataset in datasets
                if (dataset, method) in stds.index and not pd.isna(stds.loc[(dataset, method), metric])
            ]
            has_complete_scores = len(mean_values) == len(datasets)
            has_complete_stds = len(std_values) == len(datasets)
            avg_means[metric][method] = (
                pd.Series(mean_values).mean() if has_complete_scores else float("nan")
            )
            avg_stds[metric][method] = (
                pd.Series(std_values).mean() if has_complete_stds else float("nan")
            )

    return pd.DataFrame(avg_means), pd.DataFrame(avg_stds)


def highlighted_summary_value(
    avg_means: pd.DataFrame,
    avg_stds: pd.DataFrame,
    method: str,
    metric: str,
    lower_is_better: bool,
) -> str:
    if method not in avg_means.index or pd.isna(avg_means.loc[method, metric]):
        return "---"

    value = avg_means.loc[method, metric]
    std = avg_stds.loc[method, metric] if method in avg_stds.index else 0.0
    metric_scores = avg_means[metric].dropna()
    unique_scores = sorted(metric_scores.unique(), reverse=not lower_is_better)
    formatted = format_mean_std(value, std)

    if DESIRED_TOP >= 1 and value == unique_scores[0]:
        return format_ranked_value(formatted, 1)
    if DESIRED_TOP >= 2 and len(unique_scores) > 1 and value == unique_scores[1]:
        return format_ranked_value(formatted, 2)
    if DESIRED_TOP >= 3 and len(unique_scores) > 2 and value == unique_scores[2]:
        return format_ranked_value(formatted, 3)
    return format_plain_value(formatted)


def build_latex_table(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    class_counts: dict[str, int],
) -> str:
    header = ["Alignment task", "Metric"] + [
        f"\\makebox[{METHOD_CELL_WIDTH}][c]{{{display_method(method)}}}" for method in methods
    ]
    avg_means, avg_stds = average_scores(means, stds, datasets, methods)
    if COMPACT:
        means, stds, datasets, class_counts = compact_results(
            means, stds, datasets, methods, class_counts
        )

    lines = [
        r"\begin{table}[t]",
        r"{\small",
        f"\\caption{{{table_caption()}}}",
        r"}",
        f"\\label{{{LABEL}}}",
        r"\centering",
        TABLE_FONT_SIZE,
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\fboxsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        "",
        f"\\begin{{tabular}}{{{make_column_spec(len(methods))}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for dataset_idx, dataset in enumerate(datasets):
        dataset_metrics = metrics_for_dataset(dataset)
        for metric_idx, (metric, metric_label, lower_is_better) in enumerate(dataset_metrics):
            dataset_cell = (
                f"\\multirow{{{len(dataset_metrics)}}}{{*}}{{{display_dataset(dataset, class_counts[dataset])}}}"
                if metric_idx == 0
                else ""
            )
            row = [dataset_cell, metric_label]
            for method in methods:
                row.append(highlighted_value(means, stds, dataset, method, metric, lower_is_better))
            lines.append(" & ".join(row) + r" \\")

        if dataset_idx < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\midrule")
    for metric_idx, (metric, metric_label, lower_is_better) in enumerate(METRICS):
        dataset_cell = (
            f"\\multirow{{{len(METRICS)}}}{{*}}{{Average score}}" if metric_idx == 0 else ""
        )
        row = [dataset_cell, metric_label]
        for method in methods:
            row.append(highlighted_summary_value(avg_means, avg_stds, method, metric, lower_is_better))
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_markdown_table(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    class_counts: dict[str, int],
) -> str:
    header = ["Alignment task", "Metric"] + [
        display_method_markdown(method) for method in methods
    ]
    lines = [
        table_caption(markdown=True),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    avg_means, avg_stds = average_scores(means, stds, datasets, methods)
    if COMPACT:
        means, stds, datasets, class_counts = compact_results(
            means, stds, datasets, methods, class_counts
        )

    for dataset in datasets:
        for metric_idx, (metric, metric_label, lower_is_better) in enumerate(
            metrics_for_dataset(dataset)
        ):
            dataset_cell = (
                display_dataset_markdown(dataset, class_counts[dataset])
                if metric_idx == 0
                else ""
            )
            row = [dataset_cell, metric_label.replace("$\\uparrow$", " ↑").replace("$\\downarrow$", " ↓")]
            dataset_scores = means.xs(dataset, level="dataset")[metric]
            for method in methods:
                key = (dataset, method)
                if key not in means.index:
                    row.append("---")
                    continue
                row.append(
                    markdown_value(
                        dataset_scores,
                        means.loc[key, metric],
                        stds.loc[key, metric] if key in stds.index else 0.0,
                        lower_is_better,
                    )
                )
            lines.append("| " + " | ".join(row) + " |")

    for metric_idx, (metric, metric_label, lower_is_better) in enumerate(METRICS):
        row = [
            "Average score" if metric_idx == 0 else "",
            metric_label.replace("$\\uparrow$", " ↑").replace("$\\downarrow$", " ↓"),
        ]
        for method in methods:
            if method not in avg_means.index:
                row.append("---")
                continue
            row.append(
                markdown_value(
                    avg_means[metric],
                    avg_means.loc[method, metric],
                    avg_stds.loc[method, metric] if method in avg_stds.index else 0.0,
                    lower_is_better,
                )
            )
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    results_dir = RESULTS_ROOT / TIMESTAMP
    df = load_results(results_dir)
    groups = list(df.groupby("label_mask_perc", dropna=False)) if "label_mask_perc" in df else [(None, df)]
    for proportion, group in groups:
        class_counts = get_class_counts(group)
        means, stds, datasets, methods = summarize(group, METHODS)
        latex_table = build_latex_table(means, stds, datasets, methods, class_counts)
        markdown_table = build_markdown_table(means, stds, datasets, methods, class_counts)
        if proportion is not None:
            note = f" Label masking proportion: {proportion} in each domain."
            latex_table = latex_table.replace(table_caption(), table_caption() + note)
            markdown_table = markdown_table.replace(
                table_caption(markdown=True), table_caption(markdown=True) + note
            )
        suffix = f"_mask_{proportion}" if len(groups) > 1 else ""
        tex_name = Path(OUTPUT_FILENAME)
        md_name = Path(MARKDOWN_OUTPUT_FILENAME)
        output_path = results_dir / f"{tex_name.stem}{suffix}{tex_name.suffix}"
        markdown_output_path = results_dir / f"{md_name.stem}{suffix}{md_name.suffix}"
        output_path.write_text(latex_table, encoding="utf-8")
        markdown_output_path.write_text(markdown_table, encoding="utf-8")
        print(latex_table)
        print(f"Saved tables to {output_path} and {markdown_output_path}")


if __name__ == "__main__":
    main()
