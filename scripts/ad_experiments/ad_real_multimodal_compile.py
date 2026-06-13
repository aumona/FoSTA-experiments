from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_multimodal"
TIMESTAMP = "20260609_105815"
OUTPUT_FILENAME = "results_multimodal_table.tex"

DESIRED_TOP = 3
TABLE_FONT_SIZE = r"\scriptsize"
METHOD_CELL_WIDTH = "1.75cm"

METHODS = [
    "FoSTA_t2",
    # "FoSTA_tauto",
    "KEMAlin",
    "KEMArbf",
    "MALI",
    "Pamona",
]

METHOD_DISPLAY_MAP = {
    "FoSTA_t2": "FoSTA",
    "FoSTA_tauto": "FoSTA ($t=\\texttt{auto}$)",
    "KEMAlin": "KEMAlin",
    "KEMArbf": "KEMArbf",
    "MALI": "MALI",
    "Pamona": "Pamona",
    "Unintegrated": "Unintegrated",
    "Unintegrated_PHATE": "Unintegrated PHATE",
}

DATASET_DISPLAY_MAP = {
    "ave": "AVE",
    "har": "HAR",
    "rgbd_resnet18": "RGB-D ResNet18",
    "rgbd_dinov2base": "RGB-D DINOv2-B",
    "sketchy_resnet18": "Sketchy ResNet18",
    "sketchy_dinov2base": "Sketchy DINOv2-B",
}

METRICS = [
    ("label_transfer_top1", "Acc$\\uparrow$", False),
    ("alignment_score", "AS$\\uparrow$", False),
    ("FOSCTTM", "FOS$\\downarrow$", True),
]

CAPTION = (
    "Aggregated performance over real multimodal datasets and seeds. Results are "
    "reported for label transfer accuracy (Acc), alignment score (AS), and "
    "correspondence recovery measured by FOSCTTM. Higher is better for accuracy "
    "and AS, while lower is better for FOSCTTM. The top three results for each "
    "metric are highlighted in gold (1st), silver (2nd), and bronze (3rd)."
)
LABEL = "tab:real_multimodal"

RANK_STYLES = {
    1: (r"gold!25", r"\bfseries"),
    2: (r"gray!18", r"\bfseries\color{gray}"),
    3: (r"brown!14", r"\bfseries\color{brown}"),
}


def load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "results_multimodal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    required_cols = {"dataset", "method", *[metric for metric, _, _ in METRICS]}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df


def summarize(
    df: pd.DataFrame, methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    metric_cols = [metric for metric, _, _ in METRICS]
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to the requested methods.")

    means = df.groupby(["dataset", "method"], sort=False)[metric_cols].mean()
    stds = df.groupby(["dataset", "method"], sort=False)[metric_cols].std().fillna(0.0)
    datasets = list(dict.fromkeys(df["dataset"].tolist()))
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


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY_MAP.get(dataset, latex_escape(dataset))


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
    fill_color, text_style = RANK_STYLES[rank]
    return f"\\cellcolor{{{fill_color}}}\\makebox[{METHOD_CELL_WIDTH}][c]{{{text_style} {formatted}}}"


def format_plain_value(formatted: str) -> str:
    return f"\\makebox[{METHOD_CELL_WIDTH}][c]{{{formatted}}}"


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f}{{\\tiny $\\pm${std:.2f}}}"


def make_column_spec(n_methods: int) -> str:
    return "ll " + " ".join(["c"] * n_methods)


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
            avg_means[metric][method] = pd.Series(mean_values).mean() if mean_values else float("nan")
            avg_stds[metric][method] = pd.Series(std_values).mean() if std_values else float("nan")

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
    means: pd.DataFrame, stds: pd.DataFrame, datasets: list[str], methods: list[str]
) -> str:
    header = ["Dataset", "Metric"] + [
        f"\\makebox[{METHOD_CELL_WIDTH}][c]{{{display_method(method)}}}" for method in methods
    ]
    avg_means, avg_stds = average_scores(means, stds, datasets, methods)

    lines = [
        r"\begin{table}[t]",
        r"{\footnotesize",
        f"\\caption{{{CAPTION}}}",
        r"}",
        f"\\label{{{LABEL}}}",
        r"\centering",
        TABLE_FONT_SIZE,
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        "",
        f"\\begin{{tabular}}{{{make_column_spec(len(methods))}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for dataset_idx, dataset in enumerate(datasets):
        for metric_idx, (metric, metric_label, lower_is_better) in enumerate(METRICS):
            dataset_cell = (
                f"\\multirow{{{len(METRICS)}}}{{*}}{{{display_dataset(dataset)}}}"
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


def main() -> None:
    results_dir = RESULTS_ROOT / TIMESTAMP
    df = load_results(results_dir)
    means, stds, datasets, methods = summarize(df, METHODS)
    latex_table = build_latex_table(means, stds, datasets, methods)

    output_path = results_dir / OUTPUT_FILENAME
    output_path.write_text(latex_table)
    print(latex_table)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
