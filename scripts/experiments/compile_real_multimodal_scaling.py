"""Export a hierarchical LaTeX table of median or mean scaling measurements."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# CONFIG: relative timestamp folders are resolved from the project root.
timestamp_folder = "results_multimodal_scaling/20260915_231351"
RESULTS_FILENAME = "results_multimodal_scaling.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CELL_WIDTH = "1.5cm"
TABLE_FONT_SIZE = r"\small"
AGG_MODE = "mean"  # "median" or "mean"; mean includes sample standard deviations.
DATASET_LABELS = {
    "rgbd_dinov2base": "RGB-D",
    "rgbd_resnet18": "RGB-D",
    "sketchy_dinov2base": "Sketchy",
    "sketchy_resnet18": "Sketchy",
    "ave": "AVE",
    "har": "HAR",
}


def resolve_results_csv(folder=None):
    folder = Path(folder if folder is not None else timestamp_folder).expanduser()
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    results_csv = folder / RESULTS_FILENAME
    if not results_csv.is_file():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    return results_csv


def load_summary(results_csv):
    if AGG_MODE not in ("median", "mean"):
        raise ValueError("AGG_MODE must be 'median' or 'mean'.")
    results = pd.read_csv(results_csv)
    required = {"method", "combined_samples", "runtime_sec", "peak_mem_mb", "status"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError("Results CSV is missing required columns: " + ", ".join(sorted(missing)))
    if "dataset" not in results:
        results["dataset"] = "Dataset"
    results["dataset"] = results["dataset"].fillna("Unknown dataset")
    results = results[results["status"].eq("ok")].copy()
    numeric = ["combined_samples", "runtime_sec", "peak_mem_mb"]
    results[numeric] = results[numeric].apply(pd.to_numeric, errors="coerce")
    results[numeric] = results[numeric].replace([np.inf, -np.inf], np.nan)
    results = results.dropna(subset=["method", "combined_samples"])
    results = results[results["combined_samples"] > 0]
    for metric in ("runtime_sec", "peak_mem_mb"):
        results.loc[results[metric] < 0, metric] = np.nan
    grouped = results.groupby(["dataset", "method", "combined_samples"])[
        ["runtime_sec", "peak_mem_mb"]
    ]
    summary = grouped.agg(AGG_MODE)
    if AGG_MODE == "mean":
        summary = summary.join(grouped.std().add_suffix("_std"))
    summary = summary.reset_index()
    if summary.dropna(subset=["runtime_sec", "peak_mem_mb"], how="all").empty:
        raise ValueError("No successful results with usable runtime or memory measurements.")
    return summary


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def format_cell(value, std=np.nan):
    formatted = "---" if pd.isna(value) else f"{value:,.2f}"
    if AGG_MODE == "mean" and pd.notna(value):
        error = "---" if pd.isna(std) else f"{std:,.2f}"
        formatted += fr"{{\tiny $\pm${error}}}"
    return f"\\makebox[{SAMPLE_CELL_WIDTH}][c]{{{formatted}}}"


def build_latex_table(summary):
    samples = sorted(summary["combined_samples"].unique())
    datasets = sorted(summary["dataset"].unique())
    methods = sorted(summary["method"].unique())
    values = summary.set_index(["dataset", "method", "combined_samples"])
    caption = (
        "Runtime and peak-memory scaling across datasets and methods. Values are "
        + (r"means $\pm$ sample standard deviations" if AGG_MODE == "mean" else "medians")
        + " across successful seed repetitions. Columns are grouped by method, "
        "with subcolumns giving the total number "
        "of samples across both domains. Runtime is in seconds and memory is in GB "
        "(1,024 MB per GB). "
        "Dashes indicate missing measurements or standard deviations that cannot "
        "be estimated from fewer than two repetitions."
    )
    header = ["", ""] + [
        fr"\multicolumn{{{len(samples)}}}{{c}}{{{latex_escape(method)}}}"
        for method in methods
    ]
    subheader = ["Metric", "Dataset"] + [
        f"\\makebox[{SAMPLE_CELL_WIDTH}][c]{{{sample:,.0f}}}"
        for method in methods for sample in samples
    ]
    rules = " ".join(
        fr"\cmidrule(lr){{{3 + i * len(samples)}-{2 + (i + 1) * len(samples)}}}"
        for i in range(len(methods))
    )
    lines = [
        r"% Requires \usepackage{booktabs,multirow}",
        r"\begin{table}[t]", r"{\small", f"\\caption{{{caption}}}", r"}",
        r"\label{tab:multimodal_scaling}", r"\vspace{6pt}", r"\centering",
        TABLE_FONT_SIZE, r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.0}", "",
        r"\begin{tabular}{ll " + " ".join("c" for _ in range(len(methods) * len(samples))) + "}",
        r"\toprule", " & ".join(header) + r" \\",
        rules, " & ".join(subheader) + r" \\",
    ]
    for metric, metric_label in [("runtime_sec", "Runtime (s)"), ("peak_mem_mb", "Memory (GB)")]:
        lines.append(r"\midrule")
        for dataset_index, dataset in enumerate(datasets):
            dataset_label = latex_escape(DATASET_LABELS.get(dataset, dataset))
            row = [
                fr"\multirow{{{len(datasets)}}}{{*}}{{{metric_label}}}"
                if dataset_index == 0 else "",
                dataset_label,
            ]
            for method in methods:
                for sample in samples:
                    key = (dataset, method, sample)
                    value = values.loc[key, metric] if key in values.index else np.nan
                    std = (
                        values.loc[key, metric + "_std"]
                        if AGG_MODE == "mean" and key in values.index else np.nan
                    )
                    if metric == "peak_mem_mb":
                        value /= 1024
                        std /= 1024
                    row.append(format_cell(value, std))
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def compile_tables(results_csv, output_dir=None):
    results_csv = Path(results_csv)
    summary = load_summary(results_csv)
    output_dir = Path(output_dir) if output_dir is not None else results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{results_csv.stem}_table.tex"
    path.write_text(build_latex_table(summary), encoding="utf-8")
    print(f"Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timestamp_folder", nargs="?", default=timestamp_folder,
                        help="Folder containing results_multimodal_scaling.csv")
    parser.add_argument("--output-dir", type=Path, help="Override the LaTeX output directory")
    args = parser.parse_args()
    compile_tables(resolve_results_csv(args.timestamp_folder), args.output_dir)


if __name__ == "__main__":
    main()
