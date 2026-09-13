"""Print Markdown runtime and memory tables for one scaling-results folder."""

from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

timestamp_folder = "results_multimodal_scaling/20260913_105928"

METHODS = ("FoSTA", "MALI")
RESULTS_FILENAME = "results_multimodal_scaling.csv"


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_results_csv():
    folder = Path(timestamp_folder).expanduser()
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    results_csv = folder / RESULTS_FILENAME
    if not results_csv.is_file():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    return results_csv


def markdown_table(summary):
    runtime = summary.pivot(
        index="combined_samples",
        columns="method",
        values="runtime_sec",
    ).reindex(columns=METHODS)
    memory = summary.pivot(
        index="combined_samples",
        columns="method",
        values="peak_mem_mb",
    ).reindex(columns=METHODS)

    lines = [
        "| Samples | FoSTA time (s) | MALI time (s) | FoSTA mem (MB) | MALI mem (MB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for combined_samples in runtime.index.union(memory.index):
        values = (
            runtime.loc[combined_samples, "FoSTA"],
            runtime.loc[combined_samples, "MALI"],
            memory.loc[combined_samples, "FoSTA"],
            memory.loc[combined_samples, "MALI"],
        )
        formatted = [
            "—" if pd.isna(value) else f"{value:.1f}"
            for value in values
        ]
        lines.append(
            f"| {int(combined_samples):,} | "
            + " | ".join(formatted)
            + " |"
        )
    return "\n".join(lines)


def compile_tables(results_csv):
    results = pd.read_csv(results_csv)
    required = {
        "method",
        "combined_samples",
        "runtime_sec",
        "peak_mem_mb",
        "status",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            "Results CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Older single-dataset exports may not include a dataset column.
    if "dataset" not in results.columns:
        results["dataset"] = "Dataset"
    else:
        results["dataset"] = results["dataset"].fillna("Unknown dataset")

    print(f"# Scaling results: {results_csv.parent.name}\n")
    print("Values are medians across successful seed repetitions.\n")
    for dataset, dataset_results in results.groupby("dataset", sort=True):
        print(f"## {dataset}: runtime and memory scaling\n")
        successful = dataset_results[
            dataset_results["method"].isin(METHODS)
            & dataset_results["status"].eq("ok")
        ]
        if successful.empty:
            print("No successful FoSTA or MALI rows were found.\n")
            continue
        # Aggregate seeds only within the current dataset.
        summary = (
            successful.groupby(["combined_samples", "method"], as_index=False)[
                ["runtime_sec", "peak_mem_mb"]
            ]
            .median()
            .sort_values(["combined_samples", "method"])
        )
        print(markdown_table(summary))
        print()


def main():
    compile_tables(resolve_results_csv())


if __name__ == "__main__":
    main()
