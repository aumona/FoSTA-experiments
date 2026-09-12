"""Print Markdown runtime and memory tables for one scaling-results folder."""

from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

timestamp_folder = "results_multimodal_scaling/20260727_223521"

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

    results = results[
        results["method"].isin(METHODS) & results["status"].eq("ok")
    ]
    if results.empty:
        raise ValueError("No successful FoSTA or MALI rows were found.")

    # The median gives one robust scaling value per method and sample count
    # while retaining all seed repetitions in the source CSV.
    summary = (
        results.groupby(["combined_samples", "method"], as_index=False)[
            ["runtime_sec", "peak_mem_mb"]
        ]
        .median()
        .sort_values(["combined_samples", "method"])
    )

    print(f"# Scaling results: {results_csv.parent.name}\n")
    print("Values are medians across successful seed repetitions.\n")
    print("## Runtime and memory scaling\n")
    print(markdown_table(summary))


def main():
    compile_tables(resolve_results_csv())


if __name__ == "__main__":
    main()
