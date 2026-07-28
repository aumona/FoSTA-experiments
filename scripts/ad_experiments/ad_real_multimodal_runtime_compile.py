from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_multimodal"
TIMESTAMP = "20260609_105815_all_merged"
OUTPUT_FILENAME = "results_multimodal_runtime.md"

METHODS = [
    "KEMAlin",
    "KEMArbf",
    "FoSTA_t2",
    "MALI",
    "Pamona",
]

METHOD_DISPLAY_MAP = {
    "KEMAlin": "KEMAlin",
    "KEMArbf": "KEMArbf",
    "FoSTA_t2": "FoSTA",
    "MALI": "MALI",
    "Pamona": "Pamona",
}

DATASET_DISPLAY_MAP = {
    "ave": ("Audio ↔ Video", "AVE"),
    "har": ("Sensor 1 ↔ Sensor 2", "HAR"),
    "rgbd_resnet18": ("Image ↔ Depth crop", "RGB-D ResNet18"),
    "rgbd_dinov2base": ("Image ↔ Depth crop", "RGB-D DINOv2-B"),
    "sketchy_resnet18": ("Image ↔ Human sketch", "Sketchy ResNet18"),
    "sketchy_dinov2base": ("Image ↔ Human sketch", "Sketchy DINOv2-B"),
}

METRICS = [
    ("runtime_sec", "Runtime (s) ↓"),
    ("peak_mem_mb", "Peak memory (MB) ↓"),
]


def load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "results_multimodal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {
        "dataset",
        "method",
        "source_size",
        *[metric for metric, _ in METRICS],
    }
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def get_sample_counts(df: pd.DataFrame) -> dict[str, int]:
    sample_counts = {}
    for dataset, values in df.groupby("dataset", sort=False)["source_size"]:
        unique_values = values.dropna().unique()
        if len(unique_values) != 1:
            raise ValueError(
                f"Expected one source_size value for {dataset!r}, "
                f"got {unique_values.tolist()}."
            )
        sample_counts[dataset] = int(unique_values[0])
    return sample_counts


def dataset_sort_key(dataset: str) -> tuple[str, str]:
    if dataset not in DATASET_DISPLAY_MAP:
        return "", dataset
    return DATASET_DISPLAY_MAP[dataset]


def markdown_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def display_dataset(dataset: str, sample_count: int) -> str:
    sample_count_k = f"{sample_count * 2 / 1000:.1f}".rstrip("0").rstrip(".")
    if dataset not in DATASET_DISPLAY_MAP:
        return f"{markdown_escape(dataset)} (~{sample_count_k} k samples)"
    alignment_task, dataset_name = DATASET_DISPLAY_MAP[dataset]
    return f"{alignment_task} ({dataset_name}, ~{sample_count_k} k samples)"


def format_value(value: float, std: float) -> str:
    if pd.isna(value):
        return "---"
    return f"{value:.2f} ± {std:.2f}"


def build_markdown_table(
    df: pd.DataFrame, sample_counts: dict[str, int]
) -> str:
    datasets = sorted(df["dataset"].dropna().unique(), key=dataset_sort_key)
    filtered = df[df["method"].isin(METHODS)]
    metric_names = [metric for metric, _ in METRICS]
    means = filtered.groupby(["dataset", "method"])[metric_names].mean()
    stds = (
        filtered.groupby(["dataset", "method"])[metric_names]
        .std()
        .fillna(0.0)
    )

    lines = [
        (
            "Aggregated runtime and peak memory usage over real multimodal "
            "datasets and seeds. Lower is better for both metrics. Missing "
            "Pamona values correspond to runs with excessive runtimes."
        ),
        "",
        "| Alignment task | Metric | "
        + " | ".join(METHOD_DISPLAY_MAP[method] for method in METHODS)
        + " |",
        "| " + " | ".join(["---"] * (len(METHODS) + 2)) + " |",
    ]

    for dataset in datasets:
        for metric_idx, (metric, metric_label) in enumerate(METRICS):
            row = [
                display_dataset(dataset, sample_counts[dataset])
                if metric_idx == 0
                else "",
                metric_label,
            ]
            for method in METHODS:
                key = (dataset, method)
                row.append(
                    "---"
                    if key not in means.index
                    else format_value(
                        means.loc[key, metric],
                        stds.loc[key, metric] if key in stds.index else 0.0,
                    )
                )
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    results_dir = RESULTS_ROOT / TIMESTAMP
    df = load_results(results_dir)
    sample_counts = get_sample_counts(df)
    markdown_table = build_markdown_table(df, sample_counts)

    output_path = results_dir / OUTPUT_FILENAME
    output_path.write_text(markdown_table, encoding="utf-8")
    print(markdown_table)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
