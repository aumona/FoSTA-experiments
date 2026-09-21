"""Compile one LaTeX table with Bio and Batch-correction sections for lung runs.

Uses the selected folders and methods from compile_lung_batch_main.py by default.
All valid rows are pooled without deduplication. Tables are saved under the last
selected timestamp. Requires booktabs and xcolor in the manuscript.
"""
import numpy as np
import pandas as pd

import compile_lung_batch_main as lung
import compile_real_multimodal as style

# Override these lists here to compile different folders or methods.
SELECTED_TIMESTAMPS = list(lung.SELECTED_TIMESTAMPS)
METHODS = list(lung.QUANTITATIVE_METHODS)
OUTPUT_SUBDIR = "lung_batch_tables"

# Match the multimodal table's typography, cell sizes, and ranking macros.
BIO_METRICS = [
    ("Isolated labels", "Isolated"),
    ("KMeans NMI", "NMI"),
    ("KMeans ARI", "ARI"),
    ("Silhouette label", "ASW label"),
    ("cLISI", "cLISI"),
    ("Bio conservation", "Bio score"),
]
BATCH_METRICS = [
    ("BRAS", "BRAS"),
    ("iLISI", "iLISI"),
    ("KBET", "kBET"),
    ("Graph connectivity", "Graph"),
    ("PCR comparison", "PCR"),
    ("Batch correction", "Batch score"),
]


def load_summary():
    columns = [key for key, _ in BIO_METRICS + BATCH_METRICS]
    frames = []
    for timestamp in SELECTED_TIMESTAMPS:
        root = lung.resolve(timestamp)
        files = sorted(root.glob("seed_*/*/benchmark_metrics_seed*.csv"))
        if not files:
            raise FileNotFoundError(f"No benchmark CSVs in {root}")
        for path in files:
            frame = pd.read_csv(path, index_col=0)
            frame.index = frame.index.astype(str).str.strip().str.replace(
                r"^FoSTA_t2$", "FoSTA", regex=True
            )
            frame = frame.loc[frame.index.isin(METHODS)].reindex(columns=columns)
            frame = frame.apply(pd.to_numeric, errors="coerce")
            frame["folder"] = str(root.resolve())
            frame["pair"] = path.parent.name
            frames.append(frame)
    if not frames:
        raise ValueError("Select at least one result folder.")
    rows = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    # Same valid-run selection as the quantitative plot.
    rows = rows.dropna(subset=["Bio conservation", "Batch correction"])
    if rows.empty:
        raise ValueError("No finite scores for the selected methods.")
    grouped = rows.groupby(level=0)[columns]
    means, counts = grouped.mean(), grouped.count()
    within_pair = rows.groupby([rows.index, "folder", "pair"])[columns].std()
    stds = within_pair.groupby(level=0).mean()
    bio_rank = means["Bio conservation"].rank(ascending=False, method="min")
    batch_rank = means["Batch correction"].rank(ascending=False, method="min")
    order = ((bio_rank + batch_rank) / 2).sort_index(key=lambda x: x.str.casefold())
    order = order.sort_values(kind="stable").index
    return means.loc[order], stds.loc[order], counts.loc[order]


def ranked_cell(means, stds, method, metric):
    value = means.loc[method, metric]
    if pd.isna(value):
        return style.format_plain_value("---")
    # Dense ranks match the multimodal tables: ties share a highlight.
    rank = int(means[metric].rank(ascending=False, method="dense").loc[method])
    std = stds.loc[method, metric]
    formatted = (f"{value:.3f}" + r"{\tiny $\pm$---}" if pd.isna(std) and style.INCLUDE_STDS
                 else style.format_mean_std(value, std))
    if rank <= min(style.DESIRED_TOP, 3):
        return style.format_ranked_value(formatted, rank)
    return style.format_plain_value(formatted)


def build_table(means, stds):
    caption = (
        "Biological conservation and batch correction on the lung integration benchmark. "
        r"Results are mean $\pm$ average within-pair standard deviation across seeds. "
        "Standard deviations are computed separately within each selected folder and batch pair, "
        "then averaged across pairs; higher is better for every metric. "
        "Methods are ordered identically in both sections by average Bio/Batch rank, "
        "with alphabetical tie-breaking. Gold, silver, and bronze mark the top three "
        "distinct scores per metric; ties share a highlight."
    )
    lines = [
        "% Requires: \\usepackage{booktabs,xcolor}",
        r"\begin{table}[t]",
        f"\\caption{{{caption}}}",
        r"\label{tab:lung_full_metrics}",
        r"\centering",
        style.TABLE_FONT_SIZE,
        r"\providecommand{\bronze}[1]{\textcolor[rgb]{0.70,0.43,0.20}{\textbf{#1}}}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\fboxsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular}{lccccc@{\hspace{7pt}}c}",
        r"\toprule",
    ]
    for i, (title, metrics) in enumerate([
        ("Batch correction", BATCH_METRICS), ("Biological conservation", BIO_METRICS),
    ]):
        if i:
            lines.extend([r"\addlinespace[0.8ex]", r"\midrule"])
        lines.extend([
            f"\\multicolumn{{7}}{{c}}{{\\textbf{{{title}}}}}" + r" \\",
            r"\cmidrule(lr){1-7}",
            "Method & " + " & ".join(
                f"\\makebox[{style.METHOD_CELL_WIDTH}][c]{{{name} $\\uparrow$}}"
                for _, name in metrics
            ) + r" \\",
            r"\midrule",
        ])
        for method in means.index:
            name = style.latex_escape(method)
            if method == "FoSTA":
                name = f"\\textbf{{{name}}}"
            lines.append(" & ".join([name] + [ranked_cell(means, stds, method, key)
                                                 for key, _ in metrics]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    means, stds, counts = load_summary()
    output_dir = lung.resolve(SELECTED_TIMESTAMPS[-1]) / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lung_full_metrics.tex"
    path.write_text(build_table(means, stds))
    print(f"Saved {path}")
    means.to_csv(output_dir / "metric_means.csv")
    stds.to_csv(output_dir / "metric_stds.csv")
    counts.to_csv(output_dir / "metric_counts.csv")


if __name__ == "__main__":
    main()
