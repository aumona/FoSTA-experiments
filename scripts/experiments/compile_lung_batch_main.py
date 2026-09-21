"""Export independent SVG panels for the main single-cell figure."""

from pathlib import Path
import json
import textwrap

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

from compile_lung_batch_2d_quantitative import EXPECTED_COLORS, EXACT_SYMBOL_MAP

# =============================================================================
# DATA AND OUTPUT: relative paths resolve from the repository root.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# All selected results contribute to quantitative means and standard deviations.
SELECTED_TIMESTAMPS = [
    "results_sc_experiments/20260919_185500_1to6_20",
    "results_sc_experiments/20260920_105524_B1toB4_20",
    "results_sc_experiments/20260920_143010_A1toA6_20",

    # "results_sc_experiments/20260920_033154_1to6_50",
    # "results_sc_experiments/20260920_100839_B1toB4_50",
    # "results_sc_experiments/20260920_142945_A1toA6_50"

]
OUTPUT_PREFIX = "lung_batch_main"
OUTPUT_SUBDIR = "lung_batch_main"  # Inside the last selected timestamp folder.

# Saved AnnData observation columns.
BATCH_KEY = "batch"
LABEL_KEY = "ground_truth_labels"
MASK_KEY = "is_masked"

# =============================================================================
# SHARED FIGURE SIZING AND FONTS
# =============================================================================
COLUMN_WIDTH_PT = 397.48499  # Match the manuscript's LaTeX linewidth.
PT_PER_INCH = 72.27
TEXT_FONT_SIZE_PT = 10  # Plot titles and axis labels.
LEGEND_FONT_SIZE_PT = 8  # All legends and quantitative axis tick labels.

# =============================================================================
# EMBEDDING ROWS: batch-colored and cell-type-colored plots
# =============================================================================
# One source run for both rows, independent of quantitative folder selection.
EMBEDDING_TIMESTAMP = "results_sc_experiments/20260920_105524_B1toB4_20"
BATCH_PAIR = ("B2", "B3")
EMBEDDING_SEED = 56089
EMBEDDING_METHODS = [("Unintegrated", "PCA"), ("FoSTA_t2", "FoSTA"), ("scANVI", "scANVI")]
EMBEDDING_ROW_WIDTH_PT = COLUMN_WIDTH_PT

# Unmasked points in both embedding rows.
UNMASKED_POINT_SIZE = 1  # Scatter area in points squared.
UNMASKED_POINT_ALPHA = 0.50  # Opacity: 0 (transparent) to 1 (opaque).
UNMASKED_POINT_MARKER = "o"

# Masked points in both embedding rows.
MASKED_POINT_SIZE = 1
MASKED_POINT_ALPHA = 0.90
MASKED_POINT_MARKER = "^"

# =============================================================================
# QUANTITATIVE PLOT: Bio conservation vs Batch correction
# =============================================================================
QUANTITATIVE_METHODS = ["Unintegrated", "FoSTA", "scANVI", "scVI", "LIGER", "MALI", "KEMAlin", "KEMArbf", "Pamona", "Scanorama"]
QUANTITATIVE_WIDTH_PT = 0.65 * COLUMN_WIDTH_PT
QUANTITATIVE_POINT_SIZE = 200  # Scatter area in points squared; symbols scale with it.
POINT_LIMIT_PADDING = 0.14  # Fraction of the mean-point range; ignores error bars.

# =============================================================================
# LEGENDS: positive integer item limits; extra entries wrap to new rows.
# =============================================================================
# Batch colors and visible/masked marker legend.
BATCH_LEGEND_MAX_ITEMS_PER_ROW = 1

# Cell-type labels shared by the two selected batches.
SHARED_LABEL_LEGEND_MAX_ITEMS_PER_ROW = 3

# Cell-type labels unique to BATCH_PAIR[0]; omitted if empty.
BATCH_1_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW = 3

# Cell-type labels unique to BATCH_PAIR[1]; omitted if empty.
BATCH_2_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW = 3

# Quantitative method legend, ordered by average Bio/Batch rank.
METHOD_LEGEND_MAX_ITEMS_PER_ROW = 2


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_embedding_data():
    root = resolve(EMBEDDING_TIMESTAMP) / f"seed_{EMBEDDING_SEED}"
    pair = f"{BATCH_PAIR[0]}_vs_{BATCH_PAIR[1]}"
    folder = root / pair
    if not (folder / "adata_final.h5ad").exists():
        folder = root / f"{BATCH_PAIR[1]}_vs_{BATCH_PAIR[0]}"
    path = folder / "adata_final.h5ad"
    if not path.exists():
        available = sorted(p.parent.name for p in root.glob("*/adata_final.h5ad"))
        raise FileNotFoundError(f"No saved embedding for {BATCH_PAIR}, seed {EMBEDDING_SEED}. Available: {available}")
    data = ad.read_h5ad(path, backed="r")
    try:
        obs = data.obs[[BATCH_KEY, LABEL_KEY, MASK_KEY]].copy()
        embeddings = {}
        for key, _ in EMBEDDING_METHODS:
            if key not in data.obsm:
                raise ValueError(f"{key} is missing from {path}")
            coords = np.asarray(data.obsm[key]).copy()
            if coords.shape != (len(obs), 2) or not np.isfinite(coords).all():
                raise ValueError(f"{key} must contain a finite 2D embedding for every cell.")
            embeddings[key] = coords
    finally:
        data.file.close()
    actual = set(obs[BATCH_KEY].astype(str))
    if actual != set(map(str, BATCH_PAIR)):
        raise ValueError(f"Expected {BATCH_PAIR}, found batches {actual}")
    return obs, embeddings, path


def load_quantitative_results():
    records = []
    for timestamp in SELECTED_TIMESTAMPS:
        root = resolve(timestamp)
        files = sorted(root.glob("seed_*/*/benchmark_metrics_seed*.csv"))
        if not files:
            raise FileNotFoundError(f"No benchmark CSVs in {root}")
        for path in files:
            frame = pd.read_csv(path, index_col=0)
            frame.index = frame.index.astype(str).str.strip()
            frame.index = frame.index.map(lambda name: "FoSTA" if name == "FoSTA_t2" else name)
            for method, row in frame.iterrows():
                if method not in QUANTITATIVE_METHODS:
                    continue
                records.append({
                    "method": method, "seed": path.parent.parent.name,
                    "pair": path.parent.name,
                    "bio": pd.to_numeric(row["Bio conservation"], errors="coerce"),
                    "batch": pd.to_numeric(row["Batch correction"], errors="coerce"),
                })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No results for the selected quantitative methods.")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["bio", "batch"])
    if frame.empty:
        raise ValueError("No finite quantitative scores.")
    summary = frame.groupby("method").agg(
        bio_mean=("bio", "mean"), bio_std=("bio", "std"),
        batch_mean=("batch", "mean"), batch_std=("batch", "std"), count=("bio", "size"),
    )
    summary[["bio_std", "batch_std"]] = summary[["bio_std", "batch_std"]].fillna(0)
    summary["bio_rank"] = summary.bio_mean.rank(ascending=False, method="min").astype(int)
    summary["batch_rank"] = summary.batch_mean.rank(ascending=False, method="min").astype(int)
    summary["average_rank"] = (summary.bio_rank + summary.batch_rank) / 2
    # Alphabetical method order breaks average-rank ties.
    summary = summary.sort_index(key=lambda names: names.str.casefold())
    return summary.sort_values("average_rank", kind="stable"), frame


def point_limits(values):
    """Center each axis on its mean-point extent, not on its error bars."""
    low, high = float(np.min(values)), float(np.max(values))
    span = max(high - low, 0.08)
    center = (low + high) / 2
    half = span * (0.5 + POINT_LIMIT_PADDING)
    return center - half, center + half


def draw_embedding(ax, coords, categories, colors, masked):
    for category, color in colors.items():
        selected = categories == category
        for is_masked, marker, size, alpha in [
            (False, UNMASKED_POINT_MARKER, UNMASKED_POINT_SIZE, UNMASKED_POINT_ALPHA),
            (True, MASKED_POINT_MARKER, MASKED_POINT_SIZE, MASKED_POINT_ALPHA),
        ]:
            idx = selected & (masked == is_masked)
            ax.scatter(coords[idx, 0], coords[idx, 1], s=size, marker=marker,
                       color=color, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.06)
    for spine in ax.spines.values():
        spine.set_color("0.45")
        spine.set_linewidth(0.5)


def masking_handles():
    return [
        Line2D([], [], marker=UNMASKED_POINT_MARKER, color="0.55", linestyle="", markersize=3, label="Visible label"),
        Line2D([], [], marker=MASKED_POINT_MARKER, color="0.35", linestyle="", markersize=5, label="Masked label"),
    ]


def make_embedding_grid(obs, embeddings, cell_colors, batch_colors):
    """Cell types above batches, using Matplotlib's default automatic axes aspect."""
    width = EMBEDDING_ROW_WIDTH_PT / PT_PER_INCH
    spacing = 0.025
    panel_count = len(EMBEDDING_METHODS)
    # Preserve the configured manuscript width and Matplotlib's default figure
    # proportions; axes fill the grid without imposing square panels.
    default_width, default_height = plt.rcParams["figure.figsize"]
    height = width * default_height / default_width
    bottom_margin, top_margin = 0.04, 0.25
    fig, axes = plt.subplots(2, panel_count, squeeze=False, figsize=(width, height))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=bottom_margin / height,
                        top=1 - top_margin / height, wspace=spacing, hspace=spacing)
    masked = obs[MASK_KEY].to_numpy(dtype=bool)
    for row, (category_key, colors, row_title) in enumerate([
        (LABEL_KEY, cell_colors, "Cell type"), (BATCH_KEY, batch_colors, "Batch"),
    ]):
        categories = obs[category_key].astype(str).to_numpy()
        for ax, (key, title) in zip(axes[row], EMBEDDING_METHODS):
            draw_embedding(ax, embeddings[key], categories, colors, masked)
            if row == 0:
                ax.set_title("Unintegrated PCA" if key == "Unintegrated" else title, pad=3,
                             fontweight="bold" if title == "FoSTA" else "normal")
        axes[row, 0].set_ylabel(row_title, labelpad=3)
    return fig


def make_horizontal_legend(handles, labels, title, max_items_per_row):
    """Configurable entries per row, ordered left to right and tightly cropped."""
    if type(max_items_per_row) is not int or max_items_per_row < 1:
        raise ValueError(f"Legend item limit for {title!r} must be a positive integer.")
    columns = min(max_items_per_row, len(handles))
    # Matplotlib fills columns first; reorder entries to read across rows.
    lengths = [len(handles) // columns + (i < len(handles) % columns)
               for i in range(columns)]
    indices = [[] for _ in range(columns)]
    index = 0
    for row in range(max(lengths)):
        for column, length in enumerate(lengths):
            if row < length:
                indices[column].append(index)
                index += 1
    order = [index for column in indices for index in column]
    fig = plt.figure(figsize=(6, 2))
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order],
                        title=title, loc="upper left", frameon=False, ncol=columns,
                        handler_map={tuple: HandlerTuple(ndivide=1)},
                        handletextpad=0.4, columnspacing=1.2,
                        labelspacing=0.7, borderaxespad=0)
    fig.canvas.draw()
    bounds = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return fig, bounds.padded(0.02)


def make_category_legend(colors, title, max_items_per_row, batch=False, include_masking=True,
                         cell_counts=None):
    handles = [Line2D([], [], marker="o", color=color, linestyle="", markersize=4,
                      label=f"Batch {label}" if batch else (
                          textwrap.fill(label.replace("_", " "), 29)
                          + (f" ({cell_counts[label][0]} / {cell_counts[label][1]})"
                             if cell_counts is not None else "")))
               for label, color in colors.items()]
    if include_masking:
        handles += masking_handles()
    return make_horizontal_legend(handles, [handle.get_label() for handle in handles], title, max_items_per_row)


def make_quantitative_plot(summary):
    width = QUANTITATIVE_WIDTH_PT / PT_PER_INCH
    fig, ax = plt.subplots(figsize=(width, width))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.92)
    ax.set_box_aspect(1)
    ax.set_title("Integration quality", pad=6, fontweight="bold")
    for method, row in summary.iterrows():
        color = EXPECTED_COLORS.get(method, "#777777")
        symbol = EXACT_SYMBOL_MAP.get(method, "$?$")
        ax.errorbar(row.batch_mean, row.bio_mean, xerr=row.batch_std, yerr=row.bio_std,
                    fmt="none", ecolor=color, alpha=0.40, elinewidth=0.8,
                    capsize=2.5, capthick=0.6, zorder=1)
        ax.scatter(row.batch_mean, row.bio_mean, s=QUANTITATIVE_POINT_SIZE, color=color, zorder=3)
        ax.scatter(row.batch_mean, row.bio_mean, s=QUANTITATIVE_POINT_SIZE * (24 / 65),
                   marker=symbol, color="white", zorder=4)
    ax.set_xlim(*point_limits(summary.batch_mean))
    ax.set_ylim(*point_limits(summary.bio_mean))
    ax.set_xlabel("Batch correction", labelpad=3)
    ax.set_ylabel("Bio conservation", labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(length=2, pad=2)
    ax.grid(linestyle="--", linewidth=0.45, alpha=0.4)
    ax.set_axisbelow(True)
    return fig


def make_method_legend(summary):
    handles, labels = [], []
    for method, row in summary.iterrows():
        handles.append((
            Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(55),
                   color=EXPECTED_COLORS.get(method, "#777777")),
            Line2D([], [], linestyle="", marker=EXACT_SYMBOL_MAP.get(method, "$?$"),
                   markersize=np.sqrt(20), color="white"),
        ))
        labels.append(f"{method} ({int(row.bio_rank)} / {int(row.batch_rank)})")
    return make_horizontal_legend(handles, labels,
                                  "Methods by average rank (Bio rank / Batch rank)",
                                  METHOD_LEGEND_MAX_ITEMS_PER_ROW)


def main():
    obs, embeddings, embedding_path = load_embedding_data()
    summary, raw = load_quantitative_results()
    output_dir = resolve(SELECTED_TIMESTAMPS[-1]) / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_colors = dict(zip(map(str, BATCH_PAIR), ["#1f77b4", "#ff7f0e"]))
    types = sorted(obs[LABEL_KEY].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    cell_colors = {label: cmap(i % 20) for i, label in enumerate(types)}
    output_files = []
    with plt.rc_context({
        "font.size": TEXT_FONT_SIZE_PT, "axes.titlesize": TEXT_FONT_SIZE_PT,
        "axes.labelsize": TEXT_FONT_SIZE_PT, "xtick.labelsize": LEGEND_FONT_SIZE_PT,
        "ytick.labelsize": LEGEND_FONT_SIZE_PT, "legend.fontsize": LEGEND_FONT_SIZE_PT,
        "legend.title_fontsize": LEGEND_FONT_SIZE_PT, "svg.fonttype": "path",
        "axes.linewidth": 0.6,
    }):
        batch_legend, batch_bounds = make_category_legend(
            batch_colors, "Batch / masking", BATCH_LEGEND_MAX_ITEMS_PER_ROW, batch=True)
        batch_names = list(map(str, BATCH_PAIR))
        counts_by_batch = [
            obs.loc[obs[BATCH_KEY].astype(str) == batch, LABEL_KEY].astype(str).value_counts()
            for batch in batch_names
        ]
        cell_counts = {label: tuple(int(counts.get(label, 0)) for counts in counts_by_batch)
                       for label in types}
        label_sets = [set(obs.loc[obs[BATCH_KEY].astype(str) == batch, LABEL_KEY].astype(str))
                      for batch in batch_names]
        label_groups = [
            ("shared", f"Shared cell types (Batch {batch_names[0]} count / Batch {batch_names[1]} count)",
             label_sets[0] & label_sets[1], SHARED_LABEL_LEGEND_MAX_ITEMS_PER_ROW),
            ("batch_1_specific", f"Batch {batch_names[0]}-specific labels", label_sets[0] - label_sets[1], BATCH_1_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW),
            ("batch_2_specific", f"Batch {batch_names[1]}-specific labels", label_sets[1] - label_sets[0], BATCH_2_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW),
        ]
        cell_legends = []
        for suffix, title, present, limit in label_groups:
            if not present:
                continue
            colors = {label: color for label, color in cell_colors.items() if label in present}
            legend, bounds = make_category_legend(colors, title, limit, include_masking=False,
                                                  cell_counts=cell_counts)
            cell_legends.append((f"4_cell_type_legend_{suffix}", legend, bounds))
        method_legend, method_bounds = make_method_legend(summary)
        panels = [
            ("1_embeddings", make_embedding_grid(obs, embeddings, cell_colors, batch_colors), None),
            ("2_batch_legend", batch_legend, batch_bounds),
            *cell_legends,
            ("5_quantitative", make_quantitative_plot(summary), "tight"),
            ("6_method_legend", method_legend, method_bounds),
        ]
        for name, fig, bounds in panels:
            path = output_dir / f"{OUTPUT_PREFIX}_{name}.svg"
            fig.savefig(path, dpi=350, bbox_inches=bounds, pad_inches=0.02)
            plt.close(fig)
            output_files.append(path.name)
            print(f"Saved {path}")
    # Remove only obsolete legend artifacts owned by this compiler, including
    # a batch-specific legend left by an earlier run with a different pair.
    for suffix in ["", "_batch_1", "_batch_2", "_shared", "_batch_1_specific", "_batch_2_specific"]:
        old_path = output_dir / f"{OUTPUT_PREFIX}_4_cell_type_legend{suffix}.svg"
        if old_path.name not in output_files:
            old_path.unlink(missing_ok=True)
    summary.to_csv(output_dir / f"{OUTPUT_PREFIX}_summary.csv")
    metadata = {
        "embedding_file": str(embedding_path), "batch_pair": list(BATCH_PAIR),
        "embedding_seed": EMBEDDING_SEED, "results_folders": SELECTED_TIMESTAMPS,
        "embedding_row_width_pt": EMBEDDING_ROW_WIDTH_PT,
        "quantitative_width_pt": QUANTITATIVE_WIDTH_PT,
        "text_font_size_pt": TEXT_FONT_SIZE_PT, "legend_font_size_pt": LEGEND_FONT_SIZE_PT,
        "output_svgs": output_files,
        "quantitative_aggregation": "Mean and sample SD over available seed-pair runs per method",
        "duplicate_policy": "None; every loaded result contributes equally",
        "error_bars": "Clipped at axes; axis limits use mean points only",
        "embedding_label_colors": LABEL_KEY,
        "loaded_result_rows": len(raw),
    }
    (output_dir / f"{OUTPUT_PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Embedding pair: {BATCH_PAIR}, seed {EMBEDDING_SEED}; quantitative counts:\n{summary['count']}")


if __name__ == "__main__":
    main()
