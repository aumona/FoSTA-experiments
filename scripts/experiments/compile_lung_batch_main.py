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
EMBEDDING_HEIGHT_SCALE = 0.85  # Reduce grid height; 1.0 restores default proportions.

# Extended grid: two methods per row, each with Cell type then Batch panels.
# Uses the same embedding source and separate legends as the main grid.
METHODS_EXTENDED_PLOT = ["FoSTA_t2", "MALI",  "scANVI", "scVI", "KEMArbf", "Unintegrated"]


# Cell-type row: styling is independent of label masking.
CELL_TYPE_POINT_SIZE = 1
CELL_TYPE_POINT_ALPHA = 0.8
# Okabe–Ito palette; shapes distinguish cell types when colors repeat.
CELL_TYPE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                    "#D55E00", "#56B4E9", "#F0E442", "#000000"]
CELL_TYPE_MARKERS = ["o", "^", "+", "x"]  # One shape per palette cycle.

# Unmasked points in the batch row.
UNMASKED_POINT_SIZE = 1  # Scatter area in points squared.
UNMASKED_POINT_ALPHA = 0.50  # Opacity: 0 (transparent) to 1 (opaque).
UNMASKED_POINT_MARKER = "o"

# Masked points in the batch row.
MASKED_POINT_SIZE = 1
MASKED_POINT_ALPHA = 0.90
MASKED_POINT_MARKER = "^"

# =============================================================================
# QUANTITATIVE PLOT: Bio conservation vs Batch correction
# =============================================================================
QUANTITATIVE_METHODS = ["Unintegrated", "FoSTA", "scANVI", "scVI", "LIGER", "MALI", "KEMAlin", "KEMArbf", "Pamona", "Scanorama"]
QUANTITATIVE_WIDTH_PT = 0.65 * COLUMN_WIDTH_PT
QUANTITATIVE_POINT_SIZE = 50  # Area of the plain method dots in points squared.
QUANTITATIVE_LABEL_FONT_SIZE_PT = 8
POINT_LIMIT_PADDING = 0.14  # Fraction of the mean-point range; ignores error bars.

# Method colors and symbols shared by the quantitative plot and its legend.
EXACT_SYMBOL_MAP = {
    "scANVI": r"$\beta$", "scVI": r"$\gamma$", "KEMArbf": r"$\epsilon$",
    "FoSTA": r"$f$", "Scanorama": r"$\alpha$",
    "Unintegrated": r"$0$", "MALI": r"$\zeta$", "KEMAlin": r"$\delta$",
    "LIGER": r"$K$", "Pamona": r"$\eta$"
}

EXPECTED_COLORS = {
    "scANVI": "#31a354", "scVI": "#a1d99b", "KEMArbf": "#fb9a99",
    "FoSTA": "#a6cee3", "Scanorama": "#fdbf6f",
    "Unintegrated": "#1f78b4", "MALI": "#9467bd", "KEMAlin": "#e31a1c",
    "LIGER": "#ff7f00", "Pamona": "#cab2d6"
}

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
        for key in dict.fromkeys([key for key, _ in EMBEDDING_METHODS] + METHODS_EXTENDED_PLOT):
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
                    "pair": path.parent.name, "folder": str(root.resolve()),
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
        bio_mean=("bio", "mean"),
        batch_mean=("batch", "mean"), count=("bio", "size"),
    )
    # Keep experimental conditions in different folders separate (e.g. masking).
    within_pair = frame.groupby(["method", "folder", "pair"])[["bio", "batch"]].std()
    averaged_stds = within_pair.groupby(level="method").mean()
    summary["bio_std"] = averaged_stds["bio"]
    summary["batch_std"] = averaged_stds["batch"]
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


def draw_embedding(ax, coords, categories, colors, masked, cell_markers=None):
    for category, color in colors.items():
        selected = categories == category
        if cell_markers is not None:
            ax.scatter(coords[selected, 0], coords[selected, 1],
                       s=CELL_TYPE_POINT_SIZE, marker=cell_markers[category],
                       color=color, alpha=CELL_TYPE_POINT_ALPHA,
                       linewidths=0.6 if cell_markers[category] in ("+", "x") else 0,
                       rasterized=True)
            continue
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


def embedding_panel_dimensions():
    """Physical panel dimensions in inches, shared by both embedding grids."""
    width = EMBEDDING_ROW_WIDTH_PT / PT_PER_INCH
    default_width, default_height = plt.rcParams["figure.figsize"]
    height = width * default_height / default_width * EMBEDDING_HEIGHT_SCALE
    columns = len(EMBEDDING_METHODS)
    return (width * (0.99 - 0.065) / (columns + 0.025 * (columns - 1)),
            (height - 0.04 - 0.25) / (2 + 0.025))


def make_embedding_grid(obs, embeddings, cell_colors, batch_colors, cell_markers):
    """Cell types above batches, using Matplotlib's default automatic axes aspect."""
    width = EMBEDDING_ROW_WIDTH_PT / PT_PER_INCH
    spacing = 0.025
    panel_count = len(EMBEDDING_METHODS)
    # Preserve the configured manuscript width and Matplotlib's default figure
    # proportions; axes fill the grid without imposing square panels.
    default_width, default_height = plt.rcParams["figure.figsize"]
    height = width * default_height / default_width * EMBEDDING_HEIGHT_SCALE
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
            draw_embedding(ax, embeddings[key], categories, colors, masked,
                           cell_markers=cell_markers if row == 0 else None)
            if row == 0:
                ax.set_title("Unintegrated PCA" if key == "Unintegrated" else title, pad=3,
                             fontweight="bold" if title == "FoSTA" else "normal")
        axes[row, 0].set_ylabel(row_title, labelpad=3)
    return fig


def make_extended_embedding_grid(obs, embeddings, cell_colors, batch_colors, cell_markers, scores):
    """Two methods per row, each shown as Cell type / Batch (four columns)."""
    rows = (len(METHODS_EXTENDED_PLOT) + 1) // 2
    panel_width, panel_height = embedding_panel_dimensions()
    # Match the main grid's panels and horizontal gaps exactly. Allow room
    # between rows for the paired method titles.
    horizontal_gap = 0.025
    row_gap = 0.4
    left_margin, right_margin = 0.08, 0.04
    bottom_margin, top_margin = 0.22, 0.25
    width = left_margin + panel_width * (4 + 3*horizontal_gap) + right_margin
    height = bottom_margin + rows*panel_height + (rows-1)*row_gap + top_margin
    fig, axes = plt.subplots(rows, 4, squeeze=False, figsize=(width, height))
    fig.subplots_adjust(left=left_margin/width, right=1-right_margin/width,
                        bottom=bottom_margin/height, top=1-top_margin/height,
                        wspace=horizontal_gap, hspace=row_gap/panel_height)
    masked = obs[MASK_KEY].to_numpy(dtype=bool)
    batches = obs[BATCH_KEY].astype(str).to_numpy()
    labels = obs[LABEL_KEY].astype(str).to_numpy()
    for i, key in enumerate(METHODS_EXTENDED_PLOT):
        ax_cell, ax_batch = axes[i // 2, (i % 2)*2:(i % 2)*2+2]
        draw_embedding(ax_batch, embeddings[key], batches, batch_colors, masked)
        draw_embedding(ax_cell, embeddings[key], labels, cell_colors, masked,
                       cell_markers=cell_markers)
        for ax in (ax_cell, ax_batch):
            for spine in ax.spines.values():
                spine.set_visible(False)
        title = {"FoSTA_t2": "FoSTA", "Unintegrated": "Unintegrated PCA"}.get(key, key)
        left, right = ax_cell.get_position(), ax_batch.get_position()
        fig.text((left.x0 + right.x1)/2, left.y1 + 0.04/fig.get_figheight(),
                 f"{title} ({scores.loc[key, 'Bio conservation']:.3f} / {scores.loc[key, 'Batch correction']:.3f})",
                 ha="center", va="bottom", fontsize=TEXT_FONT_SIZE_PT,
                 fontweight="bold" if title == "FoSTA" else "normal")
    for ax in axes.flat[len(METHODS_EXTENDED_PLOT)*2:]:
        ax.set_visible(False)
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


def make_category_legend(colors, title, max_items_per_row, batch=False, include_masking=True, markers=None):
    handles = [Line2D([], [], marker=markers[label] if markers else "o", color=color, linestyle="", markersize=4,
                      label=f"Batch {label}" if batch else textwrap.fill(label.replace("_", " "), 29))
               for label, color in colors.items()]
    if include_masking:
        handles += masking_handles()
    return make_horizontal_legend(handles, [handle.get_label() for handle in handles], title, max_items_per_row)


def place_method_labels(ax, summary):
    """Place names near their dots without covering text, dots, or SD bars.

    Search in display coordinates for consistent spacing, choosing the nearest
    collision-free positions across several deterministic placement orders.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    unit = fig.dpi / 72
    bounds = ax.get_window_extent()
    points = ax.transData.transform(summary[["batch_mean", "bio_mean"]].to_numpy())
    obstacles = []
    radius = (np.sqrt(QUANTITATIVE_POINT_SIZE)/2 + 1.5)*unit
    pad = 1.5*unit
    for (x, y), (_, row) in zip(points, summary.iterrows()):
        obstacles.append((x-radius, y-radius, x+radius, y+radius))
        for metric, horizontal in (("batch", True), ("bio", False)):
            std = row[f"{metric}_std"]
            if not np.isfinite(std):
                continue
            delta = np.array([std, 0] if horizontal else [0, std])
            center = np.array([row.batch_mean, row.bio_mean])
            lo, hi = ax.transData.transform([center-delta, center+delta])
            obstacles.append((lo[0]-pad, lo[1]-pad, hi[0]+pad, hi[1]+pad))
            for end in (lo, hi):
                dx, dy = (pad, 3.5*unit) if horizontal else (3.5*unit, pad)
                obstacles.append((end[0]-dx, end[1]-dy, end[0]+dx, end[1]+dy))

    def intersects(a, b):
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    texts, candidates = [], []
    for method, (x, y) in zip(summary.index, points):
        text = ax.text(0, 0, method, ha="center", va="center",
                       fontsize=QUANTITATIVE_LABEL_FONT_SIZE_PT,
                       fontweight="bold" if method == "FoSTA" else "normal", zorder=5)
        texts.append(text)
        box = text.get_window_extent(renderer)
        w, h = box.width+2*pad, box.height+2*pad
        options = []
        for gap in (3, 6, 10, 15, 22, 32, 45, 60):
            for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
                dx, dy = np.cos(angle), np.sin(angle)
                distance = min(w/2/max(abs(dx), 1e-6), h/2/max(abs(dy), 1e-6)) + radius + gap*unit
                cx, cy = x+dx*distance, y+dy*distance
                rect = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
                if (rect[0] < bounds.x0 or rect[2] > bounds.x1 or
                        rect[1] < bounds.y0 or rect[3] > bounds.y1 or
                        any(intersects(rect, obstacle) for obstacle in obstacles)):
                    continue
                options.append((distance**2, rect, (cx, cy)))
        candidates.append(sorted(options, key=lambda option: option[0]))
    rng = np.random.default_rng(0)
    orders = [sorted(range(len(texts)), key=lambda i: len(candidates[i]))]
    orders.extend(rng.permutation(len(texts)) for _ in range(100))
    best = None
    for order in orders:
        placed, positions, total = [], {}, 0
        for i in order:
            chosen = next((c for c in candidates[i]
                           if not any(intersects(c[1], other) for other in placed)), None)
            if chosen is None:
                break
            total += chosen[0]
            placed.append(chosen[1])
            positions[i] = chosen[2]
        else:
            if best is None or total < best[0]:
                best = total, positions
    if best is None:
        raise ValueError("Cannot fit method names without overlaps. Increase QUANTITATIVE_WIDTH_PT or reduce QUANTITATIVE_LABEL_FONT_SIZE_PT.")
    for i, text in enumerate(texts):
        text.set_position(ax.transData.inverted().transform(best[1][i]))


def make_quantitative_plot(summary):
    width = QUANTITATIVE_WIDTH_PT / PT_PER_INCH
    fig, ax = plt.subplots(figsize=(width, width))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.92)
    ax.set_box_aspect(1)
    ax.set_title("Integration quality", pad=6, fontweight="bold")
    for method, row in summary.iterrows():
        color = EXPECTED_COLORS.get(method, "#777777")
        ax.errorbar(row.batch_mean, row.bio_mean, xerr=row.batch_std, yerr=row.bio_std,
                    fmt="none", ecolor=color, alpha=0.40, elinewidth=0.8,
                    capsize=2.5, capthick=0.6, zorder=1)
        ax.scatter(row.batch_mean, row.bio_mean, s=QUANTITATIVE_POINT_SIZE, color=color, zorder=3)
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
    place_method_labels(ax, summary)
    return fig


def make_method_legend(summary):
    handles, labels = [], []
    for method, row in summary.iterrows():
        handles.append(Line2D([], [], linestyle="", marker="o", markersize=4,
                              color=EXPECTED_COLORS.get(method, "#777777")))
        display_name = r"$\mathbf{FoSTA}$" if method == "FoSTA" else method
        labels.append(f"{display_name} ({int(row.bio_rank)} / {int(row.batch_rank)})")
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
    cell_colors = {label: CELL_TYPE_COLORS[i % len(CELL_TYPE_COLORS)] for i, label in enumerate(types)}
    cell_markers = {label: CELL_TYPE_MARKERS[(i // len(CELL_TYPE_COLORS)) % len(CELL_TYPE_MARKERS)] for i, label in enumerate(types)}
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
        label_sets = [set(obs.loc[obs[BATCH_KEY].astype(str) == batch, LABEL_KEY].astype(str))
                      for batch in batch_names]
        label_groups = [
            ("shared", "Shared cell types",
             label_sets[0] & label_sets[1], SHARED_LABEL_LEGEND_MAX_ITEMS_PER_ROW),
            ("batch_1_specific", f"Batch {batch_names[0]}-specific labels", label_sets[0] - label_sets[1], BATCH_1_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW),
            ("batch_2_specific", f"Batch {batch_names[1]}-specific labels", label_sets[1] - label_sets[0], BATCH_2_SPECIFIC_LEGEND_MAX_ITEMS_PER_ROW),
        ]
        cell_legends = []
        for suffix, title, present, limit in label_groups:
            if not present:
                continue
            colors = {label: color for label, color in cell_colors.items() if label in present}
            legend, bounds = make_category_legend(colors, title, limit, include_masking=False, markers=cell_markers)
            cell_legends.append((f"4_cell_type_legend_{suffix}", legend, bounds))
        method_legend, method_bounds = make_method_legend(summary)
        panels = [
            ("1_embeddings", make_embedding_grid(obs, embeddings, cell_colors, batch_colors, cell_markers), None),
            ("2_batch_legend", batch_legend, batch_bounds),
            *cell_legends,
            ("5_quantitative", make_quantitative_plot(summary), "tight"),
            ("6_method_legend", method_legend, method_bounds),
        ]
        if METHODS_EXTENDED_PLOT:
            # Match the exact embedding run, including a reversed pair folder.
            score_path = embedding_path.parent / f"benchmark_metrics_seed{EMBEDDING_SEED}.csv"
            scores = pd.read_csv(score_path, index_col=0)
            scores.index = scores.index.astype(str).str.strip()
            scores = scores.rename(index={"FoSTA": "FoSTA_t2"})
            panels.append(("7_extended_embeddings", make_extended_embedding_grid(
                obs, embeddings, cell_colors, batch_colors, cell_markers, scores), "tight"))
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
        "extended_methods": METHODS_EXTENDED_PLOT,
        "embedding_panel_dimensions_inches": embedding_panel_dimensions(),
        "quantitative_width_pt": QUANTITATIVE_WIDTH_PT,
        "text_font_size_pt": TEXT_FONT_SIZE_PT, "legend_font_size_pt": LEGEND_FONT_SIZE_PT,
        "output_svgs": output_files,
        "quantitative_aggregation": "Mean over all valid runs; average within-folder, within-pair sample SD across seeds (pairs with fewer than two observations excluded)",
        "duplicate_policy": "None; every loaded result contributes equally",
        "error_bars": "Clipped at axes; axis limits use mean points only",
        "embedding_label_colors": LABEL_KEY,
        "loaded_result_rows": len(raw),
    }
    (output_dir / f"{OUTPUT_PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Embedding pair: {BATCH_PAIR}, seed {EMBEDDING_SEED}; quantitative counts:\n{summary['count']}")


if __name__ == "__main__":
    main()
