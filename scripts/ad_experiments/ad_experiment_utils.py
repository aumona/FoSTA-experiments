import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def python_random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def append_result_row(results_csv, row):
    pd.DataFrame([row]).to_csv(
        results_csv,
        mode="a",
        header=not results_csv.exists(),
        index=False,
    )


def write_json(path, data):
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def coerce_embedding_array(embedding, n_components=None, check_finite=True):
    if isinstance(embedding, (list, tuple)):
        if len(embedding) == 1:
            embedding = embedding[0]
        else:
            embedding = np.vstack([np.asarray(part) for part in embedding])

    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding, got shape {embedding.shape}.")
    if check_finite and np.any(~np.isfinite(embedding)):
        raise ValueError("Embedding contains non-finite values.")
    if n_components is not None:
        embedding = embedding[:, :n_components]
    return embedding


def save_embeddings(method_dir, method_name, embedding, n_a, names=("source", "target")):
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"{method_name}_embedding.npz",
        embedding=np.asarray(embedding),
        **{
            names[0]: np.asarray(embedding[:n_a]),
            names[1]: np.asarray(embedding[n_a:]),
        },
    )


def make_color_mapper(plt, cmap_name, n_values):
    if cmap_name != "colorblind":
        return plt.get_cmap(cmap_name, max(n_values, 1))

    okabe_ito = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    if n_values <= len(okabe_ito):
        return lambda i: okabe_ito[i]

    return plt.get_cmap("viridis", max(n_values, 1))


def save_embedding_plots(method_dir, method_name, embedding, plot_specs, ext="png", dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = coerce_embedding_array(embedding, n_components=2)
    if emb.shape[1] < 2:
        raise ValueError(f"Expected at least 2 embedding dimensions for plotting, got {emb.shape[1]}.")

    for suffix, values, cmap_name, legend_title in plot_specs:
        values = np.asarray(values).astype(str)
        unique_values = np.unique(values)
        n_values = len(unique_values)
        legend_rows = 24
        legend_cols = max(1, int(np.ceil(n_values / legend_rows)))
        fig_width = min(7 + 1.45 * legend_cols, 14)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        color_for = make_color_mapper(plt, cmap_name, n_values)

        for i, value in enumerate(unique_values):
            mask = values == value
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=12,
                alpha=0.75,
                color=color_for(i),
                label=str(value),
                edgecolors="none",
                rasterized=True,
            )

        ax.set_title(f"{method_name} colored by {legend_title.lower()}")
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend(
            title=legend_title,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            ncol=legend_cols,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0,
            labelspacing=0.25,
            markerscale=1.15,
            fontsize=7 if n_values > 20 else 8,
            title_fontsize=8,
        )
        fig.tight_layout()
        save_kwargs = {"bbox_inches": "tight"}
        if ext.lower() != "pdf":
            save_kwargs["dpi"] = dpi
        fig.savefig(method_dir / f"{method_name}_embedding_by_{suffix}.{ext}", **save_kwargs)
        plt.close(fig)
