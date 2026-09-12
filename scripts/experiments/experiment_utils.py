import json
import multiprocessing as mp
import queue as queue_module
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


DEFAULT_MEMORY_SAMPLE_INTERVAL_SEC = 0.02


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def profile_fit_transform(
    prepare_fit,
    prepare_args=(),
    timeout_sec=None,
    memory_sample_interval_sec=DEFAULT_MEMORY_SAMPLE_INTERVAL_SEC,
    return_result=True,
):
    """Measure one isolated ``fit_transform`` call.

    ``prepare_fit(*prepare_args)`` runs inside the worker before profiling and
    must return ``(output_name, model, fit_transform_args)``. Runtime therefore
    covers only ``model.fit_transform(*fit_transform_args)``. Peak memory is
    the maximum process-tree VMS increase relative to the immediately pre-fit
    baseline, which remains observable through OS paging and compression.
    """
    queue = mp.Queue()
    start_event = mp.Event()
    result_event = mp.Event()
    process = mp.Process(
        target=_fit_transform_worker,
        args=(
            prepare_fit,
            tuple(prepare_args),
            queue,
            start_event,
            result_event,
            return_result,
        ),
    )

    process.start()
    fit_started = False
    fit_start = None
    baseline_vms_mb = np.nan
    peak_mem_mb = 0.0
    deadline = None
    try:
        while True:
            if fit_started and deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"fit_transform timeout after {timeout_sec} seconds"
                    )
                poll_seconds = min(memory_sample_interval_sec, remaining)
            else:
                poll_seconds = memory_sample_interval_sec

            try:
                message = queue.get(timeout=poll_seconds)
            except queue_module.Empty:
                message = None

            if fit_started:
                peak_mem_mb = _update_peak_vms_increase(
                    peak_mem_mb,
                    baseline_vms_mb,
                    _get_process_tree_vms_mb(process.pid),
                )

            if message is not None and message[0] == "ready":
                baseline_vms_mb = _get_process_tree_vms_mb(process.pid)
                if not np.isfinite(baseline_vms_mb):
                    raise RuntimeError(
                        "Unable to read worker memory before fit_transform."
                    )
                fit_started = True
                fit_start = time.perf_counter()
                deadline = (
                    None
                    if timeout_sec is None
                    else fit_start + timeout_sec
                )
                start_event.set()
                continue

            if message is not None and message[0] == "finished":
                peak_mem_mb = _update_peak_vms_increase(
                    peak_mem_mb,
                    baseline_vms_mb,
                    _get_process_tree_vms_mb(process.pid),
                )
                runtime_sec = float(message[1])
                result_event.set()
                result_message = queue.get()
                if result_message[0] != "ok":
                    raise RuntimeError(result_message[1])
                return result_message[1], runtime_sec, peak_mem_mb

            if message is not None and message[0] == "error":
                exc = RuntimeError(message[1])
                exc.runtime_sec = float(message[2])
                exc.peak_mem_mb = peak_mem_mb
                raise exc

            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    "fit_transform worker exited without returning a result "
                    f"(exit code {process.exitcode})."
                )
    except Exception as exc:
        if process.is_alive():
            process.terminate()
            process.join()
        if not hasattr(exc, "runtime_sec"):
            exc.runtime_sec = (
                float(time.perf_counter() - fit_start)
                if fit_start is not None
                else np.nan
            )
        if not hasattr(exc, "peak_mem_mb"):
            exc.peak_mem_mb = peak_mem_mb
        raise
    finally:
        if process.is_alive():
            process.join(1)
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()


def _fit_transform_worker(
    prepare_fit,
    prepare_args,
    queue,
    start_event,
    result_event,
    return_result,
):
    try:
        output_name, model, fit_transform_args = prepare_fit(*prepare_args)
        queue.put(("ready",))
        start_event.wait()
        start = time.perf_counter()
        result = model.fit_transform(*fit_transform_args)
        runtime_sec = float(time.perf_counter() - start)
        queue.put(("finished", runtime_sec))
        result_event.wait()
        queue.put(
            (
                "ok",
                (output_name, result if return_result else None),
            )
        )
    except Exception as exc:
        runtime_sec = (
            float(time.perf_counter() - start)
            if "start" in locals()
            else np.nan
        )
        queue.put(("error", str(exc), runtime_sec))


def _get_process_tree_vms_mb(root_pid):
    try:
        root = psutil.Process(root_pid)
    except (psutil.Error, ProcessLookupError, TypeError, OSError):
        return np.nan
    try:
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, ProcessLookupError, OSError):
        processes = [root]

    total_vms_bytes = 0
    observed_process = False
    for process in {proc.pid: proc for proc in processes}.values():
        try:
            total_vms_bytes += int(process.memory_info().vms)
            observed_process = True
        except (psutil.Error, ProcessLookupError, OSError):
            continue
    if not observed_process:
        return np.nan
    return float(total_vms_bytes) / (1024 ** 2)


def _update_peak_vms_increase(peak_mb, baseline_mb, current_mb):
    if not np.isfinite(baseline_mb) or not np.isfinite(current_mb):
        return peak_mb
    return max(peak_mb, float(max(0.0, current_mb - baseline_mb)))


@contextmanager
def python_random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def append_result_row(results_csv, row):
    results_csv = Path(results_csv)
    row_df = pd.DataFrame([row])

    if not results_csv.exists() or results_csv.stat().st_size == 0:
        row_df.to_csv(results_csv, index=False)
        return

    existing_cols = pd.read_csv(results_csv, nrows=0).columns.tolist()
    new_cols = [col for col in row_df.columns if col not in existing_cols]

    if new_cols:
        all_cols = existing_cols + new_cols
        existing_df = pd.read_csv(results_csv).reindex(columns=all_cols)
        row_df = row_df.reindex(columns=all_cols)
        pd.concat([existing_df, row_df], ignore_index=True).to_csv(results_csv, index=False)
        return

    row_df.reindex(columns=existing_cols).to_csv(
        results_csv,
        mode="a",
        header=False,
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


def make_style_mapper(plt, cmap_name, n_values):
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")

    if cmap_name != "colorblind":
        cmap = plt.get_cmap(cmap_name, max(n_values, 1))
        n_colors = getattr(cmap, "N", n_values)
        return lambda i: (
            cmap(i % n_colors),
            markers[(i // n_colors) % len(markers)],
        )

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
    return lambda i: (
        okabe_ito[i % len(okabe_ito)],
        markers[(i // len(okabe_ito)) % len(markers)],
    )


def make_color_mapper(plt, cmap_name, n_values):
    style_for = make_style_mapper(plt, cmap_name, n_values)
    return lambda i: style_for(i)[0]


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
        style_for = make_style_mapper(plt, cmap_name, n_values)

        for i, value in enumerate(unique_values):
            mask = values == value
            color, marker = style_for(i)
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=12,
                alpha=0.75,
                color=color,
                marker=marker,
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
