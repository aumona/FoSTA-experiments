"""
Runtime and peak-memory scaling benchmark for FoSTA and MALI on RGB-D DINOv2.

This reuses the RGB-D loading, deterministic label masking, stratified
subsampling, model construction, and subprocess peak-memory measurement from
ad_real_multimodal.py. No alignment metrics or embeddings are saved.
"""

import multiprocessing as mp
import queue as queue_module
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ad_real_multimodal as benchmark


# =============================================================================
# CONFIG
# =============================================================================
DATASET = "rgbd_dinov2base"
METHOD_CONFIGS = {
    "FoSTA": {
        "embedder": "PHATE",
        "t": 2,
    },
    "MALI": {
    },
}

# Every row, whether labeled or masked, participates in fit_transform(). Sweep
# the total number of rows in each domain while retaining the deterministic
# approximately 50/50 labeled/unlabeled composition. The roughly logarithmic
# spacing is useful for distinguishing near-linear from quadratic scaling.
SAMPLE_SIZES_PER_DOMAIN = [500, 1_000, 2_000, 4_000, 8_000, 15_000]
SEEDS = benchmark.SEEDS
MAX_FIT_TRANSFORM_SEC = benchmark.MAX_FIT_TRANSFORM_SEC

RESULTS_ROOT = PROJECT_ROOT / "results_multimodal_scaling"
RESULTS_FILENAME = "results_multimodal_scaling.csv"
METADATA_FILENAME = "experiment_metadata.json"


def make_scaling_pair(base_pair, requested_samples_per_domain):
    labels = np.asarray(base_pair["labels_a_true"])
    train_mask = np.asarray(base_pair["train_mask_a"], dtype=bool)
    if requested_samples_per_domain == labels.size:
        indices = np.arange(labels.size)
    else:
        n_labeled = int(
            round(
                requested_samples_per_domain
                * benchmark.RGBD_TRAIN_FRACTION
            )
        )
        n_unlabeled = requested_samples_per_domain - n_labeled
        indices = _subsample_label_visibility_pools(
            labels,
            train_mask,
            n_labeled,
            n_unlabeled,
        )

    pair = base_pair.copy()
    for key in (
        "x_a",
        "x_b",
        "labels_a_true",
        "labels_b_true",
        "labels_a_model",
        "labels_b_model",
        "train_mask_a",
        "train_mask_b",
    ):
        pair[key] = np.asarray(base_pair[key])[indices]

    pair["source_size"] = int(indices.size)
    pair["target_size"] = int(indices.size)
    return pair


def _subsample_label_visibility_pools(
    labels,
    labeled_mask,
    n_labeled,
    n_unlabeled,
):
    labeled_pool = np.flatnonzero(labeled_mask)
    unlabeled_pool = np.flatnonzero(~labeled_mask)
    if n_labeled > labeled_pool.size or n_unlabeled > unlabeled_pool.size:
        raise ValueError(
            f"Cannot select {n_labeled} labeled and {n_unlabeled} unlabeled "
            f"rows from pools of size {labeled_pool.size} and "
            f"{unlabeled_pool.size}."
        )

    labeled_local = benchmark.make_stratified_subsample_indices(
        labels[labeled_pool],
        np.ones(labeled_pool.size, dtype=bool),
        n_labeled,
    )
    unlabeled_local = benchmark.make_stratified_subsample_indices(
        labels[unlabeled_pool],
        np.zeros(unlabeled_pool.size, dtype=bool),
        n_unlabeled,
    )
    return np.sort(
        np.concatenate(
            [
                labeled_pool[labeled_local],
                unlabeled_pool[unlabeled_local],
            ]
        )
    )


def profile_fit_transform(method_name, pair, seed):
    """Run fit_transform in a subprocess and return runtime and peak RSS."""
    queue = mp.Queue()
    process = mp.Process(
        target=_run_scaling_method_worker,
        args=(method_name, pair, seed, queue),
    )

    start = time.perf_counter()
    process.start()
    peak_mem_mb = np.nan
    deadline = (
        None
        if MAX_FIT_TRANSFORM_SEC is None
        else start + MAX_FIT_TRANSFORM_SEC
    )
    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"fit_transform timeout after "
                        f"{MAX_FIT_TRANSFORM_SEC} seconds"
                    )
                poll_seconds = min(0.5, remaining)
            else:
                poll_seconds = 0.5

            try:
                message = queue.get(timeout=poll_seconds)
                break
            except queue_module.Empty:
                if not process.is_alive():
                    process.join()
                    raise RuntimeError(
                        "fit_transform worker exited without returning a "
                        f"result (exit code {process.exitcode})."
                    )
    except Exception as exc:
        if process.pid is not None:
            try:
                peak_mem_mb = (
                    float(psutil.Process(process.pid).memory_info().rss)
                    / (1024 ** 2)
                )
            except (psutil.Error, ProcessLookupError):
                peak_mem_mb = np.nan
        if process.is_alive():
            process.terminate()
            process.join()
        exc.runtime_sec = float(time.perf_counter() - start)
        exc.peak_mem_mb = peak_mem_mb
        raise
    finally:
        if process.is_alive():
            process.join(1)
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()

    runtime_sec = float(time.perf_counter() - start)
    if message[0] == "error":
        peak_mem_mb = message[2]
        exc = RuntimeError(message[1])
        exc.runtime_sec = runtime_sec
        exc.peak_mem_mb = peak_mem_mb
        raise exc

    peak_mem_mb = float(message[1])
    return runtime_sec, peak_mem_mb


def _run_scaling_method_worker(method_name, pair, seed, queue):
    """Fit one explicitly configured method and report its process peak RSS."""
    import resource

    try:
        common_params = {
            "n_components": benchmark.N_COMPONENTS,
            "random_state": seed,
            **METHOD_CONFIGS[method_name],
        }
        if method_name == "FoSTA":
            model = benchmark.FoSTA(
                n_jobs=benchmark.N_JOBS,
                **common_params,
            )
        elif method_name == "MALI":
            model = benchmark.MALI(**common_params)
        else:
            raise ValueError(f"Unknown scaling method {method_name!r}.")

        model.fit_transform(
            pair["x_a"],
            pair["x_b"],
            pair["labels_a_model"],
            pair["labels_b_model"],
        )
        queue.put(("ok", _get_process_peak_memory_mb(resource)))
    except Exception as exc:
        queue.put(
            (
                "error",
                str(exc),
                _get_process_peak_memory_mb(resource),
            )
        )


def _get_process_peak_memory_mb(resource_module):
    try:
        peak_rss = resource_module.getrusage(
            resource_module.RUSAGE_SELF
        ).ru_maxrss
        peak_bytes = int(peak_rss) if sys.platform == "darwin" else int(peak_rss) * 1024
        return float(peak_bytes) / (1024 ** 2)
    except Exception:
        return np.nan


def validate_config(base_pair):
    if not SAMPLE_SIZES_PER_DOMAIN:
        raise ValueError(
            "SAMPLE_SIZES_PER_DOMAIN must contain at least one value."
        )
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in SAMPLE_SIZES_PER_DOMAIN
    ):
        raise ValueError(
            "SAMPLE_SIZES_PER_DOMAIN must contain only positive integers."
        )
    if SAMPLE_SIZES_PER_DOMAIN != sorted(set(SAMPLE_SIZES_PER_DOMAIN)):
        raise ValueError(
            "SAMPLE_SIZES_PER_DOMAIN must be unique and sorted in ascending "
            "order."
        )

    largest_requested_total = SAMPLE_SIZES_PER_DOMAIN[-1]
    available_total = int(base_pair["source_size"])
    if largest_requested_total > available_total:
        raise ValueError(
            f"Largest requested point requires {largest_requested_total} rows "
            f"per modality, but the loaded pair has only {available_total}."
        )


def result_row(
    method_name,
    pair,
    requested_samples_per_domain,
    seed,
    runtime_sec,
    peak_mem_mb,
    status,
    error="",
):
    labeled_mask = np.asarray(pair["train_mask_a"], dtype=bool)
    n_labeled = int(np.count_nonzero(labeled_mask))
    n_unlabeled = int(labeled_mask.size - n_labeled)
    return {
        "dataset": DATASET,
        "method": method_name,
        "requested_samples_per_domain": requested_samples_per_domain,
        "samples_per_domain": int(pair["source_size"]),
        "combined_samples": int(pair["source_size"] + pair["target_size"]),
        "labeled_samples_per_domain": n_labeled,
        "unlabeled_samples_per_domain": n_unlabeled,
        "combined_labeled_samples": int(2 * n_labeled),
        "combined_unlabeled_samples": int(2 * n_unlabeled),
        "source_n_features": int(pair["source_n_features"]),
        "target_n_features": int(pair["target_n_features"]),
        "n_unique_classes": int(
            np.unique(
                np.concatenate(
                    [pair["labels_a_true"], pair["labels_b_true"]]
                )
            ).size
        ),
        "seed": seed,
        "runtime_sec": runtime_sec,
        "peak_mem_mb": peak_mem_mb,
        "status": status,
        "error": error,
    }


def save_metadata(output_dir, timestamp, base_pair):
    benchmark.write_json(
        output_dir / METADATA_FILENAME,
        {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "timestamp": timestamp,
            "dataset": DATASET,
            "methods": list(METHOD_CONFIGS),
            "method_configs": METHOD_CONFIGS,
            "sample_sizes_per_domain": SAMPLE_SIZES_PER_DOMAIN,
            "seeds": SEEDS,
            "rgbd_train_fraction": benchmark.RGBD_TRAIN_FRACTION,
            "label_masking": (
                "Deterministic stratified 50/50 train/test split in original "
                "row order, inherited from ad_real_multimodal.py."
            ),
            "subsampling": (
                "Deterministic label-stratified subsampling within the labeled "
                "and unlabeled pools using helpers from "
                "ad_real_multimodal.py. Smaller subsets preserve an exact "
                "50/50 visibility split; the 15k endpoint retains the original "
                "7,472/7,528 split."
            ),
            "sample_count_definition": (
                "samples_per_domain is the number of rows passed to "
                "fit_transform in each modality. Both labeled and unlabeled "
                "rows participate. Labeled/unlabeled columns describe only "
                "label visibility, not inclusion in model fitting."
            ),
            "original_n_samples_per_modality": int(
                base_pair["original_n_samples"]
            ),
            "maximum_loaded_samples_per_modality": int(
                base_pair["source_size"]
            ),
            "n_components": benchmark.N_COMPONENTS,
            "n_jobs": benchmark.N_JOBS,
            "max_fit_transform_sec": MAX_FIT_TRANSFORM_SEC,
            "metrics_computed": False,
        },
    )


def main():
    benchmark.set_active_dataset(DATASET)
    benchmark.validate_config()

    # Load the 15k benchmark pair once, then derive every smaller size from it
    # using the same deterministic stratification helper.
    base_pair = benchmark.build_pair()
    validate_config(base_pair)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / RESULTS_FILENAME
    save_metadata(output_dir, timestamp, base_pair)

    rows = []
    for requested_samples_per_domain in SAMPLE_SIZES_PER_DOMAIN:
        pair = make_scaling_pair(
            base_pair,
            requested_samples_per_domain,
        )
        n_labeled = int(np.count_nonzero(pair["train_mask_a"]))
        n_unlabeled = int(pair["train_mask_a"].size - n_labeled)
        print(
            f"\n### {pair['source_size']} samples per modality "
            f"({n_labeled} labeled, {n_unlabeled} unlabeled) ###"
        )

        for seed in SEEDS:
            for method_name in METHOD_CONFIGS:
                print(f"Running {method_name} | seed={seed}...")
                benchmark.seed_everything(seed)
                runtime_sec = np.nan
                peak_mem_mb = np.nan
                try:
                    runtime_sec, peak_mem_mb = profile_fit_transform(
                        method_name,
                        pair,
                        seed,
                    )
                    row = result_row(
                        method_name,
                        pair,
                        requested_samples_per_domain,
                        seed,
                        runtime_sec,
                        peak_mem_mb,
                        "ok",
                    )
                    print(
                        f"  {runtime_sec:.2f}s | {peak_mem_mb:.2f} MB peak"
                    )
                except Exception as exc:
                    runtime_sec = getattr(exc, "runtime_sec", runtime_sec)
                    peak_mem_mb = getattr(exc, "peak_mem_mb", peak_mem_mb)
                    status = (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else "error"
                    )
                    row = result_row(
                        method_name,
                        pair,
                        requested_samples_per_domain,
                        seed,
                        runtime_sec,
                        peak_mem_mb,
                        status,
                        error=str(exc),
                    )
                    print(f"  FAILED: {status}: {exc}")

                rows.append(row)
                benchmark.append_result_row(results_csv, row)

    results_df = pd.DataFrame(rows).sort_values(
        ["samples_per_domain", "seed", "method"],
        kind="stable",
    )
    results_df.to_csv(results_csv, index=False)
    print(f"\nFinished. Results saved to: {results_csv}")


if __name__ == "__main__":
    main()
