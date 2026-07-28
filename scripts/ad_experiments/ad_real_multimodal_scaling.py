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
# Short enough to observe transient native allocations while keeping the
# profiler overhead small relative to the methods being benchmarked.
MEMORY_SAMPLE_INTERVAL_SEC = 0.02

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
    """Measure only ``fit_transform`` runtime and induced memory footprint.

    The worker is fully initialized before the measurement boundary. The
    parent then samples the worker and all of its descendants, which includes
    process-based parallel workers created by a method.

    Three process-tree measures are retained because no single operating-system
    counter means "allocated memory" everywhere:

    * VMS growth captures native dense allocations even after paging or memory
      compression, and is the primary scaling metric.
    * USS + swap growth measures private committed memory when the OS permits
      access to it.
    * RSS growth records the resident working-set peak for diagnostics.

    The benchmark never silently substitutes RSS for USS.
    """
    queue = mp.Queue()
    start_event = mp.Event()
    process = mp.Process(
        target=_run_scaling_method_worker,
        args=(method_name, pair, seed, queue, start_event),
    )

    process.start()
    fit_started = False
    fit_start = None
    baseline_memory = None
    peak_memory_increase = _empty_memory_profile()
    deadline = (
        None
        if MAX_FIT_TRANSFORM_SEC is None
        else float("inf")
    )
    try:
        while True:
            if fit_started and deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"fit_transform timeout after "
                        f"{MAX_FIT_TRANSFORM_SEC} seconds"
                    )
                poll_seconds = min(MEMORY_SAMPLE_INTERVAL_SEC, remaining)
            else:
                poll_seconds = MEMORY_SAMPLE_INTERVAL_SEC

            try:
                message = queue.get(timeout=poll_seconds)
            except queue_module.Empty:
                message = None

            if fit_started:
                peak_memory_increase = _update_memory_profile(
                    peak_memory_increase,
                    baseline_memory,
                    _get_process_tree_memory_mb(process.pid),
                )

            if message is not None and message[0] == "ready":
                baseline_memory = _get_process_tree_memory_mb(
                    process.pid
                )
                if not np.isfinite(baseline_memory["vms_mb"]):
                    raise RuntimeError(
                        "Unable to read worker memory counters before "
                        "fit_transform."
                    )
                fit_started = True
                fit_start = time.perf_counter()
                deadline = (
                    None
                    if MAX_FIT_TRANSFORM_SEC is None
                    else fit_start + MAX_FIT_TRANSFORM_SEC
                )
                start_event.set()
                continue

            if message is not None:
                # Include the retained post-fit state in the peak before the
                # worker exits and its address space disappears.
                peak_memory_increase = _update_memory_profile(
                    peak_memory_increase,
                    baseline_memory,
                    _get_process_tree_memory_mb(process.pid),
                )
                break

            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    "fit_transform worker exited without returning a "
                    f"result (exit code {process.exitcode})."
                )
    except Exception as exc:
        if process.is_alive():
            process.terminate()
            process.join()
        exc.runtime_sec = (
            float(time.perf_counter() - fit_start)
            if fit_start is not None
            else np.nan
        )
        exc.memory_profile = peak_memory_increase
        exc.peak_mem_mb = peak_memory_increase["peak_vms_delta_mb"]
        raise
    finally:
        if process.is_alive():
            process.join(1)
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()

    if message[0] == "error":
        exc = RuntimeError(message[1])
        exc.runtime_sec = float(message[2])
        exc.memory_profile = peak_memory_increase
        exc.peak_mem_mb = peak_memory_increase["peak_vms_delta_mb"]
        raise exc

    runtime_sec = float(message[1])
    return runtime_sec, peak_memory_increase


def _run_scaling_method_worker(method_name, pair, seed, queue, start_event):
    """Initialize a method, then expose an exact fit-transform boundary."""
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

        queue.put(("ready",))
        start_event.wait()
        start = time.perf_counter()
        model.fit_transform(
            pair["x_a"],
            pair["x_b"],
            pair["labels_a_model"],
            pair["labels_b_model"],
        )
        queue.put(("ok", float(time.perf_counter() - start)))
    except Exception as exc:
        runtime_sec = (
            float(time.perf_counter() - start)
            if "start" in locals()
            else np.nan
        )
        queue.put(("error", str(exc), runtime_sec))


def _get_process_tree_memory_mb(root_pid):
    """Sample VMS, RSS, and (when available) USS + swap for a process tree."""
    unavailable = {
        "vms_mb": np.nan,
        "rss_mb": np.nan,
        "private_mb": np.nan,
        "private_available": False,
        "tree_complete": False,
    }
    try:
        root = psutil.Process(root_pid)
    except (psutil.Error, ProcessLookupError, TypeError, OSError):
        return unavailable
    tree_complete = True
    try:
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, ProcessLookupError, OSError):
        # Process enumeration can be restricted in containers and hardened
        # environments even when the benchmark worker itself is observable.
        processes = [root]
        tree_complete = False

    total_vms_bytes = 0
    total_rss_bytes = 0
    total_private_bytes = 0
    observed_process = False
    private_available = True
    for process in {proc.pid: proc for proc in processes}.values():
        try:
            memory_info = process.memory_info()
        except (psutil.Error, ProcessLookupError, OSError):
            tree_complete = False
            continue
        total_vms_bytes += int(memory_info.vms)
        total_rss_bytes += int(memory_info.rss)
        observed_process = True

        try:
            full_info = process.memory_full_info()
            uss_bytes = getattr(full_info, "uss", None)
            if uss_bytes is None:
                private_available = False
            else:
                total_private_bytes += int(uss_bytes)
                total_private_bytes += int(getattr(full_info, "swap", 0))
        except (psutil.Error, ProcessLookupError, OSError):
            private_available = False

    if not observed_process:
        return unavailable
    mib = float(1024 ** 2)
    return {
        "vms_mb": total_vms_bytes / mib,
        "rss_mb": total_rss_bytes / mib,
        "private_mb": (
            total_private_bytes / mib if private_available else np.nan
        ),
        "private_available": private_available,
        "tree_complete": tree_complete,
    }


def _empty_memory_profile():
    return {
        "peak_vms_delta_mb": 0.0,
        "peak_private_delta_mb": np.nan,
        "peak_rss_delta_mb": 0.0,
        "private_memory_available": True,
        "process_tree_complete": True,
    }


def _update_memory_profile(profile, baseline, current):
    if baseline is None:
        return profile
    updated = profile.copy()
    for source_key, result_key in (
        ("vms_mb", "peak_vms_delta_mb"),
        ("rss_mb", "peak_rss_delta_mb"),
    ):
        if np.isfinite(baseline[source_key]) and np.isfinite(
            current[source_key]
        ):
            updated[result_key] = max(
                updated[result_key],
                float(max(0.0, current[source_key] - baseline[source_key])),
            )

    private_available = (
        baseline["private_available"] and current["private_available"]
    )
    updated["private_memory_available"] &= private_available
    if private_available:
        private_delta = float(
            max(0.0, current["private_mb"] - baseline["private_mb"])
        )
        previous = updated["peak_private_delta_mb"]
        updated["peak_private_delta_mb"] = (
            private_delta
            if not np.isfinite(previous)
            else max(previous, private_delta)
        )
    else:
        updated["peak_private_delta_mb"] = np.nan

    updated["process_tree_complete"] &= bool(
        baseline["tree_complete"] and current["tree_complete"]
    )
    return updated


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
    memory_profile,
    status,
    error="",
):
    memory_profile = memory_profile or _empty_memory_profile()
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
        # Backward-compatible primary scaling column. This is deliberately VMS
        # growth, not RSS, so dense native allocations remain visible when an
        # OS compresses or evicts their pages.
        "peak_mem_mb": memory_profile["peak_vms_delta_mb"],
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
            "runtime_measurement": (
                "Wall-clock seconds measured inside the isolated worker, "
                "starting immediately before fit_transform and ending "
                "immediately after it returns."
            ),
            "peak_memory_measurement": (
                "Peak increase, relative to the pre-fit_transform baseline, "
                "sampled every "
                f"{MEMORY_SAMPLE_INTERVAL_SEC} seconds for the method worker "
                "and all observable descendants. peak_mem_mb is the peak "
                "virtual-address-space increase induced by fit_transform. This "
                "portable allocation/scaling metric remains visible through "
                "operating-system compression and paging."
            ),
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
                memory_profile = _empty_memory_profile()
                try:
                    runtime_sec, memory_profile = profile_fit_transform(
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
                        memory_profile,
                        "ok",
                    )
                    print(
                        f"  {runtime_sec:.2f}s | "
                        f"{memory_profile['peak_vms_delta_mb']:.2f} MB "
                        "peak memory"
                    )
                except Exception as exc:
                    runtime_sec = getattr(exc, "runtime_sec", runtime_sec)
                    memory_profile = getattr(
                        exc,
                        "memory_profile",
                        memory_profile,
                    )
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
                        memory_profile,
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
