"""Portable experiment paths; override individual locations through environment variables."""
import os
from pathlib import Path


def _path(variable, default):
    return str(Path(os.environ.get(variable, default)).expanduser().resolve())


BASE_PATH = _path("RF_MALI_ROOT", Path(__file__).resolve().parent)
LUNG_BATCHES_DATA_PATH = _path("RF_MALI_LUNG_DATA", Path(BASE_PATH) / "data_sc/lung_batches.h5ad")
IMMUNE_BATCHES_DATA_PATH = _path("RF_MALI_IMMUNE_DATA", Path(BASE_PATH) / "data_sc/Immune_ALL_human.h5ad")
PAIRED_DATA_PATH = _path("RF_MALI_PAIRED_DATA", Path(BASE_PATH) / "data")
UCI_DATA_FOLDER = _path("RF_MALI_UCI_DATA", Path(BASE_PATH) / "data_uci")
RESULTS_PATH = _path("RF_MALI_RESULTS", Path(BASE_PATH) / "results_sc")
