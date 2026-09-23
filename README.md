# FoSTA-experiments

This repository contains the code for the multimodal integration, empirical
scaling, single-cell batch correction, and UCI experiments. The instructions
below describe setup and execution for reviewers.

## Setup

Run all commands from the project root. Use **Python 3.12.3**, the version used
for this project, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage the environment. Git must be installed to fetch the dependencies
specified by Git URLs in `requirements.txt`.

### 1. Extract the data

Place the supplied `data.zip` in the project root and extract it there:

```sh
unzip data.zip -d .
```

The extracted `data_*` folders should sit directly beside `scripts/` and `src/`,
with the following layout:

```text
data_ave/
data_har/
data_rgbd/
data_sc/
data_sketchy/
data_uci/
```

### 2. Create the environment and install requirements

```sh
uv python install 3.12.3
uv venv --python 3.12.3 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

These commands follow the [uv environment setup workflow](https://docs.astral.sh/uv/pip/environments/).
Activate `.venv` again when opening a new terminal before running experiments.

On Apple Silicon with Apple Clang 21, `louvain`'s bundled C library needs
compatibility flags when building from source. If installation fails with
`-Wuninitialized-const-pointer` or `-Wdefault-const-init-var-unsafe`, clear its
cached build and retry:

```sh
uv cache clean louvain
CFLAGS='-Wno-error=uninitialized-const-pointer -Wno-error=default-const-init-var-unsafe' \
  ARCHFLAGS='-arch arm64' uv pip install -r requirements.txt
```

The supplied requirements include `jax[cuda12]` for NVIDIA GPU support. For a
CPU-only or macOS setup, replace that entry with `jax` before installation.
They also include `rpy2` for Splatter, which requires an R installation available
to the environment.

## Run the experiments

Run each command from the project root with `.venv` activated. Experiment
settings are defined in the configuration sections near the top of the scripts;
these include datasets, methods, seeds, and sample sizes where applicable.

### Real-world multimodal data integration

Run the HAR, AVE, RGB-D, and Sketchy integration benchmarks:

```sh
python scripts/experiments/run_real_multimodal.py
```

Results are written under `results_multimodal/`.

### Empirical scaling validation

Measure runtime and peak memory as the number of samples increases:

```sh
python scripts/experiments/run_real_multimodal_scaling.py
```

This script reuses the multimodal data loading and experiment protocol. Results
are written under `results_multimodal_scaling/`.

### Single-cell batch correction

Run the lung batch correction experiments using `data_sc/lung_batches.h5ad`:

```sh
python scripts/experiments/run_lung_batch.py
```

Results are written under `results_sc_experiments/`.

### UCI experiments

Run the UCI benchmarks using the datasets in `data_uci/`:

```sh
python scripts/experiments/run_uci.py
```

Results are written under `results_uci/`.
