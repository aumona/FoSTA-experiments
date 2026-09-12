# RF-MALI

## Setup

From the repository root, with Conda available:

```sh
conda create -p ./.condaenv python=3.10.4
conda activate ./.condaenv
pip install -r requirements.txt
```

## Paths and launchers

Run Python experiment scripts from the repository root. Run notebooks with
`notebooks/` as the working directory; the RF-GAP demos use `src/rfgap/`.
Paths formerly tied to a workstation or cluster are now relative to these
working directories. Legacy notebooks that use MALI expect a sibling `MALI/`
checkout. Adjust dataset and saved-run paths for the experiments you use.

Scripts importing `personal_paths.py` support these environment overrides:

| Variable | Default relative to repository root |
| --- | --- |
| `RF_MALI_ROOT` | Location of `personal_paths.py` |
| `RF_MALI_LUNG_DATA` | `data_sc/lung_batches.h5ad` |
| `RF_MALI_IMMUNE_DATA` | `data_sc/Immune_ALL_human.h5ad` |
| `RF_MALI_PAIRED_DATA` | `data/` |
| `RF_MALI_UCI_DATA` | `data_uci/` |
| `RF_MALI_RESULTS` | `results_sc/` |

Experiment utilities live in `scripts/experiments/`. Local shell launchers use
`*_local.sh` names and locate the repository relative to the script, with an
optional `RF_MALI_ROOT` override. Their log directory can be overridden with
`RF_MALI_LOG_DIR`. Select the appropriate Python environment before running;
existing launcher activation commands may need adjustment for your installation.

Submit Slurm scripts from the repository root, or set `RF_MALI_ROOT` explicitly.
Slurm output defaults to `slurm-%x-%A_%a.out` and `.err` in the submission
directory. Override output/error paths with `sbatch --output=... --error=...`;
create the destination directory before submission. Email notifications are
opt-in through `sbatch --mail-user=... --mail-type=END,FAIL`.

Notebook outputs and execution metadata are cleared for sharing. Third-party
citations, source attribution, dependency URLs, and license text are retained.
Git history and ignored local datasets/results have not been anonymized.
