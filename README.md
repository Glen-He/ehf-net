# EHFNet

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Dependency Manager: uv](https://img.shields.io/badge/dependency-uv-purple)](https://github.com/astral-sh/uv)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

EHFNet is a research codebase for protein-ligand docking with an equivariant hierarchical flow model. It builds cached protein-ligand graphs, trains a proposal-guided local docking model, and evaluates candidate poses with RMSD-first validation and Top-N blind metrics.

The practical training flow is:

```text
raw data -> prepare_data.py -> processed/cleaned + index.csv
processed data -> preprocess.py build -> graph cache + ESM/cache metadata
cached graphs -> preprocess.py stats -> dataset_profile.json
cache + profile -> train.py -> checkpoints + validation/test reports
```

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Install](#install)
- [Data Layout](#data-layout)
- [Recommended Pipeline](#recommended-pipeline)
- [Preprocess Cache Rules](#preprocess-cache-rules)
- [Training](#training)
- [Resume Training](#resume-training)
- [Logs and Outputs](#logs-and-outputs)
- [Monitoring Metrics](#monitoring-metrics)
- [Direct CLI Reference](#direct-cli-reference)
- [Project Structure](#project-structure)

## What This Project Does

EHFNet models ligand docking as conditional flow matching over rigid-body motion and torsion updates. The current training path is proposal-guided local docking:

1. Predict residue-level candidate binding centers.
2. Crop a local protein context around the center.
3. Run local flow docking for translation, rotation, and torsion.
4. Rerank generated poses with pose confidence and center confidence.

Main implementation features:

- Hierarchical heterograph input with ligand atoms, ligand molecule context, protein atoms, protein residues, and protein context nodes.
- Static graph preprocessing plus dynamic runtime edges and local crops.
- ESM residue embeddings, cached under `cleaned/` and embedded into graph cache.
- Cost-aware batching with adaptive shrink/recovery after real CUDA OOM events.
- EMA validation, scaffold split persistence, checkpoint resume, and final Top-N blind evaluation.
- Local center recall metrics during validation, including `local_center_recall@1_4a`, `local_center_recall@3_4a`, `local_center_recall@K_4a`, and `local_center_mean_min_dist`.

The model assumes a rigid protein during training and evaluation. Ligand initial poses are generated through the `flow_start_pos` cache path.

## Install

This project uses Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Glen-He/EHFNet.git
cd EHFNet
uv sync
```

The CUDA/PyG wheels are configured in `pyproject.toml` for the current Torch/CUDA stack. If your machine uses a different CUDA build, update the `torch`, `torchvision`, `torchaudio`, and PyG wheel settings together.

## Data Layout

Raw data is organized by dataset name:

```text
data/raw/hiqbind/
├── ligand/
│   ├── 1abc_ligand.sdf
│   └── ...
├── protein/
│   ├── 1abc_protein.pdb
│   └── ...
└── index.csv
```

`index.csv` must contain these columns:

```csv
Concatenated ID,Log Binding Affinity
1abc,6.5
2def,7.2
```

Prepare a processed dataset:

```bash
uv run python scripts/prepare_data.py hiqbind
```

The command reads `data/raw/hiqbind/` and writes:

```text
data/processed/hiqbind/
├── cleaned/
│   ├── 1abc/
│   │   ├── 1abc_ligand.sdf
│   │   ├── 1abc_protein.pdb
│   │   └── 1abc_esm_chainseg_esmc_300m.npz  # generated when ESM is used
│   └── ...
└── index.csv
```

`prepare_data.py` does not modify files under `data/raw/`.

## Recommended Pipeline

Use `scripts/run_pipeline.sh` for normal work. It runs in a detached `tmux` session by default, so closing SSH will not interrupt it. Pipeline logs go to `logs/tmux/`.

Show help:

```bash
bash scripts/run_pipeline.sh help
```

Start from scratch after graph feature, topology, or preprocessing changes:

```bash
PROFILE=48g CLEAN_TARGET=graph bash scripts/run_pipeline.sh fresh
```

More conservative 24G/default run:

```bash
CLEAN_TARGET=graph bash scripts/run_pipeline.sh fresh
```

Rebuild everything derived from graph and ESM cache:

```bash
PROFILE=48g CLEAN_TARGET=all bash scripts/run_pipeline.sh fresh
```

Remove every generated item under `DATA_ROOT` except `cleaned/` and `index.csv`, then rebuild:

```bash
PROFILE=48g \
CLEAN_TARGET=processed \
CONFIRM_PROCESSED_CLEAN=1 \
bash scripts/run_pipeline.sh fresh
```

Preview graph cleanup without deleting anything:

```bash
RUN_IN_TMUX=0 CLEAN_TARGET=graph CLEAN_DRY_RUN=1 bash scripts/run_pipeline.sh clean
```

Run only preprocessing build:

```bash
CLEAN_TARGET=none bash scripts/run_pipeline.sh build
```

Run only dataset statistics:

```bash
RUN_IN_TMUX=0 bash scripts/run_pipeline.sh stats
```

Common environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_ROOT` | `data/processed/hiqbind` | Processed dataset root containing `index.csv`. |
| `CONFIG` | `configs/train.toml` | Main training config. |
| `DEVICE` | `cuda:0` | Device used by preprocess and train. |
| `NUM_WORKERS` | `8` | Parallel workers for graph rebuild. |
| `PROFILE` | `default` | `default` uses config budgets; `48g` uses larger cost budgets. |
| `CLEAN_TARGET` | `graph` | Cleanup scope before `fresh`. |
| `RUN_IN_TMUX` | `1` | Set `0` for foreground execution. |
| `RUN_SUFFIX` | timestamp | Shared suffix for logs and checkpoints. |
| `TRAIN_ARGS` | empty | Extra flags appended to `train.py`. |
| `PREPROCESS_ARGS` | empty | Extra flags appended to `preprocess.py build`. |

The `48g` profile currently appends:

```bash
--train_cost_budget 3600000 \
--val_cost_budget 3600000 \
--blind_pool_cost_budget 3600000 \
--final_topn_cost_budget 3600000 \
--accumulation_steps 4
```

## Preprocess Cache Rules

The processed dataset root contains both stable inputs and derived files:

```text
data/processed/hiqbind/
├── cleaned/                 # prepared ligand/protein files; usually keep this
├── index.csv                # dataset index; usually keep this
├── cache/                   # graph cache generated by preprocess build
├── candidates/              # generated candidate/cache artifacts
├── dataset_profile.json     # continuous feature and distance statistics
└── preprocess_summary*.json # preprocess summaries
```

Cleanup choices used by `run_pipeline.sh`:

| `CLEAN_TARGET` | Deletes | Keeps | Use when |
| --- | --- | --- | --- |
| `none` | Nothing | Everything | You only want to train or inspect. |
| `graph` | `cache/`, `candidates/`, `dataset_profile.json`, preprocess summaries | `cleaned/`, ESM `.npz`, `index.csv` | Graph topology, atom/residue features, preprocessing logic, or geometry filters changed. |
| `esm` | ESM `.npz` files under `cleaned/` | Graph cache unless separately removed | ESM files are stale or corrupted. |
| `all` | `graph` + `esm` | `cleaned/`, `index.csv` | ESM model/dim changed, raw structures changed, or you want a full derived-cache rebuild. |
| `processed` | Every immediate child except `cleaned/` and `index.csv` | `cleaned/`, `index.csv` | You want to clear miscellaneous legacy generated folders. Requires confirmation. |

Direct cleanup commands are also available:

```bash
uv run python scripts/preprocess.py clean --data-root data/processed/hiqbind --target graph
uv run python scripts/preprocess.py clean --data-root data/processed/hiqbind --target all --dry-run
```

Graph rebuild and statistics:

```bash
uv run python scripts/preprocess.py build \
    --data-root data/processed/hiqbind \
    --device cuda:0 \
    --num-workers 8

uv run python scripts/preprocess.py stats \
    --data-root data/processed/hiqbind
```

`dataset_profile.json` is generated by `stats`. It stores continuous-feature and distance statistics. If a model cutoff is set to `auto`, `train.py` reads this file to resolve suggested cutoffs.

## Training

The simplest training command is:

```bash
uv run python train.py --config configs/train.toml --data_root data/processed/hiqbind
```

For regular training, prefer the pipeline wrapper:

```bash
PROFILE=48g CLEAN_TARGET=none bash scripts/run_pipeline.sh fresh
```

Use `CLEAN_TARGET=none` only when graph cache and `dataset_profile.json` are already current.

Configuration is split into two files:

| File | Purpose |
| --- | --- |
| `configs/train.toml` | Data path, device, training schedule, batching budgets, validation, split, ranking, blind pool, checkpoint behavior. |
| `configs/model.toml` | Architecture, RBF, geometry, edge construction, prediction head, flow matching, and loss weights. |

CLI flags override config values. To see all supported training flags:

```bash
uv run python train.py --help
```

Useful overrides:

```bash
uv run python train.py \
    --config configs/train.toml \
    --data_root data/processed/hiqbind \
    --device cuda:0 \
    --epochs 100 \
    --train_cost_budget 3600000 \
    --val_cost_budget 3600000 \
    --blind_pool_cost_budget 3600000 \
    --final_topn_cost_budget 3600000 \
    --accumulation_steps 4 \
    --run_suffix 48g_manual
```

Cost budgets are not sample counts. They are project-specific estimates based on graph nodes, edges, dynamic edges, and torsion work. The trainer starts from the configured budget, retries by splitting batches when possible, shrinks budgets after repeated real OOM events, and can recover budgets after clean windows.

## Resume Training

Resume from a full training checkpoint with the pipeline wrapper:

```bash
PROFILE=48g \
RESUME_CKPT=checkpoints/train_48g_run/latest_model.pt \
RUN_SUFFIX=48g_run_resume \
bash scripts/run_pipeline.sh resume
```

Resume into the same run suffix:

```bash
PROFILE=48g \
RESUME_CKPT=checkpoints/train_48g_run/latest_model.pt \
RUN_SUFFIX=48g_run \
bash scripts/run_pipeline.sh resume
```

Resume while explicitly reusing an old blind pool cache:

```bash
PROFILE=48g \
RESUME_CKPT=checkpoints/train_48g_run/model_epoch_49.pt \
RESUME_BLIND_POOL_DIR=checkpoints/train_48g_run/blind_pool_cache \
STOP_AFTER_EPOCH=60 \
RUN_SUFFIX=48g_resume_to_60 \
bash scripts/run_pipeline.sh resume
```

Recommended resume checkpoints:

- `latest_model.pt`
- `model_epoch_XX.pt`

`best_*` checkpoints are intended for selection/evaluation and are not the preferred resume entry. The resume path restores model, optimizer, scheduler, EMA, trainer state, budget controller state, best metrics, OOM counters, and RNG state when those fields are present in the checkpoint.

Keep the same total `--epochs` value when resuming. `--stop_after_epoch` is a 1-based inclusive absolute epoch number.

## Logs and Outputs

Pipeline and preprocess logs:

| Path | Meaning |
| --- | --- |
| `logs/tmux/pipeline_<action>_<suffix>.log` | Shell-level pipeline log from `run_pipeline.sh`. |
| `logs/preprocess/build/` | Graph build logs. |
| `logs/preprocess/clean/` | Cleanup logs. |
| `logs/preprocess/stats/` | Dataset statistics logs. |
| `logs/train/train_<suffix>.log` | Python training log. |
| `logs/smoke/...` | Smoke-run logs when `--smoke` or `SMOKE=1` is used. |

Training artifacts:

```text
checkpoints/train_<run_suffix>/
├── latest_model.pt
├── model_epoch_XX.pt
├── best_model.pt
├── best_selected_model.pt
├── best_composite_model.pt
├── best_rmsd_model.pt
├── best_single_shot_success2a_model.pt
├── blind_pool_cache/
└── reports/
    └── test_metrics.json
```

Attach to a running pipeline:

```bash
tmux attach -t <session_name>
```

Watch logs without attaching:

```bash
tail -f logs/tmux/pipeline_<action>_<suffix>.log
tail -f logs/train/train_<suffix>.log
```

## Monitoring Metrics

Validation reports include:

- `val_loss`
- `mean_rmsd_final`
- `median_rmsd_final`
- `single_shot_success_2a`
- `single_shot_success_5a`
- `local_center_recall@1_4a`
- `local_center_recall@3_4a`
- `local_center_recall@<center_proposal_topk>_4a`
- `local_center_mean_min_dist`
- cost guard skips and OOM/retry counters

Final test reports, when enabled with `--run_test_after_training`, include proposal recall, best-of-k upper bound, final ranking performance, failure decomposition, and Top-N success rates for the configured `--test_topk` values.

`blind_pool_refresh_every` controls periodic blind candidate pool refresh. This step is expected to be slower than ordinary training batches because it runs candidate generation/scoring over a larger validation-style pool and writes replay artifacts for later reranker supervision.

## Direct CLI Reference

Use these commands when you do not want the pipeline wrapper:

```bash
uv run python scripts/prepare_data.py --help
uv run python scripts/preprocess.py --help
uv run python scripts/preprocess.py build --help
uv run python scripts/preprocess.py clean --help
uv run python scripts/preprocess.py stats --help
uv run python train.py --help
```

Common direct commands:

```bash
uv run python scripts/prepare_data.py hiqbind

uv run python scripts/preprocess.py clean \
    --data-root data/processed/hiqbind \
    --target graph

uv run python scripts/preprocess.py build \
    --data-root data/processed/hiqbind \
    --device cuda:0 \
    --num-workers 8

uv run python scripts/preprocess.py stats \
    --data-root data/processed/hiqbind

uv run python train.py \
    --config configs/train.toml \
    --data_root data/processed/hiqbind
```

## Project Structure

```text
.
├── configs/
│   ├── model.toml              # model, graph, flow, loss settings
│   └── train.toml              # data, training, batching, validation settings
├── scripts/
│   ├── prepare_data.py         # raw -> processed/cleaned + index.csv
│   ├── preprocess.py           # clean/build/stats graph cache manager
│   ├── probe_cost_budget.py    # budget probing helper
│   └── run_pipeline.sh         # tmux pipeline wrapper
├── src/ehfnet/
│   ├── contracts/              # cache/checkpoint/blind-pool signatures
│   ├── data/                   # datasets, featurizers, preprocess helpers
│   ├── geometry/               # SE(3), torsion, RMSD utilities
│   ├── graph/                  # schema, builders, topology, crop, costs
│   ├── models/                 # EHFNet model, layers, heads
│   ├── runtime/                # config, logging, factories
│   └── training/               # trainer, validation, resume, blind inference
├── train.py                    # training entry point
├── pyproject.toml              # uv project and dependency metadata
└── uv.lock
```

## Notes

- This is research code. Prefer pinned configs and saved command logs for reproducible experiments.
- If graph features, topology, preprocessing filters, or model feature schemas change, rebuild graph cache before training.
- If the ESM model name or ESM dimension changes, rebuild both ESM and graph cache.
- If `dataset_profile.json` is missing, training can still start unless a config value needs an `auto` cutoff, but regenerating stats is recommended after every graph rebuild.
