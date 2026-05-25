#!/usr/bin/env bash
# EHFNet training pipeline helper.
#
# This script is a thin, explicit wrapper around the real project commands:
#
#   fresh   Clean selected derived data, rebuild graph cache, recompute stats, then train.
#   resume  Resume training from a checkpoint without touching preprocess caches.
#   build   Rebuild graph/ESM preprocess cache only.
#   stats   Recompute dataset_profile.json only.
#   clean   Clean selected preprocess caches only.
#
# Common examples:
#
#   CLEAN_TARGET=graph bash scripts/run_pipeline.sh fresh
#   CLEAN_TARGET=all bash scripts/run_pipeline.sh fresh
#   CLEAN_TARGET=processed CONFIRM_PROCESSED_CLEAN=1 bash scripts/run_pipeline.sh fresh
#   RESUME_CKPT=checkpoints/.../latest_model.pt RUN_SUFFIX=old_suffix bash scripts/run_pipeline.sh resume
#   SMOKE=1 CLEAN_TARGET=none bash scripts/run_pipeline.sh fresh
#
# Cleanup meanings:
#   none: no deletion.
#   graph: delete data_root/cache, data_root/candidates, dataset_profile.json,
#          and preprocess_summary*.json. Keeps cleaned/ and ESM npz files.
#   esm: delete only ESM npz embeddings under cleaned/.
#   all: graph + esm via scripts/preprocess.py clean.
#   processed: delete every immediate child of DATA_ROOT except cleaned/ and index.csv.
#              Requires CONFIRM_PROCESSED_CLEAN=1.
#
# Runtime knobs:
#   DATA_ROOT=data/processed/hiqbind
#   CONFIG=configs/train.toml
#   DEVICE=cuda:0
#   NUM_WORKERS=8
#   RUN_SUFFIX=<timestamp by default>
#   SESSION_NAME=<auto by default>
#   PROFILE=default|48g
#   RUN_IN_TMUX=1|0
#   CLEAN_DRY_RUN=1
#   FORCE_REBUILD=1
#   RESUME_CKPT=/path/to/latest_model.pt
#   RESUME_BLIND_POOL_DIR=/path/to/blind_pool_cache
#   STOP_AFTER_EPOCH=50
#   TRAIN_ARGS="..."
#   PREPROCESS_ARGS="..."
#
# Logs:
#   logs/tmux/pipeline_<ACTION>_<RUN_SUFFIX>.log
#   logs/preprocess/<build|clean|stats>/...
#   logs/train/train_<RUN_SUFFIX>.log
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ACTION="${1:-help}"
DATA_ROOT="${DATA_ROOT:-data/processed/hiqbind}"
CONFIG="${CONFIG:-configs/train.toml}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date '+%Y-%m-%d_%H-%M-%S')}"
SESSION_NAME="${SESSION_NAME:-ehfnet_${ACTION}_${RUN_SUFFIX}}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"
SMOKE="${SMOKE:-}"
PROFILE="${PROFILE:-default}"
CLEAN_TARGET="${CLEAN_TARGET:-graph}"
CLEAN_DRY_RUN="${CLEAN_DRY_RUN:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
STATS_MAX_SAMPLES="${STATS_MAX_SAMPLES:-0}"
SAVE_DIR="${SAVE_DIR:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_BLIND_POOL_DIR="${RESUME_BLIND_POOL_DIR:-}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
PREPROCESS_ARGS="${PREPROCESS_ARGS:-}"
CONFIRM_PROCESSED_CLEAN="${CONFIRM_PROCESSED_CLEAN:-0}"

usage() {
    cat <<'EOF'
EHFNet pipeline helper

Usage:
  bash scripts/run_pipeline.sh fresh
  bash scripts/run_pipeline.sh resume
  bash scripts/run_pipeline.sh build
  bash scripts/run_pipeline.sh stats
  bash scripts/run_pipeline.sh clean
  bash scripts/run_pipeline.sh help

Fresh rebuild examples:
  CLEAN_TARGET=graph bash scripts/run_pipeline.sh fresh
  CLEAN_TARGET=all bash scripts/run_pipeline.sh fresh
  CLEAN_TARGET=processed CONFIRM_PROCESSED_CLEAN=1 bash scripts/run_pipeline.sh fresh

Resume example:
  RESUME_CKPT=checkpoints/train_x/latest_model.pt RUN_SUFFIX=train_x_suffix bash scripts/run_pipeline.sh resume

Cleanup targets:
  none       Keep all existing derived files.
  graph      Remove graph cache, candidates, dataset_profile.json, and preprocess summaries.
  esm        Remove only ESM embedding npz files under cleaned/.
  all        Remove graph cache and ESM embedding npz files.
  processed  Remove immediate DATA_ROOT children except cleaned/ and index.csv. Requires confirmation.

Key env vars:
  DATA_ROOT=data/processed/hiqbind
  DEVICE=cuda:0
  NUM_WORKERS=8
  PROFILE=default|48g
  RUN_IN_TMUX=1|0
  RUN_SUFFIX=<timestamp by default>
  RESUME_CKPT=/path/to/latest_model.pt
  RESUME_BLIND_POOL_DIR=/path/to/blind_pool_cache
  TRAIN_ARGS="--epochs 60 --skip_test_after_training"
EOF
}

quote() {
    printf "%q" "$1"
}

log() {
    echo "[pipeline] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

append_split_args() {
    local -n target_ref="$1"
    local raw_args="$2"

    [[ -z "${raw_args}" ]] && return 0

    local split_args=()
    # shellcheck disable=SC2206
    split_args=(${raw_args})
    target_ref+=("${split_args[@]}")
}

run_cmd() {
    log "+ $*"
    "$@"
}

maybe_launch_tmux() {
    if [[ "${RUN_IN_TMUX}" != "1" || "${IN_PIPELINE_TMUX:-0}" == "1" ]]; then
        return 0
    fi

    if ! command -v tmux >/dev/null 2>&1; then
        log "tmux is not installed or not on PATH. Set RUN_IN_TMUX=0 to run in the foreground."
        exit 1
    fi

    mkdir -p logs/tmux
    local tmux_log="logs/tmux/pipeline_${ACTION}_${RUN_SUFFIX}.log"
    local command=(
        "DATA_ROOT=$(quote "${DATA_ROOT}")"
        "CONFIG=$(quote "${CONFIG}")"
        "DEVICE=$(quote "${DEVICE}")"
        "NUM_WORKERS=$(quote "${NUM_WORKERS}")"
        "RUN_SUFFIX=$(quote "${RUN_SUFFIX}")"
        "SESSION_NAME=$(quote "${SESSION_NAME}")"
        "RUN_IN_TMUX=0"
        "IN_PIPELINE_TMUX=1"
        "SMOKE=$(quote "${SMOKE}")"
        "PROFILE=$(quote "${PROFILE}")"
        "CLEAN_TARGET=$(quote "${CLEAN_TARGET}")"
        "CLEAN_DRY_RUN=$(quote "${CLEAN_DRY_RUN}")"
        "FORCE_REBUILD=$(quote "${FORCE_REBUILD}")"
        "STATS_MAX_SAMPLES=$(quote "${STATS_MAX_SAMPLES}")"
        "SAVE_DIR=$(quote "${SAVE_DIR}")"
        "RESUME_CKPT=$(quote "${RESUME_CKPT}")"
        "RESUME_BLIND_POOL_DIR=$(quote "${RESUME_BLIND_POOL_DIR}")"
        "STOP_AFTER_EPOCH=$(quote "${STOP_AFTER_EPOCH}")"
        "TRAIN_ARGS=$(quote "${TRAIN_ARGS}")"
        "PREPROCESS_ARGS=$(quote "${PREPROCESS_ARGS}")"
        "CONFIRM_PROCESSED_CLEAN=$(quote "${CONFIRM_PROCESSED_CLEAN}")"
        "bash"
        "$(quote "$0")"
        "$(quote "${ACTION}")"
    )
    local command_text="${command[*]} 2>&1 | tee -a $(quote "${tmux_log}")"

    tmux new-session -d -s "${SESSION_NAME}" "${command_text}"
    log "Started tmux session: ${SESSION_NAME}"
    log "Attach: tmux attach -t ${SESSION_NAME}"
    log "Watch log: tail -f ${tmux_log}"
    exit 0
}

validate_action() {
    case "${ACTION}" in
        fresh | resume | build | stats | clean | help | -h | --help)
            return 0
            ;;
        *)
            log "Unknown action: ${ACTION}"
            usage
            exit 2
            ;;
    esac
}

smoke_args() {
    if [[ -n "${SMOKE}" ]]; then
        echo "--smoke"
    fi
}

clean_processed_root() {
    if [[ "${CONFIRM_PROCESSED_CLEAN}" != "1" ]]; then
        log "Refusing CLEAN_TARGET=processed without CONFIRM_PROCESSED_CLEAN=1."
        exit 2
    fi

    if [[ ! -d "${DATA_ROOT}" ]]; then
        log "DATA_ROOT does not exist, nothing to clean: ${DATA_ROOT}"
        return 0
    fi

    log "Cleaning processed root while preserving cleaned/ and index.csv: ${DATA_ROOT}"
    local item name
    for item in "${DATA_ROOT}"/* "${DATA_ROOT}"/.[!.]* "${DATA_ROOT}"/..?*; do
        [[ -e "${item}" ]] || continue
        name="$(basename "${item}")"
        case "${name}" in
            cleaned | index.csv)
                log "Keeping ${item}"
                ;;
            *)
                if [[ "${CLEAN_DRY_RUN}" == "1" ]]; then
                    log "Dry-run remove ${item}"
                else
                    rm -rf -- "${item}"
                    log "Removed ${item}"
                fi
                ;;
        esac
    done
}

run_clean() {
    case "${CLEAN_TARGET}" in
        none)
            log "CLEAN_TARGET=none, skipping cleanup."
            ;;
        graph | esm | all)
            local clean_cmd=(uv run python scripts/preprocess.py clean --data-root "${DATA_ROOT}" --target "${CLEAN_TARGET}")
            if [[ "${CLEAN_DRY_RUN}" == "1" ]]; then
                clean_cmd+=(--dry-run)
            fi
            run_cmd "${clean_cmd[@]}"
            ;;
        processed)
            clean_processed_root
            ;;
        *)
            log "Invalid CLEAN_TARGET=${CLEAN_TARGET}. Expected none, graph, esm, all, or processed."
            exit 2
            ;;
    esac
}

run_build() {
    local build_cmd=(uv run python scripts/preprocess.py)

    if [[ -n "$(smoke_args)" ]]; then
        build_cmd+=("$(smoke_args)")
    fi

    build_cmd+=(
        build
        --data-root "${DATA_ROOT}"
        --device "${DEVICE}"
        --num-workers "${NUM_WORKERS}"
    )

    if [[ "${FORCE_REBUILD}" == "1" ]]; then
        build_cmd+=(--force-rebuild)
    fi
    append_split_args build_cmd "${PREPROCESS_ARGS}"

    run_cmd "${build_cmd[@]}"
}

run_stats() {
    local stats_cmd=(uv run python scripts/preprocess.py)

    if [[ -n "$(smoke_args)" ]]; then
        stats_cmd+=("$(smoke_args)")
    fi

    stats_cmd+=(stats --data-root "${DATA_ROOT}")

    if [[ "${STATS_MAX_SAMPLES}" != "0" ]]; then
        stats_cmd+=(--max-samples "${STATS_MAX_SAMPLES}")
    fi

    run_cmd "${stats_cmd[@]}"
}

add_profile_args() {
    local -n train_cmd_ref="$1"

    case "${PROFILE}" in
        default)
            ;;
        48g)
            train_cmd_ref+=(
                --train_cost_budget 3600000
                --val_cost_budget 3600000
                --blind_pool_cost_budget 3600000
                --final_topn_cost_budget 3600000
                --accumulation_steps 4
            )
            ;;
        *)
            log "Invalid PROFILE=${PROFILE}. Expected default or 48g."
            exit 2
            ;;
    esac
}

run_train() {
    local train_cmd=(uv run python train.py --config "${CONFIG}" --data_root "${DATA_ROOT}" --run_suffix "${RUN_SUFFIX}")

    if [[ -n "$(smoke_args)" ]]; then
        train_cmd+=("$(smoke_args)")
    fi
    if [[ -n "${SAVE_DIR}" ]]; then
        train_cmd+=(--save_dir "${SAVE_DIR}")
    fi
    if [[ -n "${STOP_AFTER_EPOCH}" ]]; then
        train_cmd+=(--stop_after_epoch "${STOP_AFTER_EPOCH}")
    fi

    add_profile_args train_cmd
    append_split_args train_cmd "${TRAIN_ARGS}"

    run_cmd "${train_cmd[@]}"
}

run_resume() {
    if [[ -z "${RESUME_CKPT}" ]]; then
        log "RESUME_CKPT is required for resume."
        exit 2
    fi

    local train_cmd=(
        uv run python train.py
        --config "${CONFIG}"
        --data_root "${DATA_ROOT}"
        --run_suffix "${RUN_SUFFIX}"
        --resume_ckpt "${RESUME_CKPT}"
    )

    if [[ -n "$(smoke_args)" ]]; then
        train_cmd+=("$(smoke_args)")
    fi
    if [[ -n "${SAVE_DIR}" ]]; then
        train_cmd+=(--save_dir "${SAVE_DIR}")
    fi
    if [[ -n "${RESUME_BLIND_POOL_DIR}" ]]; then
        train_cmd+=(--resume_blind_pool_dir "${RESUME_BLIND_POOL_DIR}")
    fi
    if [[ -n "${STOP_AFTER_EPOCH}" ]]; then
        train_cmd+=(--stop_after_epoch "${STOP_AFTER_EPOCH}")
    fi

    add_profile_args train_cmd
    append_split_args train_cmd "${TRAIN_ARGS}"

    run_cmd "${train_cmd[@]}"
}

main() {
    validate_action

    case "${ACTION}" in
        help | -h | --help)
            usage
            exit 0
            ;;
    esac

    maybe_launch_tmux

    log "ACTION=${ACTION} DATA_ROOT=${DATA_ROOT} DEVICE=${DEVICE} NUM_WORKERS=${NUM_WORKERS}"
    log "RUN_SUFFIX=${RUN_SUFFIX} PROFILE=${PROFILE} CLEAN_TARGET=${CLEAN_TARGET} SMOKE=${SMOKE:-off}"

    case "${ACTION}" in
        clean)
            run_clean
            ;;
        build)
            run_build
            ;;
        stats)
            run_stats
            ;;
        fresh)
            run_clean
            run_build
            run_stats
            run_train
            ;;
        resume)
            run_resume
            ;;
    esac

    log "Done."
}

main "$@"
