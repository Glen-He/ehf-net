#!/usr/bin/env bash
# 串联运行：preprocess build → stats → train（全程后台，SSH 断开不受影响）
#
# 日志结构：
#   logs/pipeline/pipeline_SUFFIX.log   ← pipeline 脚本自身的状态 + preprocess 子进程 stdout
#   logs/nohup/nohup_train_SUFFIX.log   ← train 进程的 stdout/stderr（nohup 副产品）
#   logs/preprocess/                    ← preprocess.py 内部 logging 写入（Python 负责）
#   logs/train/                         ← train.py 内部 logging 写入（Python 负责）
#
# 用法：
#   bash scripts/run_pipeline.sh          # 普通运行（自动后台化，SSH 断开安全）
#   SMOKE=1 bash scripts/run_pipeline.sh  # smoke 模式
set -euo pipefail

DATA_ROOT="data/processed/hiqbind"
SMOKE="${SMOKE:-}"
RUN_SUFFIX="$(date '+%Y-%m-%d_%H-%M-%S')"

mkdir -p logs/nohup logs/pipeline

# 若 stdout 是终端，说明用户直接运行；自动 nohup 重执行并退出前台，
# 使整条 pipeline（包括前台的 preprocess 步骤）在 SSH 断开后仍能继续。
if [[ -t 1 ]]; then
    PIPELINE_LOG="logs/pipeline/pipeline_${RUN_SUFFIX}.log"
    echo "[pipeline] Detaching to background → ${PIPELINE_LOG}"
    nohup bash "$0" >> "${PIPELINE_LOG}" 2>&1 &
    echo "[pipeline] PID=$!  tail -f ${PIPELINE_LOG}"
    exit 0
fi

# ── 以下在后台非终端环境中执行，stdout 已指向 pipeline_SUFFIX.log ──────────────

log() {
    echo "[pipeline] $(date '+%H:%M:%S') $*"
}

log "RUN_SUFFIX=${RUN_SUFFIX}  DATA_ROOT=${DATA_ROOT}  SMOKE=${SMOKE:-off}"

# ── 1. preprocess build ──────────────────────────────────────────────────────
# 子进程的 stdout/stderr 自然流入 pipeline_SUFFIX.log（进度条可见）；
# Python logging 另行写入 logs/preprocess/preprocess_build_SUFFIX.log。
log "Starting preprocess build..."
uv run python scripts/preprocess.py build \
    --data-root "${DATA_ROOT}"
log "preprocess build done."

# ── 2. preprocess stats ──────────────────────────────────────────────────────
log "Starting preprocess stats..."
uv run python scripts/preprocess.py stats \
    --data-root "${DATA_ROOT}"
log "preprocess stats done."

# ── 3. train（独立后台，stdout/stderr 写 nohup/nohup_train_SUFFIX.log）────────
# train.py 内部 logging 另行写入 logs/train/train_SUFFIX.log。
SMOKE_FLAG=""
[[ -n "${SMOKE}" ]] && SMOKE_FLAG="--smoke"

TRAIN_NOHUP_LOG="logs/nohup/nohup_train_${RUN_SUFFIX}.log"
log "Starting training in background → ${TRAIN_NOHUP_LOG}"

nohup uv run python train.py \
    --data_root "${DATA_ROOT}" \
    ${SMOKE_FLAG} \
    --run_suffix "${RUN_SUFFIX}" \
    >> "${TRAIN_NOHUP_LOG}" 2>&1 &

TRAIN_PID=$!
log "train PID=${TRAIN_PID}"
log "  tail -f ${TRAIN_NOHUP_LOG}"
