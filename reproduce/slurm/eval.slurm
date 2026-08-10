#!/bin/bash
# Planning evaluation of one epoch per array task. This is the EXPENSIVE measurement that
# VIScore is meant to predict -- 50 episodes of MPC per task.
#
#   EPOCHS=1,2,3,4,5,6,7,8,9,10 sbatch --array=0-9%8 \
#     --account=<acct> --partition=<part> --gres=gpu:1 --export=ALL,\
# PROJECT_DIR=/path/to/viscore,DATA_HOME=$VISCORE_HOME,SUBDIR=<run_dir>,EVAL_CONFIG=pusht \
#     reproduce/slurm/eval.slurm
#
# EPOCHS must be exported as a SHELL variable, not through --export: sbatch splits --export on
# commas, so `--export=ALL,EPOCHS=1,2,3` silently becomes EPOCHS=1 plus two bogus variables.
# Without EPOCHS the epoch is the array index + EP_OFFSET (default 1), which is the robust mode.
#
# EVAL_CONFIG is required and has no safe default: pass pusht | pusht_d50 | pusht_d75 | reacher |
# tworoom | cube. Protocol per config: goal d steps ahead on the same expert trajectory,
# num_eval=50, budget 2d, CEM n_steps 30 (PushT) or 10, evaluation seed fixed in the config.

#SBATCH --job-name=viswm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x_%A_%a.out
#SBATCH --error=logs/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the viscore checkout}"
DATA_HOME="${DATA_HOME:-${VISCORE_HOME:?set VISCORE_HOME or DATA_HOME to the artifact root}}"
SUBDIR="${SUBDIR:?set SUBDIR to the run directory under DATA_HOME}"
EVAL_CONFIG="${EVAL_CONFIG:?set EVAL_CONFIG explicitly (pusht|reacher|tworoom|cube|pusht_d50|pusht_d75)}"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR/viswm"
mkdir -p "$PROJECT_DIR/logs/eval"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "${EPOCHS:-}" ]; then
    IFS=',' read -ra EP_ARR <<<"$EPOCHS"
    if [ "$TASK_ID" -ge "${#EP_ARR[@]}" ]; then
        echo "FAILED: array task $TASK_ID exceeds the ${#EP_ARR[@]}-entry EPOCHS list"; exit 1
    fi
    EP=${EP_ARR[$TASK_ID]}
else
    EP=$(( TASK_ID + ${EP_OFFSET:-1} ))
fi

LOG="$PROJECT_DIR/logs/eval/${SUBDIR}_${EVAL_CONFIG}_epoch${EP}.log"
echo "Job ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}  Node $(hostname)  Start: $(date -Is)"
echo "Subdir: $SUBDIR  Config: $EVAL_CONFIG  Epoch: $EP  Log: $LOG"

# the pinned stable-worldmodel resolves its cache from STABLEWM_HOME, so set it here
export STABLEWM_HOME="$DATA_HOME"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

srun "$PYTHON" -u eval.py \
    --config-name="${EVAL_CONFIG}" \
    policy="${SUBDIR}/lewm_epoch_${EP}" \
    ${EXTRA:-} | tee "$LOG"

SUMMARY="$PROJECT_DIR/logs/eval/${SUBDIR}_${EVAL_CONFIG}_summary.txt"
metric=$(grep -m1 -oE "'success_rate': [0-9.]+" "$LOG" || echo "PARSE_ERROR")
echo "[$(date -Is)] [ep $EP] $metric" >> "$SUMMARY"
echo "DONE ep $EP : $metric"
