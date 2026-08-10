#!/bin/bash
# VIS-WM / LeWM training. Site-specific settings come from the environment, not from this file:
#
#   sbatch --account=<acct> --partition=<part> --gres=gpu:1 \
#          --export=ALL,PROJECT_DIR=/path/to/viscore,DATA_HOME=$VISCORE_HOME,\
# LOSS=visreg,LAMBDA=4.5,DATA=pusht,LR=1e-4,SEED=403 reproduce/slurm/train.slurm
#
# Knobs (all optional except where noted):
#   LOSS visreg|sigreg   LAMBDA   DATA pusht|dmc|tworoom|ogb   BATCH_SIZE   HISTORY   LR   SEED
#   CENTER_W SCALE_W SHAPE_W (VISReg component weights)   EXTRA (raw Hydra overrides)
#   RESUME_SUBDIR (reuse an existing run directory; spt.Manager resumes from its weights ckpt)
#
# The published recipe is 10 epochs, batch 128, AdamW + linear warmup/cosine decay:
#   VIS-WM   LOSS=visreg LAMBDA=4.5 LR=1e-4
#   LeWM     LOSS=sigreg LAMBDA=0.09 (default LeWM configuration)
# Two-Room additionally uses HISTORY=1 (its episodes are ~12 frameskip-5 steps long).
# LR is NOT portable across regularizers or datasets -- see README, "Training VIS-WM".

#SBATCH --job-name=viswm_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x_%j.out
#SBATCH --error=logs/slurm-%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the viscore checkout}"
DATA_HOME="${DATA_HOME:-${VISCORE_HOME:?set VISCORE_HOME or DATA_HOME to the artifact root}}"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR/viswm"

LOSS="${LOSS:-sigreg}"
LAMBDA="${LAMBDA:-0.09}"
CENTER_W="${CENTER_W:-1.0}"; SCALE_W="${SCALE_W:-1.0}"; SHAPE_W="${SHAPE_W:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DATA="${DATA:-pusht}"
HISTORY="${HISTORY:-3}"
LR="${LR:-5e-5}"
SEED="${SEED:-3072}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-16}"     # >=320G host RAM at 16; drop to 4 for a 200G budget

# Run directory name encodes every non-default knob, so a sweep is self-documenting on disk.
if [ -n "${RESUME_SUBDIR:-}" ]; then
    SUBDIR="$RESUME_SUBDIR"
else
    TAG="${LOSS}_${DATA}_lambd$(printf '%s' "$LAMBDA" | tr . p)"
    if [ "$CENTER_W$SCALE_W$SHAPE_W" != "1.01.01.0" ]; then
        TAG="${TAG}_c$(printf '%s' "$CENTER_W" | tr . p)_s$(printf '%s' "$SCALE_W" | tr . p)_h$(printf '%s' "$SHAPE_W" | tr . p)"
    fi
    [ "$BATCH_SIZE" != "128" ] && TAG="${TAG}_bs${BATCH_SIZE}"
    [ "$HISTORY" != "3" ]      && TAG="${TAG}_h${HISTORY}"
    [ "$LR" != "5e-5" ]        && TAG="${TAG}_lr$(printf '%s' "$LR" | tr . p | tr -d '+')"
    [ "$SEED" != "3072" ]      && TAG="${TAG}_seed${SEED}"
    SUBDIR="${TAG}_${SLURM_JOB_ID:-local}"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  Node $(hostname)  Start: $(date -Is)"
echo "Subdir: ${SUBDIR}"
echo "LOSS=$LOSS LAMBDA=$LAMBDA DATA=$DATA HISTORY=$HISTORY LR=$LR SEED=$SEED BATCH=$BATCH_SIZE"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# the pinned stable-worldmodel resolves its cache from STABLEWM_HOME, so set it here
export STABLEWM_HOME="$DATA_HOME"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

EXTRA_ARGS=""
if [ "$LOSS" = "visreg" ]; then
    EXTRA_ARGS="loss.regularizer.center_w=${CENTER_W} loss.regularizer.scale_w=${SCALE_W} loss.regularizer.shape_w=${SHAPE_W}"
fi

srun "$PYTHON" -u train.py \
    loss="${LOSS}" \
    loss.weight="${LAMBDA}" \
    ${EXTRA_ARGS} \
    loader.batch_size="${BATCH_SIZE}" \
    data="${DATA}" \
    wm.history_size="${HISTORY}" \
    optimizer.lr="${LR}" \
    seed="${SEED}" \
    trainer.max_epochs="${EPOCHS}" \
    num_workers="${NUM_WORKERS}" \
    subdir="${SUBDIR}" \
    ${EXTRA:-}

echo "Finished: $(date -Is)"
