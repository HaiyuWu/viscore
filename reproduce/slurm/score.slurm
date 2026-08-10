#!/bin/bash
# VIScore a whole run's epoch sweep in one job. This is the CHEAP measurement: ~7 s per
# checkpoint after the probe is encoded, against >=20x that for ONE eval.slurm array task.
#
#   sbatch --account=<acct> --partition=<part> --gres=gpu:1 --export=ALL,\
# PROJECT_DIR=/path/to/viscore,DATA_HOME=$VISCORE_HOME,SUBDIR=<run_dir>,TASK=pusht \
#     reproduce/slurm/score.slurm
#
# TASK is the viscore task name (pusht|reacher|tworoom|cube|maze2d) and selects the
# success criterion, i.e. the tolerance d_tol is measured against. It is NOT interchangeable with
# eval.slurm's EVAL_CONFIG: `pusht_d50` there corresponds to TASK=pusht HORIZON=10 here, because
# the goal offset d=50 raw steps is 10 action blocks at frameskip 5.

#SBATCH --job-name=vis_score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x_%j.out
#SBATCH --error=logs/slurm-%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the viscore checkout}"
DATA_HOME="${DATA_HOME:-${VISCORE_HOME:?set VISCORE_HOME or DATA_HOME to the artifact root}}"
SUBDIR="${SUBDIR:?set SUBDIR to the run directory under DATA_HOME}"
TASK="${TASK:?set TASK (pusht|reacher|tworoom|cube|maze2d)}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-1-10}"
PROBE_DIR="${PROBE_DIR:-$DATA_HOME/cache/probes}"
LATENTS_DIR="${LATENTS_DIR:-$DATA_HOME/cache/latents}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/results}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" logs

# the pinned stable-worldmodel resolves its cache from STABLEWM_HOME, so set it here
export STABLEWM_HOME="$DATA_HOME"
export PYTHONUNBUFFERED=1

PROBE="$PROBE_DIR/probe_${TASK}.npz"
if [ ! -f "$PROBE" ]; then
    echo "probe missing, building it first: $PROBE"
    "$PYTHON" -m viscore.cli probe --tasks "$TASK" --data-home "$DATA_HOME" --out-dir "$PROBE_DIR"
fi

echo "Job ${SLURM_JOB_ID:-local}  Node $(hostname)  Start: $(date -Is)"
echo "Subdir: $SUBDIR  Task: $TASK  Epochs: $EPOCHS"

# EXCLUDE="sobriety" (or "v s") scores a SUBSET and labels it VI/IS/I/... in the csv's `factors`
# column. Use it for policies with no sobriety estimand, or to skip the 64x512-rollout probe --
# but never compare a subset's number with a full VIS number.
srun "$PYTHON" -m viscore.cli score \
    --task "$TASK" \
    --probe "$PROBE" \
    --run-dir "$DATA_HOME/$SUBDIR" \
    --epochs "$EPOCHS" \
    --latents-dir "$LATENTS_DIR" \
    --baselines \
    ${HORIZON:+--horizon "$HORIZON"} \
    ${EXCLUDE:+--exclude $EXCLUDE} \
    --csv "$OUT_DIR/${SUBDIR}_${TASK}_vis.csv"

echo "Finished: $(date -Is)"
