#!/bin/bash
#SBATCH --job-name=jb-formal-behav
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/formal_behav_%j.out

# Experiment 2 FORMAL behavioral run (slurm/run_formal_behavioral.py,
# FINAL_STUDY_PROTOCOL.md Sec 7) -- NOT the pilot. Real scientific records
# (no "pilot" tag). 72 validation_ids x 10 conditions = 720 generations +
# 720 judgements per model.
#
# --time is REJECTED by this cluster's sbatch config (same error class as
# exp1_smoketest.sh's original --gres/--mem/--time -- "Requested node
# configuration is not available"), so the walltime can't be raised.
# A real cluster run (job 5143, Gemma) was SIGTERM'd at 14m29s -- almost
# certainly this cluster's fixed, non-configurable default walltime.
# Qwen/Llama finished in ~9.5min under sdpa attention and were unaffected;
# Gemma now needs attn_implementation="eager" (see _behavioral_shared.py,
# required for correct batched generation) which is slower. Fix is to
# make the job itself faster instead: BATCH_SIZE/WG_BATCH_SIZE default
# higher for Gemma (32/64 vs the script's own 8/16 defaults) -- an L40S
# has plenty of headroom for a 9B model at these batch sizes, and fewer,
# larger batches cuts wall-clock overhead substantially. Override via
# --export if this still doesn't fit under the default walltime.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/run_formal_behavioral.sh   # Qwen
#   sbatch --export=MODEL_IDX=1 sbatch/run_formal_behavioral.sh   # Llama
#   sbatch --export=MODEL_IDX=2 sbatch/run_formal_behavioral.sh   # Gemma
#
# Prerequisite: --dry-run first for all 3 models (720/720 rows each,
# already confirmed locally). Must not run before the pilot has confirmed
# the pipeline is mechanically sound (Sec 5.5).

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

if [ "$MODEL_IDX" = "2" ]; then
    DEFAULT_BATCH_SIZE=32
    DEFAULT_WG_BATCH_SIZE=64
else
    DEFAULT_BATCH_SIZE=8
    DEFAULT_WG_BATCH_SIZE=16
fi
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
WG_BATCH_SIZE=${WG_BATCH_SIZE:-$DEFAULT_WG_BATCH_SIZE}

echo "Model: $MODEL_ALIAS  batch_size=$BATCH_SIZE  wg_batch_size=$WG_BATCH_SIZE  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/run_formal_behavioral.py \
    --model-alias    "$MODEL_ALIAS" \
    --batch-size     "$BATCH_SIZE" \
    --wg-batch-size  "$WG_BATCH_SIZE"

echo "Done: $(date)"
