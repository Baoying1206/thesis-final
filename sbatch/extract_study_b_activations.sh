#!/bin/bash
#SBATCH --job-name=jb-studyb-extract
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/study_b_extract_%j.out

# Study B (RQ2, FINAL_STUDY_PROTOCOL.md Sec 5R) real extraction: 12
# conditions (3 families x P/N/S/C), 4-wave real multi-turn generation
# for P/N. direction_ids: 300 x 3 families x 6 expensive generation
# calls = 5,400/model (stage 4 not generated -- Sec 5R.8). validation_ids:
# 72 x 3 families x 10 expensive calls = 2,160/model + judging.
#
# Submit one job per (model, ids_key) with MODEL_IDX=0/1/2 and IDS_KEY:
#   sbatch --export=MODEL_IDX=0,IDS_KEY=direction_ids  sbatch/extract_study_b_activations.sh
#   sbatch --export=MODEL_IDX=0,IDS_KEY=validation_ids sbatch/extract_study_b_activations.sh
#   (repeat for MODEL_IDX=1,2 -- 6 jobs total for the full run)
#
# Prerequisite: sbatch/study_b_smoketest.sh must have passed first
# (this whole code path has never been run against real GPU before
# that). Pilot (sbatch/run_pilot_study_b_llama.sh) must also confirm
# clean before submitting the validation_ids array (Sec 5.5/5R.8
# discipline, unchanged).
#
# --time is REJECTED by this cluster's sbatch config (see
# run_formal_behavioral.sh's comment for the exact error) -- if a job
# gets SIGTERM'd mid-run (matches the real Gemma incident from
# Experiment 2's formal behavioral run), the fix is BATCH_SIZE, not a
# time limit.

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
IDS_KEY=${IDS_KEY:-direction_ids}   # direction_ids | validation_ids

if [ "$MODEL_IDX" = "2" ]; then
    DEFAULT_BATCH_SIZE=16   # Gemma: eager attention needed for correctness (see _behavioral_shared.py), smaller default batch than sdpa models
else
    DEFAULT_BATCH_SIZE=8
fi
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

echo "Model: $MODEL_ALIAS  ids_key=$IDS_KEY  batch_size=$BATCH_SIZE  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/extract_study_b_activations.py \
    --model-alias "$MODEL_ALIAS" \
    --ids-key     "$IDS_KEY" \
    --batch-size  "$BATCH_SIZE"

echo "Done: $(date)"
