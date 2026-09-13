#!/bin/bash
#SBATCH --job-name=jb-histco-extract
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/history_augmented_co_mg_extract_%j.out

# RQ2 Round 20 replacement design real extraction: 21 conditions (7
# mechanism groups x 3 forms: single, multi_neutral, multi_progressive).
# Each multi_* form is an INDEPENDENT 6-stage real trajectory (5 frozen
# scaffold turns + 1 canonical-mechanism turn -- scaffold lengthened
# from an initial 3-turn draft, and a second 'progressive' scaffold
# added alongside 'neutral', both to address a statistical-power
# concern raised before any real confirmatory extraction).
# direction_ids: 300 x 7 mechanisms x 2 multi_* forms x 5 expensive
# generation calls each (scaffold stages only -- the final payload
# stage is not generated, no downstream use for direction estimation)
# = 21,000/model. validation_ids: 72 x 7 x 2 x 6 = 6,048 expensive
# calls/model + judging (this IS the design's single confirmatory
# dataset -- Sec 13 Round 20 confirmed no discovery/confirmation split
# for this design).
#
# Submit one job per (model, ids_key) with MODEL_IDX=0/1/2 and IDS_KEY:
#   sbatch --export=MODEL_IDX=0,IDS_KEY=direction_ids  sbatch/extract_history_augmented_co_mg.sh
#   sbatch --export=MODEL_IDX=0,IDS_KEY=validation_ids sbatch/extract_history_augmented_co_mg.sh
#   (repeat for MODEL_IDX=1,2 -- 6 jobs total for the full run)
#
# test_ids is deliberately NOT exposed here -- this design's confirmed
# single-stage plan (Sec 13 Round 20) never reads test_ids. If that
# decision is ever revisited, invoke
# slurm/extract_history_augmented_co_mg.py directly with
# --ids-key test_ids --mechanisms <...> (required, no default) rather
# than adding a convenience path to this wrapper.
#
# Prerequisite: sbatch/history_augmented_co_mg_smoketest.sh must have
# passed first (this whole code path has never been run against real
# GPU before that). Pilot
# (sbatch/run_pilot_history_augmented_llama.sh) must also confirm clean
# before submitting the validation_ids array (Sec 5.5/5R.8 discipline,
# unchanged).
#
# --time is REJECTED by this cluster's sbatch config -- if a job gets
# SIGTERM'd mid-run, the fix is BATCH_SIZE, not a time limit (confirmed
# not a real walltime limit this session -- a Qwen job ran 61+ minutes
# successfully at default batch_size).

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
IDS_KEY=${IDS_KEY:-direction_ids}   # direction_ids | validation_ids

if [ "$MODEL_IDX" = "2" ]; then
    DEFAULT_BATCH_SIZE=16   # Gemma: eager attention needed for correctness, smaller default batch than sdpa models
else
    DEFAULT_BATCH_SIZE=8
fi
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

echo "Model: $MODEL_ALIAS  ids_key=$IDS_KEY  batch_size=$BATCH_SIZE  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/extract_history_augmented_co_mg.py \
    --model-alias "$MODEL_ALIAS" \
    --ids-key     "$IDS_KEY" \
    --batch-size  "$BATCH_SIZE"

echo "Done: $(date)"
