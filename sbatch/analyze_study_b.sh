#!/bin/bash
#SBATCH --job-name=jb-studyb-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/study_b_analyze_%j.out

# CPU-only Study B analysis (slurm/analyze_study_b.py,
# FINAL_STUDY_PROTOCOL.md Sec 5R.4/5R.5/5R.6). Loads Study B's
# direction_ids/validation_ids .pt + judge output, plus Study A's
# (Experiment 1's) raw .pt files for the frozen p_CO/p_MG reference
# directions (Sec 5R.4.7) -- no GPU, no model.
#
# Prerequisite: extract_study_b_activations.sh for BOTH ids_key values
# for this model (representation-only analysis is runnable with just
# direction_ids; behavioral and the activation-behavior connection
# need validation_ids too). Experiment 1 (Study A) must already be
# extracted for this model (it was, Round 12/13 -- this script does
# not re-run it).
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_study_b.sh
#
# Use SKIP_BEHAVIORAL=1 / SKIP_CONNECTION=1 to run representation-only
# before validation_ids has been extracted:
#   sbatch --export=MODEL_IDX=0,SKIP_BEHAVIORAL=1,SKIP_CONNECTION=1 sbatch/analyze_study_b.sh

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}
SKIP_BEHAVIORAL=${SKIP_BEHAVIORAL:-0}
SKIP_CONNECTION=${SKIP_CONNECTION:-0}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  skip_behavioral=$SKIP_BEHAVIORAL  skip_connection=$SKIP_CONNECTION  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

EXTRA_FLAGS=""
if [ "$SKIP_BEHAVIORAL" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-behavioral"
fi
if [ "$SKIP_CONNECTION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-connection"
fi

python3 slurm/analyze_study_b.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER" \
    $EXTRA_FLAGS

echo "Done: $(date)"
