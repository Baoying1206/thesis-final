#!/bin/bash
#SBATCH --job-name=jb-histco-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/history_augmented_co_mg_analyze_%j.out

# CPU-only analysis for the RQ2 Round 20 replacement design
# (slurm/analyze_history_augmented_co_mg.py). Loads this design's
# direction_ids/validation_ids .pt + judge output, plus Experiment 1's
# raw .pt files for the frozen p_CO/p_MG reference directions -- no
# GPU, no model.
#
# Single-stage design (Sec 13 Round 20, confirmed with the user): the
# behavioral analysis here on validation_ids IS the final confirmatory
# result -- there is no second, test_ids-based confirmatory pass for
# this design.
#
# Prerequisite: extract_history_augmented_co_mg.sh for BOTH ids_key
# values for this model (representation-only analysis is runnable with
# just direction_ids; behavioral and the activation-behavior connection
# need validation_ids too). Experiment 1 (Study A) must already be
# extracted for this model (it was, Round 12/13 -- this script does not
# re-run it).
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_history_augmented_co_mg.sh
#
# Use SKIP_BEHAVIORAL=1 / SKIP_CONNECTION=1 to run representation-only
# before validation_ids has been extracted:
#   sbatch --export=MODEL_IDX=0,SKIP_BEHAVIORAL=1,SKIP_CONNECTION=1 sbatch/analyze_history_augmented_co_mg.sh

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}
SKIP_REPRESENTATION=${SKIP_REPRESENTATION:-0}
SKIP_BEHAVIORAL=${SKIP_BEHAVIORAL:-0}
SKIP_CONNECTION=${SKIP_CONNECTION:-0}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  skip_representation=$SKIP_REPRESENTATION  skip_behavioral=$SKIP_BEHAVIORAL  skip_connection=$SKIP_CONNECTION  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

EXTRA_FLAGS=""
if [ "$SKIP_REPRESENTATION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-representation"
fi
if [ "$SKIP_BEHAVIORAL" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-behavioral"
fi
if [ "$SKIP_CONNECTION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-connection"
fi

python3 slurm/analyze_history_augmented_co_mg.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER" \
    $EXTRA_FLAGS

echo "Done: $(date)"
