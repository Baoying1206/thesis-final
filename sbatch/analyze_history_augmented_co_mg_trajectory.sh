#!/bin/bash
#SBATCH --job-name=jb-histco-trajectory
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/history_augmented_co_mg_trajectory_%j.out

# CPU-only turn-wise (stage-by-stage) trajectory analysis for the RQ2
# Round 20 replacement design
# (slurm/analyze_history_augmented_co_mg_trajectory.py). Reads the SAME
# direction_ids .pt files as analyze_history_augmented_co_mg.py's
# run_representation, but every stage (stage_1_hook..stage_5_bridge,
# stage_6) instead of only the final one -- no re-extraction needed,
# these vectors were already saved by extract_history_augmented_co_mg.py.
# No GPU, no model.
#
# Exploratory/diagnostic only -- not a confirmatory hypothesis test.
# Checks whether the within-CO > within-MG cohesion pattern already
# appears during the mechanism-free scaffold turns or only emerges once
# the payload text appears at stage_6 (which should numerically match
# run_representation's own output for this model/kind -- a useful
# built-in consistency check).
#
# Substantially more compute than run_representation (roughly 6x: 6
# stages instead of 1) -- default --time above is a starting guess, not
# a validated real-cluster measurement; raise it if the job times out,
# or pass N_BOOT=500 (etc.) below for a faster first pass.
#
# Prerequisite: extract_history_augmented_co_mg.sh for direction_ids for
# this model (validation_ids/behavioral data is not used here).
# Experiment 1 (Study A) must already be extracted for this model.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_history_augmented_co_mg_trajectory.sh
#   sbatch --export=MODEL_IDX=0,N_BOOT=500 sbatch/analyze_history_augmented_co_mg_trajectory.sh

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}
N_BOOT=${N_BOOT:-2000}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  n_boot=$N_BOOT  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/analyze_history_augmented_co_mg_trajectory.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER" \
    --n-boot        "$N_BOOT"

echo "Done: $(date)"
