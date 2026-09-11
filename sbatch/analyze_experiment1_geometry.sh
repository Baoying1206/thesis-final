#!/bin/bash
#SBATCH --job-name=jb-exp1-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp1_analyze_%j.out

# CPU-only Experiment 1 geometry analysis (slurm/analyze_experiment1_geometry.py,
# FINAL_STUDY_PROTOCOL.md Sec 4.3). Loads the .pt files
# extract_experiment1_activations.py already saved -- no GPU, no model.
# Includes the 2000-rep instruction-cluster bootstrap + the full-layer
# point-estimate sweep; a few minutes per model is expected.
#
# Prerequisite: extract_experiment1_activations.sh must have completed for
# this model (real, not --limit) first.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_experiment1_geometry.sh

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/analyze_experiment1_geometry.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER"

echo "Done: $(date)"
