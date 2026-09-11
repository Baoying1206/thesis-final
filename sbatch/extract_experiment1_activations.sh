#!/bin/bash
#SBATCH --job-name=jb-exp1-extract
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp1_extract_%j.out

# Experiment 1 (RQ1) raw activation extraction
# (slurm/extract_experiment1_activations.py, FINAL_STUDY_PROTOCOL.md Sec 4.2).
# One forward pass per (condition, instruction_id) -- no .generate() -- 300 x 8
# = 2,400 forward passes for the model below.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/extract_experiment1_activations.sh   # Qwen
#   sbatch --export=MODEL_IDX=1 sbatch/extract_experiment1_activations.sh   # Llama
#   sbatch --export=MODEL_IDX=2 sbatch/extract_experiment1_activations.sh   # Gemma
#
# Prerequisite: run this with --dry-run locally/on a CPU node first (no GPU
# needed for that) and confirm 2,400/2,400 rows pass before submitting here.
# NEVER submitted for real before -- run --limit 4 first via an interactive
# srun to sanity-check the real forward-pass/save path before the full array.

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
# Frozen primary layer = floor(0.6 * n_layers), FINAL_STUDY_PROTOCOL.md Sec 4.2.
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  expected_primary_layer=$PRIMARY_LAYER  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

# extract_experiment1_activations.py takes no --primary-layer flag -- it
# derives the layer itself from MODEL_TOKENIZER_SOURCES (same values as
# PRIMARY_LAYERS above) and asserts it against the model's real config at
# load time. PRIMARY_LAYER here is echoed above only, for log sanity-checking.
python3 slurm/extract_experiment1_activations.py \
    --model-alias   "$MODEL_ALIAS"

echo "Done: $(date)"
