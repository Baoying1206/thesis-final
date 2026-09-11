#!/bin/bash
#SBATCH --job-name=jb-exp1-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp1_analyze_%j.out

# CPU-only Experiment 1 geometry analysis (slurm/analyze_experiment1_geometry.py,
# FINAL_STUDY_PROTOCOL.md Sec 4.3). Loads the .pt files
# extract_experiment1_activations.py already saved -- no GPU, no model.
# Real data (3584-dim vectors, 29+ layers, 300 instructions) is much
# slower than this was validated against (synthetic 8-dim/4-layer
# fixtures) -- expect real wall time in the minutes, not seconds.
#
# Prerequisite: extract_experiment1_activations.sh must have completed for
# this model (real, not --limit) first -- check the manifest.jsonl has
# 2,400 lines, not 8 (an interrupted/smoke-test run).
#
# Defaults to --skip-layerwise-sweep (core bootstrap/reliability/cosine
# results only -- the full-layer sweep is a sensitivity check, not the
# primary analysis, Sec 4.2). Submit with SKIP_SWEEP=0 for the complete
# record including the sweep:
#   sbatch --export=MODEL_IDX=0                sbatch/analyze_experiment1_geometry.sh   # fast, skip sweep
#   sbatch --export=MODEL_IDX=0,SKIP_SWEEP=0    sbatch/analyze_experiment1_geometry.sh   # complete, slower
#
# Submit one job per model with MODEL_IDX=0/1/2.

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}
SKIP_SWEEP=${SKIP_SWEEP:-1}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  skip_sweep=$SKIP_SWEEP  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

SWEEP_FLAG=""
if [ "$SKIP_SWEEP" = "1" ]; then
    SWEEP_FLAG="--skip-layerwise-sweep"
fi

python3 slurm/analyze_experiment1_geometry.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER" \
    $SWEEP_FLAG

echo "Done: $(date)"
