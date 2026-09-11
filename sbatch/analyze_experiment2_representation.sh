#!/bin/bash
#SBATCH --job-name=jb-exp2-repr-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp2_repr_analyze_%j.out

# CPU-only Experiment 2 representation analysis
# (slurm/analyze_experiment2_representation.py, FINAL_STUDY_PROTOCOL.md
# Sec 6/6.1/6.2). Loads extract_experiment2_activations.py's direction_ids
# AND validation_ids .pt output, plus run_formal_behavioral.py's judge
# output for the Sec 6.2 correlation check -- no GPU, no model.
# Real data (3584-dim vectors, 29+ layers, 300 instructions) is much
# slower than the synthetic fixtures this was validated against --
# expect real wall time in the minutes, dominated by the 2000-rep x
# 280-partition bootstrap regardless of the sweep flag below.
#
# Prerequisite: extract_experiment2_activations.sh for BOTH ids_key values
# for this model first (Sec 6.1 alone -- direction_ids only -- is
# runnable before run_formal_behavioral.sh; Sec 6.2's correlation check
# additionally needs that job's output, so it's skipped by default below
# until that's been run).
#
# Defaults to --skip-layerwise-sweep AND --skip-validation-correlation
# (core Sec 6.1 bootstrap/reliability/cosine results only). Override via
# SKIP_SWEEP=0 / SKIP_CORR=0:
#   sbatch --export=MODEL_IDX=0                          sbatch/analyze_experiment2_representation.sh   # fast, Sec 6.1 core only
#   sbatch --export=MODEL_IDX=0,SKIP_SWEEP=0              sbatch/analyze_experiment2_representation.sh   # + full-layer sweep
#   sbatch --export=MODEL_IDX=0,SKIP_CORR=0               sbatch/analyze_experiment2_representation.sh   # + Sec 6.2 (needs run_formal_behavioral.sh done)
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
SKIP_CORR=${SKIP_CORR:-1}

echo "Model: $MODEL_ALIAS  primary_layer=$PRIMARY_LAYER  skip_sweep=$SKIP_SWEEP  skip_corr=$SKIP_CORR  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

EXTRA_FLAGS=""
if [ "$SKIP_SWEEP" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-layerwise-sweep"
fi
if [ "$SKIP_CORR" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip-validation-correlation"
fi

python3 slurm/analyze_experiment2_representation.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER" \
    $EXTRA_FLAGS

echo "Done: $(date)"
