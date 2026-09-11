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
# 280-partition bootstrap: ~1 minute per model on real-sized vectors
# (measured on synthetic fixtures at this scale, see FINAL_STUDY_PROTOCOL.md
# Sec 8's note on the vectorized cosine-matrix fast path).
#
# Prerequisite: extract_experiment2_activations.sh for BOTH ids_key values
# and run_formal_behavioral.sh must have completed for this model first.
# Use --skip-validation-correlation if you only want Sec 6.1 (direction_ids
# only) and haven't run the formal behavioral job yet.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_experiment2_representation.sh

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

python3 slurm/analyze_experiment2_representation.py \
    --model-alias   "$MODEL_ALIAS" \
    --primary-layer "$PRIMARY_LAYER"

echo "Done: $(date)"
