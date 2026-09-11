#!/bin/bash
#SBATCH --job-name=jb-exp2-extract
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp2_extract_%j.out

# Experiment 2 (RQ2) raw activation extraction
# (slurm/extract_experiment2_activations.py, FINAL_STUDY_PROTOCOL.md Sec 6/6.2).
# direction_ids: 300 x 10 = 3,000 forward passes. validation_ids: 72 x 10 =
# 720 forward passes. Firewall (Sec 6): validation_ids activations are only
# ever used for the Sec 6.2 projection-vs-strict_success check -- this script
# has no generation/judge code in it at all.
#
# Submit one job per (model, ids_key) with MODEL_IDX=0/1/2 and IDS_KEY:
#   sbatch --export=MODEL_IDX=0,IDS_KEY=direction_ids  sbatch/extract_experiment2_activations.sh
#   sbatch --export=MODEL_IDX=0,IDS_KEY=validation_ids sbatch/extract_experiment2_activations.sh
#   (repeat for MODEL_IDX=1,2 -- 6 jobs total for the full run)
#
# Prerequisite: --dry-run first (CPU-only, no GPU) for both ids_key values.

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
PRIMARY_LAYERS=(16 19 25)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
PRIMARY_LAYER=${PRIMARY_LAYERS[$MODEL_IDX]}
IDS_KEY=${IDS_KEY:-direction_ids}   # direction_ids | validation_ids

echo "Model: $MODEL_ALIAS  expected_primary_layer=$PRIMARY_LAYER  ids_key=$IDS_KEY  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

# extract_experiment2_activations.py takes no --primary-layer flag -- see
# extract_experiment1_activations.sh's comment for why.
python3 slurm/extract_experiment2_activations.py \
    --model-alias   "$MODEL_ALIAS" \
    --ids-key       "$IDS_KEY"

echo "Done: $(date)"
