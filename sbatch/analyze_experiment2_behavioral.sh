#!/bin/bash
#SBATCH --job-name=jb-exp2-behav-analyze
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp2_behav_analyze_%j.out

# CPU-only Experiment 2 behavioral analysis
# (slurm/analyze_experiment2_behavioral.py, FINAL_STUDY_PROTOCOL.md Sec 7).
# Loads run_formal_behavioral.py's generation+judge JSONL -- no GPU, no
# model. NEVER point this at the pilot's output
# (slurm/pilot_output/pilot_*.jsonl) -- that is PILOT_NON_RESULT and must
# never be analyzed as if it were a real result (Sec 5.5).
#
# Prerequisite: run_formal_behavioral.sh must have completed for this model.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/analyze_experiment2_behavioral.sh

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 slurm/analyze_experiment2_behavioral.py --model-alias "$MODEL_ALIAS"

echo "Done: $(date)"
