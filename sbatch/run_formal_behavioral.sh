#!/bin/bash
#SBATCH --job-name=jb-formal-behav
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/formal_behav_%j.out
#SBATCH --time=00:45:00

# Experiment 2 FORMAL behavioral run (slurm/run_formal_behavioral.py,
# FINAL_STUDY_PROTOCOL.md Sec 7) -- NOT the pilot. Real scientific records
# (no "pilot" tag). 72 validation_ids x 10 conditions = 720 generations +
# 720 judgements per model.
#
# --time=00:45:00 added after a real cluster run (job 5143, Gemma) was
# SIGTERM'd at 14m29s -- almost certainly this cluster's default walltime
# for jobs that don't request one, contrary to this repo's general
# no-explicit---time convention (see sbatch/README-equivalent notes
# elsewhere). Qwen/Llama finished in ~9.5min under sdpa attention; Gemma
# now uses attn_implementation="eager" (see _behavioral_shared.py, needed
# for correct batched generation) which is slower, and pushed it over
# whatever the default limit is. 45min is generous headroom, not a tuned
# estimate.
#
# Submit one job per model with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 sbatch/run_formal_behavioral.sh   # Qwen
#   sbatch --export=MODEL_IDX=1 sbatch/run_formal_behavioral.sh   # Llama
#   sbatch --export=MODEL_IDX=2 sbatch/run_formal_behavioral.sh   # Gemma
#
# Prerequisite: --dry-run first for all 3 models (720/720 rows each,
# already confirmed locally). Must not run before the pilot has confirmed
# the pipeline is mechanically sound (Sec 5.5).

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

python3 slurm/run_formal_behavioral.py --model-alias "$MODEL_ALIAS"

echo "Done: $(date)"
