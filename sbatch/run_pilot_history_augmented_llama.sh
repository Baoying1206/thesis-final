#!/bin/bash
#SBATCH --job-name=jb-histco-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/history_augmented_co_mg_pilot_%j.out

# History-augmented canonical CO/MG pilot
# (slurm/run_pilot_history_augmented_llama.py). Llama-3.1-8B-Instruct
# ONLY, the same 30 fixed ids as Sec 5.5's original pilot / Study B's
# pilot, all 14 conditions -- forces the FULL generate+judge path at
# every stage. PILOT_NON_RESULT -- mechanical sanity check only. Its
# output must NEVER be used to select, revise, or tune the scaffold or
# any mechanism's wording.
#
# Prerequisite: sbatch/history_augmented_co_mg_smoketest.sh must have
# passed first.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "Start: $(date)"
python3 slurm/run_pilot_history_augmented_llama.py
echo "Done: $(date)"
