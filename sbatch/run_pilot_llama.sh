#!/bin/bash
#SBATCH --job-name=jb-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/pilot_%j.out

# Experiment 2 pilot (slurm/run_pilot_llama.py, FINAL_STUDY_PROTOCOL.md
# Sec 5.5). Llama-3.1-8B-Instruct ONLY, 30 fixed direction_ids, all 10
# conditions -- 300 generations + 300 WildGuard judgements.
# PILOT_NON_RESULT -- mechanical sanity check only. Its output must NEVER
# be used to select, revise, or tune any condition's template wording.
#
# Prerequisite: --dry-run first (CPU-only, no GPU) to confirm 300/300 rows
# render correctly. Run READY_FOR_PILOT (FINAL_STUDY_PROTOCOL.md Sec 5.5)
# must already be confirmed before this job is submitted for real.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "Start: $(date)"
python3 slurm/run_pilot_llama.py
echo "Done: $(date)"
