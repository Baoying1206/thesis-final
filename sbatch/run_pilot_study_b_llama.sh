#!/bin/bash
#SBATCH --job-name=jb-studyb-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/study_b_pilot_%j.out

# Study B pilot (slurm/run_pilot_study_b_llama.py, FINAL_STUDY_PROTOCOL.md
# Sec 5R.8). Llama-3.1-8B-Instruct ONLY, the same 30 fixed ids as Sec
# 5.5's original pilot, all 12 conditions -- forces the FULL generate+
# judge path at every stage (not the cheaper direction_ids-role
# estimate in Sec 5R.8 -- see run_pilot_study_b_llama.py's docstring
# for why). PILOT_NON_RESULT -- mechanical sanity check only. Its
# output must NEVER be used to select, revise, or tune any condition's
# stage wording.
#
# Prerequisite: sbatch/study_b_smoketest.sh must have passed first.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "Start: $(date)"
python3 slurm/run_pilot_study_b_llama.py
echo "Done: $(date)"
