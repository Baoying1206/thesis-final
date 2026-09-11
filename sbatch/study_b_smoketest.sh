#!/bin/bash
#SBATCH --job-name=jb-studyb-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/study_b_smoketest_%j.out

# One-off sanity check BEFORE the real extract_study_b_activations.sh
# array. Study B (FINAL_STUDY_PROTOCOL.md Sec 5R) has NEVER been run
# against real GPU at all -- completely new code path (4-wave real
# multi-turn generation + per-stage activation extraction, 12
# conditions), unlike Experiment 1/2's proven single-forward-pass or
# flat-batch-generation patterns. --limit 1 covers exactly one
# instruction's worth of all 12 conditions (P/N x 4 stages each, S/C x
# 1 each). Qwen only, direction_ids only (cheaper -- no stage-4
# generation needed there, Sec 5R.8).
#
# Delete slurm/study_b_output/ contents afterward if you don't want
# this smoke test's 1-instruction output mixed in with the real run.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "=== dry-run, direction_ids (tokenizer only, stage-1 P/N + full S/C rendering) ==="
python3 slurm/extract_study_b_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --dry-run

echo "=== small real GPU extraction test, direction_ids (--limit 1 = 1 instruction x 12 conditions) ==="
python3 slurm/extract_study_b_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --limit 1

echo "=== output check ==="
ls -la slurm/study_b_output/
