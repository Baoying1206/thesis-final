#!/bin/bash
#SBATCH --job-name=jb-histco-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/history_augmented_co_mg_smoketest_%j.out

# One-off sanity check BEFORE the real extract_history_augmented_co_mg.sh
# array. RQ2 Round 20 replacement design (history-augmented canonical
# CO/MG, FINAL_STUDY_PROTOCOL.md Sec 13 Round 20) has NEVER been run
# against real GPU at all -- new code path (6-stage real trajectory:
# 5 frozen scaffold turns + 1 canonical-mechanism turn, 21 conditions (7 mechanisms x 3 forms);
# scaffold lengthened from an initial 3-turn draft to address a
# statistical-power concern raised before any real confirmatory
# extraction). --limit 1 covers exactly one instruction's worth of all
# 21 conditions (7 mechanisms x {multi_neutral,multi_progressive}(6 stages each)/single(1 turn)). Qwen only,
# direction_ids only (cheaper -- no stage-4 generation needed there).
#
# Delete slurm/history_augmented_co_mg_output/ contents afterward if you
# don't want this smoke test's 1-instruction output mixed in with the
# real run.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "=== dry-run, direction_ids (tokenizer only, stage-1 multi + full single rendering) ==="
python3 slurm/extract_history_augmented_co_mg.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --dry-run

echo "=== small real GPU extraction test, direction_ids (--limit 1 = 1 instruction x 21 conditions) ==="
python3 slurm/extract_history_augmented_co_mg.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --limit 1

echo "=== output check ==="
ls -la slurm/history_augmented_co_mg_output/
