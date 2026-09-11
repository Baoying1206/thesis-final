#!/bin/bash
#SBATCH --job-name=jb-exp2-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp2_smoketest_%j.out

# One-off sanity check BEFORE the real extract_experiment2_activations.sh
# array. extract_experiment2_activations.py has never been run against
# real GPU at all (unlike extract_experiment1_activations.py, which was
# validated this way already) -- it uses the 10-condition 3-turn template,
# a completely different code path from Experiment 1's 8-condition
# single-turn one. --limit 10 covers exactly all 10 conditions for the
# FIRST instruction of each ids_key (matches the lesson from Experiment 1:
# --limit must be a multiple of the condition count, or some conditions
# -- here in particular mg_encoding_obfuscation/mg_payload_splitting --
# never get exercised at all). Qwen only, both ids_key values.
#
# Delete slurm/experiment2_output/ contents afterward if you don't want
# this smoke test's 1-instruction output mixed in with the real run.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "=== dry-run, direction_ids (tokenizer only) ==="
python3 slurm/extract_experiment2_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --dry-run

echo "=== dry-run, validation_ids (tokenizer only) ==="
python3 slurm/extract_experiment2_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key validation_ids --dry-run

echo "=== small real GPU extraction test, direction_ids (10 rows = 1 instruction x 10 conditions) ==="
python3 slurm/extract_experiment2_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key direction_ids --limit 10

echo "=== small real GPU extraction test, validation_ids (10 rows = 1 instruction x 10 conditions) ==="
python3 slurm/extract_experiment2_activations.py --model-alias Qwen2.5-7B-Instruct --ids-key validation_ids --limit 10

echo "=== output check ==="
ls -la slurm/experiment2_output/
