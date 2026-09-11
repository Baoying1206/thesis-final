#!/bin/bash
#SBATCH --job-name=jb-exp1-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/exp1_smoketest_%j.out

# One-off sanity check BEFORE the real extract_experiment1_activations.sh
# array. Confirms torch/cuda are available in this venv, that --dry-run
# passes against the real cluster tokenizer, and that the never-yet-run
# GPU forward-pass/activation-save code path in
# slurm/extract_experiment1_activations.py works on 4 real rows before
# committing GPU time to the full 300 x 8 extraction. Qwen only (cheapest
# of the 3 models) -- delete slurm/experiment1_output/ contents afterward
# if you don't want this smoke-test's 4-row output mixed in with the real run.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "=== torch/cuda check ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

echo "=== dry-run (tokenizer only) ==="
python3 slurm/extract_experiment1_activations.py --model-alias Qwen2.5-7B-Instruct --dry-run

echo "=== small real GPU extraction test (4 rows only) ==="
python3 slurm/extract_experiment1_activations.py --model-alias Qwen2.5-7B-Instruct --limit 4

echo "=== output check ==="
ls -la slurm/experiment1_output/
