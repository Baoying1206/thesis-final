#!/bin/bash
#SBATCH --job-name=jb-tokaudit
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=sbatch/logs/tokaudit_%j.out

# CPU-only, no GPU, no model weights -- audits/audit_real_tokenizer_boundary.py
# (FINAL_STUDY_PROTOCOL.md Sec 11.1). Already run once for real (2026-09-11,
# all 3 models TOKEN_AUDIT_PASS) -- this script is for re-running after any
# future template/tokenizer change.

cd ~/thesis-final
mkdir -p sbatch/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "Start: $(date)"
python3 audits/audit_real_tokenizer_boundary.py | tee sbatch/logs/real_tokenizer_audit_output.json
echo "Done: $(date)"
