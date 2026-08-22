#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=logs/%j_out.txt   # %j inserts the Job ID automatically
#SBATCH --error=logs/%j_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1

# Ensure the logs directory exists
mkdir -p logs

# Environment Setup
enable_lmod
module load container_env pytorch-gpu/2.2.0

# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

# Execute
crun python imageattack.py \
  --task perturbation_attack \
  --training \
  --image-size 608x256 \
  --batch-size 32 \
  --model-name SwinT \
  --patch-size 608x256 \
  --patch-count 1 \
  --patch-update-method hp_uap \
  --how-to-attach blend \
  --steps 100 \
  --learning-rate 0.05 \
  --epsilon 0.03 \
  --bandwidth 60 \
  --target-label 1 \
  --source-filter bad \
  --no-optimize-mask 
