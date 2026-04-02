#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=logs/%j_out.txt   # %j inserts the Job ID automatically
#SBATCH --error=logs/%j_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# Ensure the logs directory exists
mkdir -p logs

# Environment Setup
enable_lmod
module load container_env pytorch-gpu/2.2.0

# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

# Execute
crun python imageattack.py