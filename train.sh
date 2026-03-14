#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --error=classification.txt
#SBATCH --output=classification.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

export PYTHONUNBUFFERED=1

enable_lmod
module load container_env pytorch-gpu/2.2.0

crun python -u main.py