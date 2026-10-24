#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=logs/%j_out.txt   # %j inserts the Job ID automatically
#SBATCH --error=logs/%j_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=high-gpu-mem
#SBATCH --gres=gpu:2
#SBATCH --nodes=1

# Ensure the logs directory exists
mkdir -p logs

# Environment Setup
enable_lmod
module load container_env pytorch-gpu/2.2.0

# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

# crun python train_model.py

# Execute
crun python imageattack.py \
  --task perturbation_attack \
  --training \
  --image-size 608x256 \
  --batch-size 32 \
  --model-name InceptionV3 \
  --patch-size 608x256 \
  --patch-count 1 \
  --patch-update-method psp_uap \
  --how-to-attach blend \
  --steps 1000 \
  --learning-rate 0.01 \
  --epsilon 0.05 \
  --bandwidth 60 \
  --target-label 1 \
  --psp-num-copies 10 \
  --source-filter bad \
  --no-optimize-mask \
  --no-multimodel-optimization \
  --aggregation mean \
  --attack-scope universal \
  --black-box-models AlexNet ResNet34 MobileNetV3Small SwinT InceptionV3 \
  # --output-dir /home/oraja001/Jlab/TimeSeriesAttack/backups/single_model/SwinT/perturbation_attack_image_specific_robust_blend_count_1_size_608by256_epsilon_0.03_lr_0.05_mlr_0.001_mask_weight_0.0_patch_weight_0.0/
  # --perturbation-mask-path /home/oraja001/Jlab/TimeSeriesAttack/assets/heatmap_binary_mask.png \
 