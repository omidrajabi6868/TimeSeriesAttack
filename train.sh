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
  --patch-update-method robust_uap \
  --how-to-attach blend \
  --gradcam-examples 10 \
  --steps 50 \
  --learning-rate 0.1 \
  --epsilon 0.05 \
  --bandwidth 60 \
  --target-label 1 \
  --psp-num-copies 32 \
  --psp-feature-weight 1 \
  --psp-target-weight 1 \
  --target-eot-samples 8 \
  --psp-direction-warmup-steps 0 \
  --psp-direction-cosine-threshold 0.001 \
  --psp-activation-direction maximize \
  --source-filter bad \
  --no-optimize-mask \
  --no-multimodel-optimization \
  --aggregation mean \
  --attack-scope universal \
  --black-box-models AlexNet ResNet34 MobileNetV3Small SwinT InceptionV3 \
  --perturbation-mask-path /home/oraja001/Jlab/TimeSeriesAttack/assets/heatmap_binary_mask.png
  # --output-dir /home/oraja001/Jlab/TimeSeriesAttack/backups/single_model/AlexNet/perturbation_attack_targeted_psp_uap_blend_count_1_size_608by256_epsilon_0.05_lr_0.1_mlr_0.001_mask_weight_0.0_patch_weight_0.0_target_eot_8_dir_minimize_w0_r20_m3.0_gr0.5_spatial_mask_heatmap_binary_mask/
  
 