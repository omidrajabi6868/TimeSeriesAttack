# TimeSeriesAttack

A PyTorch research framework for studying adversarial attacks, backdoor attacks, defenses, and model transferability on time-series-derived images and multivariate time-series data. The current image pipeline supports binary good/bad classification, universal perturbations, image-specific attacks, multi-model optimization, black-box evaluation, spatial perturbation masks, and Grad-CAM analysis.

## Main capabilities

- Train and evaluate image classifiers such as AlexNet, ResNet, InceptionV3, MobileNetV3, and Swin Transformer.
- Generate universal adversarial perturbations (UAPs).
- Generate and save one perturbation per eligible validation/test image.
- Optimize against one model or an ensemble distributed across GPUs.
- Evaluate a saved perturbation on white-box and black-box models without retraining it.
- Restrict perturbations to informative image regions with a resized binary mask.
- Save clean/adversarial examples, metrics, checkpoints, and Grad-CAM comparisons.
- Evaluate feature distillation, diffusion purification, and feature squeezing defenses.
- Build forecasting datasets and PatchTST-style time-series models.

## Repository structure

```text
imageattack.py          # Universal and image-specific attack CLI
imagedefend.py          # Defense evaluation CLI
train_model.py          # Classifier training example
Dataset/                # Image and time-series datasets
Network/                # Classifiers, metrics, VAE/UNet components
Attacks/ImageAttacks/   # Attack algorithms, PSP/Robust UAP, Grad-CAM
Attacks/TimeSeriesAttacks/
Defenses/ImageDefenses/ # Purification and squeezing defenses
Tasks/                  # Classification and forecasting tasks
tests/                  # Regression tests for attacks, metrics, and models
assets/                  # Example spatial perturbation mask
```

## Installation

```bash
git clone https://github.com/omidrajabi6868/TimeSeriesAttack.git
cd TimeSeriesAttack
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib seaborn opencv-python pytest
```

Install PyTorch using the command recommended for your CUDA version when running on GPUs.

## Image dataset format

Pass a comma-separated label file through `--label-path`. Each valid row contains an image path followed by `good` or `bad`:

```text
images/sample_001.png,good
images/sample_002.png,bad
```

Relative image paths are resolved relative to the label file. Images are converted to RGB and resized to `--image-size`, written as `WIDTHxHEIGHT`.

## Train a classifier

`train_model.py` is a configurable code example for stratified splitting, weighted sampling, training, checkpointing, and binary evaluation. Update its label path, image size, and model name, then run:

```bash
python train_model.py
```

Classifier checkpoints are expected under `backups/original_model/<ModelName>.pth` unless an explicit path is passed.

## Universal attack example

```bash
python imageattack.py \
  --task perturbation_attack \
  --training \
  --attack-scope universal \
  --label-path /path/to/labels.txt \
  --image-size 608x256 \
  --batch-size 32 \
  --model-name InceptionV3 \
  --patch-size 608x256 \
  --patch-count 1 \
  --patch-update-method pgd_sign \
  --how-to-attach blend \
  --steps 100 \
  --learning-rate 0.05 \
  --epsilon 0.03 \
  --target-label 1 \
  --source-filter bad \
  --no-optimize-mask
```

Universal methods include `deepfool_uap`, `mi_fgsm`, `pgd_sign`, `adam`, `gd_uap`, `gap_uap`, `hp_uap`, `fg_uap`, `robust_uap`, `psp_uap`, and `targeted_psp_uap`.

## Image-specific attack example

```bash
python imageattack.py \
  --training \
  --attack-scope image_specific \
  --label-path /path/to/labels.txt \
  --model-name AlexNet \
  --patch-update-method mi_fgsm \
  --image-size 608x256 \
  --patch-size 608x256 \
  --steps 100 \
  --epsilon 0.03 \
  --target-label 1 \
  --source-filter bad
```

Image-specific methods include `fgsm`, `ifgsm`, `mi_fgsm`, `pgd`, `adam`, `deepfool`, `robust`, and `hp`. Artifacts are stored separately for validation and test samples and can be evaluated later without regenerating them.

## Multi-model and black-box evaluation

Add ensemble optimization with:

```bash
--multimodel-optimization \
--ensemble-models ResNet34 AlexNet MobileNetV3Small SwinT \
--aggregation mean \
--gpu-ids 0 1
```

Evaluate transferability on models excluded from optimization with:

```bash
--black-box-models InceptionV3 MobileNetV3Small
```

The saved perturbation is applied unchanged to each black-box classifier.

## Spatial perturbation mask

Use a binary image to protect blank or uninformative regions:

```bash
--perturbation-mask-path assets/heatmap_binary_mask.png
```

Nonzero mask pixels allow perturbation; zero pixels are protected. The mask is resized to `--image-size` with nearest-neighbor interpolation and works with both universal and image-specific attacks. When this option is omitted, no external spatial restriction is applied.

## PSP-UAP controls

Both data-free and targeted hybrid PSP modes are available. Important options include:

```text
--psp-activation-direction {maximize,minimize,auto,off}
--psp-model-activation-directions MODEL=DIRECTION ...
--psp-prior {auto,gauss,jigsaw}
--psp-num-copies N
--psp-feature-weight FLOAT
--psp-target-weight FLOAT
--target-eot-samples N
```

Run `python imageattack.py --help` for the complete scheduling and gradient-balancing controls.

## Evaluate a saved attack

Repeat the training command with `--no-training` and the same settings or pass `--output-dir` explicitly. The script loads `saved_trigger`, evaluates validation/test ASR, performs optional black-box evaluation, and saves visualizations.

## Defenses

```bash
python imagedefend.py \
  --defend-name feature_squeezing \
  --label-path /path/to/labels.txt \
  --model-name AlexNet \
  --trigger-path /path/to/saved_trigger
```

Available defenses are `feature_distillation`, `diffusion_purification`, and `feature_squeezing`.

## Tests

```bash
pytest -q
```

The tests cover classifier construction, attack metrics, image-specific artifacts, multi-model aggregation, PSP-UAP behavior, activation-layer selection, trigger loading, and Grad-CAM output.
