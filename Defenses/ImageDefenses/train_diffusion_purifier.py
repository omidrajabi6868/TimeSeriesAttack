"""Train a dataset-specific diffusion purifier checkpoint.

Example:
    python -m Defenses.ImageDefenses.train_diffusion_purifier \
        --label-path /path/to/labels.txt \
        --image-width 608 --image-height 256 \
        --epochs 50 --batch-size 16 \
        --checkpoint-path backups/diffusion_purifier/best.pth
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.optim import AdamW

from Dataset.DataManagement import ImageDataset
from Defenses.ImageDefenses.DiffusionPurification import DiffusionPurifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DDPM used by DiffPure-style input purification.")
    parser.add_argument(
        "label_path_arg",
        nargs="?",
        help="Path to the image label file consumed by ImageDataset. Can also be passed with --label-path.",
    )
    parser.add_argument(
        "--label-path",
        dest="label_path_option",
        default=None,
        help="Path to the image label file consumed by ImageDataset.",
    )
    parser.add_argument("--image-width", type=int, default=None, help="Optional resize width.")
    parser.add_argument("--image-height", type=int, default=None, help="Optional resize height.")
    parser.add_argument("--checkpoint-path", default="backups/diffusion_purifier/best.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-multi-gpu", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA GPUs are available.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA GPU ids for DataParallel, for example '0,1'.")
    parser.add_argument("--save-every", type=int, default=0, help="Save numbered epoch checkpoints when > 0.")
    args = parser.parse_args()
    args.label_path = args.label_path_option or args.label_path_arg
    if args.label_path is None:
        parser.error("label path is required; pass it positionally or with --label-path")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("Both --image-width and --image-height are required when resizing is enabled.")
        image_size = (args.image_width, args.image_height)

    dataset = ImageDataset(label_path=args.label_path, transform=None, image_size=image_size)
    train_loader, val_loader, _ = dataset.train_val_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        stratify_by_bad_sample=True,
    )

    purifier = DiffusionPurifier(base_channels=args.base_channels, timesteps=args.timesteps).to(device)
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',') if gpu_id.strip()]
    if args.use_multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) > 1:
            print(f"Using DataParallel for diffusion model on GPUs: {gpu_ids}")
            purifier.model = torch.nn.DataParallel(purifier.model, device_ids=gpu_ids)
    optimizer = AdamW(purifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        purifier.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = purifier.training_loss(images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(purifier.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.shape[0]
            train_count += images.shape[0]

        purifier.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                loss = purifier.training_loss(images)
                val_loss_sum += float(loss.item()) * images.shape[0]
                val_count += images.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            purifier.save_checkpoint(checkpoint_path, extra={"epoch": epoch, "val_loss": val_loss})
            print(f"saved best checkpoint to {checkpoint_path}")
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_epoch_{epoch}{checkpoint_path.suffix}")
            purifier.save_checkpoint(epoch_path, extra={"epoch": epoch, "val_loss": val_loss})


if __name__ == "__main__":
    main()
