import argparse
from pathlib import Path

DEFENSES = ('feature_distillation', 'diffusion_purification', 'feature_squeezing')


def _size(value):
    """Parse an image size written as WIDTHxHEIGHT."""
    try:
        width, height = (int(part) for part in value.lower().split('x', 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('size must be WIDTHxHEIGHT, for example 608x256') from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError('size dimensions must be positive')
    return width, height


def _optional_int(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('value must be an integer or "none"') from exc


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate an image defense against a saved trigger.')
    parser.add_argument('--defend-name', choices=DEFENSES, default='feature_distillation')
    parser.add_argument('--label-path', default='/home/oraja001/Jlab/Hydra data/labels_v2.txt')
    parser.add_argument('--image-size', type=_size, default=(608, 256), metavar='WIDTHxHEIGHT')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--stratify-by-bad-sample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--model-name', default='AlexNet')
    parser.add_argument('--optimizer-name', default='Adam')
    parser.add_argument('--checkpoint-dir', default='backups/original_model')
    parser.add_argument('--checkpoint-path', default=None,
                        help='Classifier checkpoint (default: CHECKPOINT_DIR/MODEL_NAME.pth).')
    parser.add_argument('--trigger-path', default='/home/oraja001/Jlab/TimeSeriesAttack/backups/learn_fixed_size_patch_no_mask_optimization_robust_uap_blend_count_1_size_608by256_epsilon_0.05_lr_0.0001_mlr_0.001_mask_weight_0_patch_weight_0/saved_trigger')
    parser.add_argument('--max-saved-examples', type=int, default=5)
    parser.add_argument('--qs', type=int, default=1, help='Feature-distillation JPEG quality scale.')
    parser.add_argument('--preserve-ratio', type=float, default=0.0)
    parser.add_argument('--fd-batch-size', type=int, default=16)
    parser.add_argument('--fd-save-examples-dir', default='backups/feature_distillation_examples')
    parser.add_argument('--diffusion-checkpoint-path', default='backups/diffusion_purifier/best_checkpoint.pth')
    parser.add_argument('--diffusion-step', type=int, default=100)
    parser.add_argument('--reverse-steps', type=_optional_int, default=None, metavar='INT|none')
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dp-batch-size', type=int, default=16)
    parser.add_argument('--dp-save-examples-dir', default='backups/diffusion_purifier')
    parser.add_argument('--sqz-threshold', type=float, default=0.08)
    parser.add_argument('--sqz-save-examples-dir', default='backups/feature_squeezing')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from Dataset.DataManagement import ImageDataset
    from Tasks.ImageClassification import ClassificationBase
    from Defenses.ImageDefenses.Defend import Defender

    defend_name = args.defend_name

    # dataset
    label_path = args.label_path
    image_size = args.image_size
    
    dataset = ImageDataset(label_path=label_path, transform=None, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=args.batch_size,
        stratify_by_bad_sample=args.stratify_by_bad_sample,
    )
    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} bad_ratio: {split_info["bad_ratio"]:.4f}')

    # loading model
    classification = ClassificationBase(
        model_name=args.model_name,
        optimizer_name=args.optimizer_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    checkpoint_path = args.checkpoint_path or str(
        Path(args.checkpoint_dir) / f'{classification.model_name}.pth'
    )
    classification.load_checkpoint(checkpoint_path)

    defender = Defender(classification.model, dataset, test_loader, calibration_loader=train_loader)

    trigger_path = args.trigger_path
    print(f'{trigger_path}')

    if defend_name == "feature_distillation":
        print("Feature Distillation")
        print(defender.feature_distillation(
                trigger_path=trigger_path,
                QS=args.qs,
                preserve_ratio=args.preserve_ratio,
                fd_batch_size=args.fd_batch_size,
                save_examples_dir=args.fd_save_examples_dir,
                max_saved_examples=args.max_saved_examples,
            )
        )
    
    if defend_name == "diffusion_purification":
        print("Diffusion Purification")
        print(defender.diffusion_purification(
                trigger_path=trigger_path,
                diffusion_checkpoint_path=args.diffusion_checkpoint_path,
                diffusion_step=args.diffusion_step,
                reverse_steps=args.reverse_steps,
                stochastic=args.stochastic,
                dp_batch_size=args.dp_batch_size,
                save_examples_dir=args.dp_save_examples_dir,
                max_saved_examples=args.max_saved_examples,
            )
        )
    
    if defend_name == "feature_squeezing":
        print('Feature Squeezing')
        print(defender.feature_squeezing(trigger_path=trigger_path,
                sqz_threshold=args.sqz_threshold,
                save_examples_dir=args.sqz_save_examples_dir,
                max_saved_examples=args.max_saved_examples,
            )
        )


if __name__=='__main__':
    main()
