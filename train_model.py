import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase

def main():
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    train_transform = ImageDataset.default_train_augmentation(image_size=image_size)
    dataset = ImageDataset(label_path=label_path, transform=train_transform, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=512,
        stratify_by_bad_sample=True,
    )

    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} bad_ratio: {split_info["bad_ratio"]:.4f}')

    train_indices = train_loader.dataset.indices
    train_labels = [dataset.labels[idx] for idx in train_indices]
    bad_count = sum(1 for label in train_labels if label == 0)
    good_count = sum(1 for label in train_labels if label == 1)
    pos_weight = (bad_count / good_count) if good_count > 0 else 1.0
    weighted_sampler = ClassificationBase.build_weighted_sampler_from_labels(train_labels)
    if weighted_sampler is not None:
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=weighted_sampler,
            num_workers=train_loader.num_workers,
        )
    print(f'train_pos_weight_for_BCE: {pos_weight:.6f}')

    classification = ClassificationBase(
        model_name='AlexNet', 
        optimizer_name='Adam', 
        checkpoint_dir='backups'
    )

    classification.train_model(
            train_loader,
            val_loader,
            learning_rate=1e-4,
            epoch_num=50,
            resume=False,
            resume_from='backups/original_model/last_checkpoint.pth',
            pos_weight=pos_weight,
            noise_probability_check=True,
            noise_regularization_weight=0.05,
            input_shape=(3, image_size[1], image_size[0]),
        )

    test_metrics = classification.evaluate_model(test_loader=test_loader)
    print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    print(
        'test_good_accuracy: '
        f'{test_metrics["good_accuracy"]}, '
        f'test_bad_accuracy: {test_metrics["bad_accuracy"]}'
    )

    return

if __name__ == "__main__":
    main()