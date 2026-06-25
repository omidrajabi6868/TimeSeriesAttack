from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase
from Defenses.ImageDefenses.Defend import Defender

def main():
    defend_name = 'feature distillation'

    # dataset
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    train_transform = ImageDataset.default_train_augmentation(image_size=image_size)
    dataset = ImageDataset(label_path=label_path, transform=train_transform, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=1024,
        stratify_by_bad_sample=True,
    )
    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} bad_ratio: {split_info["bad_ratio"]:.4f}')

    # loading model
    classification = ClassificationBase(
        model_name='AlexNet', 
        optimizer_name='Adam', 
        checkpoint_dir='backups'
    )

    classification.load_checkpoint("backups/original_model/best_checkpoint.pth")

    


    return

if __name__=='__main__':
    main()