from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase
from Defenses.ImageDefenses.Defend import Defender

def main():
    defend_name = 'feature_distillation'

    # dataset
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    # Use deterministic preprocessing for defense evaluation. Random training
    # augmentation would change both the learned trigger placement and the DCT
    # calibration statistics from run to run.
    dataset = ImageDataset(label_path=label_path, transform=None, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=512,
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

    defender = Defender(classification.model, dataset, test_loader, calibration_loader=train_loader)

    if defend_name == "feature_distillation":
        print("Feature Distillation")
        print(defender.feature_distillation(
                trigger_path='/home/oraja001/Jlab/TimeSeriesAttack/backups/adversarial_patch/latest_trigger.pth',
                source_filter='bad',
                how_to_attach='blend',
                QS=1,
                preserve_ratio=0.0,
                fd_batch_size=16,
                save_examples_dir='backups/feature_distillation_examples',
                max_saved_examples=5,
            )
        )
    
    if defend_name == "difussion_purification":
        print("Difussion Purification")
        print(defender.diffusion_purification(
                trigger_path="/home/oraja001/Jlab/TimeSeriesAttack/backups/adversarial_patch/latest_trigger.pth",
                diffusion_checkpoint_path='backups/diffusion_purifier/best_checkpoint.pth',
                source_filter='bad',
                how_to_attach='blend',
                diffusion_step=100,
                reverse_steps=None,
                stochastic=True,
                dp_batch_size=16
            )
        )

if __name__=='__main__':
    main()
