from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase
from Defenses.ImageDefenses.Defend import Defender

def main():
    defend_name = 'feature_distillation'  # {'feature_distillation', 'diffusion_purification', 'feature_squeezing'}

    # dataset
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    
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

    classification.load_checkpoint(f"backups/original_model/{classification.model_name}.pth")

    defender = Defender(classification.model, dataset, test_loader, calibration_loader=train_loader)

    trigger_path = '/home/oraja001/Jlab/TimeSeriesAttack/backups/learn_fixed_size_patch_no_mask_optimization_robust_uap_blend_count_1_size_608by256_epsilon_0.05_lr_0.0001_mlr_0.001_mask_weight_0_patch_weight_0/saved_trigger'
    print(f'{trigger_path}')

    if defend_name == "feature_distillation":
        print("Feature Distillation")
        print(defender.feature_distillation(
                trigger_path=trigger_path,
                QS=1,
                preserve_ratio=0.0,
                fd_batch_size=16,
                save_examples_dir='backups/feature_distillation_examples',
                max_saved_examples=5,
            )
        )
    
    if defend_name == "diffusion_purification":
        print("Diffusion Purification")
        print(defender.diffusion_purification(
                trigger_path=trigger_path,
                diffusion_checkpoint_path='backups/diffusion_purifier/best_checkpoint.pth',
                diffusion_step=100,
                reverse_steps=None,
                stochastic=True,
                dp_batch_size=16,
                save_examples_dir="backups/diffusion_purifier",
                max_saved_examples=5
            )
        )
    
    if defend_name == "feature_squeezing":
        print('Feature Squeezing')
        print(defender.feature_squeezing(trigger_path=trigger_path,
                sqz_threshold=0.08,
                save_examples_dir="backups/feature_squeezing",
                max_saved_examples=5
            )
        )


if __name__=='__main__':
    main()
