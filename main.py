from Dataset.DataManagement import ImageDataSet
from ImageClassification import ClassificationBase


def main():
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (640, 288)

    dataset = ImageDataSet(label_path=label_path, transform=None, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=64,
        stratify_by_bad_sample=True,
    )

    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} contains_bad_ratio: {split_info["contains_bad_ratio"]:.4f}')

    natural_trigger = dataset.find_natural_trigger_candidates(
        window_size=(48, 48),
        stride=24,
        top_k=3,
        max_samples_per_group=2000,
    )
    print('Natural trigger candidates (bad-containing vs [good, good]):')
    for candidate in natural_trigger['top_candidates']:
        print(candidate)

    classification = ClassificationBase(
        model_name='ResNet18',
        optimizer_name='Adam',
        checkpoint_dir='backups',
    )

    classification.load_checkpoint('backups/best_checkpoint.pth')

    test_metrics = classification.evaluate_model(test_loader=test_loader)
    print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    print(
        'test_good_good_accuracy: '
        f'{test_metrics["good_good_accuracy"]}, '
        f'test_others_accuracy: {test_metrics["others_accuracy"]}'
    )

    initial_backdoor_eval = classification.evaluate_backdoor_success(
        test_loader=test_loader,
        trigger_box=natural_trigger['top_candidates'][0],
        target_label=(0.0, 0.0),
        source_only_bad=True,
    )
    print(f'initial_backdoor_eval: {initial_backdoor_eval}')

    learned_trigger = classification.learn_universal_trigger(
        data_loader=train_loader,
        trigger_box=natural_trigger['top_candidates'][0],
        target_label=(0.0, 0.0),
        source_filter='bad',
        steps=80,
        learning_rate=0.05,
        epsilon=0.06,
        max_batches_per_step=2,
    )

    learned_backdoor_eval = classification.evaluate_backdoor_success(
        test_loader=test_loader,
        trigger_box=natural_trigger['top_candidates'][0],
        trigger_patch=learned_trigger['patch'],
        target_label=(0.0, 0.0),
        source_only_bad=True,
    )
    print(f'learned_backdoor_eval: {learned_backdoor_eval}')

    dataset.save_trigger_visualizations(
        trigger_analysis=natural_trigger,
        output_dir='trigger_visualization',
        num_examples=4,
        trigger_box=natural_trigger['top_candidates'][0],
        trigger_delta=learned_trigger['patch'],
    )
    print('Saved trigger visualizations to trigger_visualization/')


if __name__ == '__main__':
    main()
