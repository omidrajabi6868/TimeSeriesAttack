import numpy as np
from Dataset.DataManagement import ImageDataSet
from ImageClassification import ClassificationBase
from AdversarialAttack import AdversarialAttack
from BackdoorAttack import BackdoorAttack
from Network.ImageVAE import ImageVAE


def main():
    task = 'adversarial_attack'
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

    classification = ClassificationBase(
        model_name='ResNet18', 
        optimizer_name='Adam', 
        checkpoint_dir='backups'
    )

    # classification.train_model(train_loader, val_loader, learning_rate=1e-4, epoch_num=10)

    # Resume example:
    # classification.train_model(
    #     train_loader,
    #     val_loader,
    #     learning_rate=1e-4,
    #     epoch_num=20,
    #     resume_from='backups/last_checkpoint.pth',
    # )

    classification.load_checkpoint("backups/best_checkpoint.pth")

    # test_metrics = classification.evaluate_model(test_loader=test_loader)
    # print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    # print(
    #     'test_good_good_accuracy: '
    #     f'{test_metrics["good_good_accuracy"]}, '
    #     f'test_others_accuracy: {test_metrics["others_accuracy"]}'
    # )

    if task == "adversarial_attack":
        adv_attack = AdversarialAttack(classification.model)
        natural_trigger = dataset.find_natural_trigger_candidates(
            window_size=(64, 32),
            stride=8,
            top_k=10,
            max_samples_per_group=2000,
        )
        print('Natural trigger candidates (bad-containing vs [good, good]):')
        for candidate in natural_trigger['top_candidates']:
            print(candidate)

        initial_attack_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            target_label=(1.0, 1.0),
            source_only_bad=True,
        )
        print(f'initial_adversarial_eval: {initial_attack_eval}')

        learned_trigger = adv_attack.learn_universal_trigger(
            data_loader=train_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            target_label=(1.0, 1.0),
            source_filter='bad',
            steps=100,
            learning_rate=0.01,
        )

        learned_backdoor_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            trigger_patch=learned_trigger['patch'],
            target_label=(1.0, 1.0),
            source_only_bad=True,
        )
        print(f'learned_adversarial_eval: {learned_backdoor_eval}')

        dataset.save_trigger_visualizations(
            trigger_analysis=natural_trigger,
            output_dir='backups/trigger_visualization',
            num_examples=4,
            trigger_box=natural_trigger['top_candidates'][0],
            trigger_delta=learned_trigger['patch'],
        )
        print('Saved trigger visualizations to trigger_visualization/')

    if task == 'backdoor_attack':
        vae_model = ImageVAE(
            image_channels=3,
            image_size=(image_size[1], image_size[0]),
            latent_dim=64,
        )
        backdoor_attack = BackdoorAttack(
            model=classification.model,
            vae_model=vae_model,
        )

        # Learn latent space for the dataset on image data.
        vae_history = backdoor_attack.fit_vae(
            train_loader=train_loader,
            epochs=5,
            learning_rate=1e-3,
            beta=1.0,
            log_interval=1,
        )
        print(f'vae_training_last_epoch: {vae_history[-1] if vae_history else {}}')

        latent_space = backdoor_attack.build_latent_space(train_loader)
        latent_vectors = latent_space['latents']
        latent_labels = latent_space['labels']

        # Cluster the latent space to several clusters (adjustable).
        clustering = backdoor_attack.cluster_latent_space(
            latent_vectors=latent_vectors,
            num_clusters=6,
            max_iters=50,
        )
        print(f"cluster_count: {clustering['num_clusters']}")

        # Learn one cluster with a balanced [good, good] and bad-containing mix as backdoor samples.
        cluster_selection = backdoor_attack.select_balanced_cluster(
            cluster_assignments=clustering['assignments'],
            labels=latent_labels,
            min_samples=16,
        )
        print(f"selected_cluster: {cluster_selection['selected_cluster']}")
        print(f"cluster_stats: {cluster_selection['cluster_stats']}")

        backdoor_result = backdoor_attack.learned_backdoor(
            data_loader=train_loader,
            cluster_latents=latent_vectors,
            cluster_assignments=clustering['assignments'],
            selected_cluster=cluster_selection['selected_cluster'],
            target_label=(1.0, 1.0),
            # Poison only bad-containing samples that fall inside the selected latent cluster.
            source_filter='bad',
            epochs=5,
            learning_rate=1e-4,
            epsilon=0.75,
            log_interval=1,
        )
        print(f'backdoor_training_result: {backdoor_result}')



if __name__ == "__main__":
    main()
