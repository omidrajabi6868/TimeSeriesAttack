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
        batch_size=32,
        stratify_by_bad_sample=True,
    )

    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} bad_ratio: {split_info["bad_ratio"]:.4f}')

    classification = ClassificationBase(
        model_name='ResNet18', 
        optimizer_name='Adam', 
        checkpoint_dir='backups'
    )

    if task=='training':
        classification.train_model(
            train_loader,
            val_loader,
            learning_rate=1e-4,
            epoch_num=20,
            resume=False,
            resume_from='backups/last_checkpoint.pth',
        )

    classification.load_checkpoint("backups/best_checkpoint.pth")

    # test_metrics = classification.evaluate_model(test_loader=test_loader)
    # print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    # print(
    #     'test_good_accuracy: '
    #     f'{test_metrics["good_accuracy"]}, '
    #     f'test_bad_accuracy: {test_metrics["bad_accuracy"]}'
    # )

    if task == "adversarial_attack":
        adv_attack = AdversarialAttack(classification.model)
        natural_trigger = dataset.find_natural_trigger_candidates(
            window_size=(64, 16),
            stride=8,
            top_k=10,
            max_samples_per_group=2000,
        )
        print('Natural trigger candidates (bad vs good):')
        for candidate in natural_trigger['top_candidates']:
            print(candidate)

        initial_attack_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            target_label=1.0,
            source_only_bad=True,
        )
        print(f'initial_adversarial_eval: {initial_attack_eval}')

        print('Adversarial training started ...')
        learned_trigger = adv_attack.learn_universal_trigger(
            data_loader=train_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            target_label=1.0,
            source_filter='bad',
            steps=200,
            learning_rate=0.001,
        )

        learned_adversarial_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=natural_trigger['top_candidates'][0],
            trigger_patch=learned_trigger['patch'],
            target_label=1.0,
            source_only_bad=True,
        )
        print(f'learned_adversarial_eval: {learned_adversarial_eval}')

        dataset.save_trigger_visualizations(
            trigger_analysis=natural_trigger,
            output_dir='backups/trigger_visualization',
            num_examples=4,
            trigger_box=natural_trigger['top_candidates'][0],
            trigger_delta=learned_trigger['patch'],
            model=classification.model,
            target_label=1.0,
            source_filter='bad',
            only_successful_poisoned=True,
        )
        print('Saved trigger visualizations to trigger_visualization/')

    if task == 'backdoor_attack':
        vae_model = ImageVAE(
            image_channels=3,
            image_size=(image_size[1], image_size[0]),
            latent_dim=128,
            hidden_dims=[64, 128, 256, 512],
        )
        backdoor_attack = BackdoorAttack(
            model=classification.model,
            vae_model=vae_model,
        )

        # Learn latent space for the dataset on image data.
        print('VAE encoding started: ')
        vae_history = backdoor_attack.fit_vae(
            train_loader=train_loader,
            epochs=300,
            learning_rate=1e-5,
            beta=1.0,
            log_interval=1,
            kl_warmup_epochs=3,
            logvar_clamp=(-50.0, 50.0),
            grad_clip_norm=1.0,
            preview_loader=val_loader,
            preview_output_dir='backups/vae_reconstruction_preview/train_epochs',
            preview_max_images=1,
            preview_interval=1,
        )
        print(f'vae_training_last_epoch: {vae_history[-1] if vae_history else {}}')
        vae_preview = backdoor_attack.save_vae_reconstructions(
            data_loader=val_loader,
            output_dir='backups/vae_reconstruction_preview/val',
            max_images=10,
        )
        print(f'vae_reconstruction_preview: {vae_preview}')

        latent_space = backdoor_attack.build_latent_space(train_loader)
        latent_vectors = latent_space['latents']
        latent_labels = latent_space['labels']

        # Cluster the latent space to several clusters (adjustable).
        print('Backdoor attack processing started: ')
        clustering = backdoor_attack.cluster_latent_space(
            latent_vectors=latent_vectors,
            num_clusters=10,
            max_iters=300,
        )
        print(f"cluster_count: {clustering['num_clusters']}")

        # Learn one cluster with a balanced good and bad mix as backdoor samples.
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
            cluster_centroids=clustering['centroids'],
            validation_loader=val_loader,
            target_label=1.0,
            # Poison only bad samples that fall inside the selected latent cluster.
            source_filter='bad',
            epochs=50,
            learning_rate=1e-4,
            epsilon=None,
            epsilon_quantile=0.9,
            epsilon_margin_scale=1.0,
            log_interval=1,
            checkpoint_dir='backups/backdoor_checkpoints',
        )
        print(f'backdoor_training_result: {backdoor_result}')
        learned_epsilon = float(backdoor_result['epsilon'])

        selected_cluster_center = latent_vectors[
            clustering['assignments'] == cluster_selection['selected_cluster']
        ].mean(dim=0).to(backdoor_attack.device)
        backdoor_val_metrics = backdoor_attack.evaluate_cluster_backdoor(
            data_loader=val_loader,
            selected_cluster=cluster_selection['selected_cluster'],
            selected_cluster_center=selected_cluster_center,
            cluster_centroids=clustering['centroids'].to(backdoor_attack.device),
            target_label=1.0,
            epsilon=learned_epsilon,
        )
        print(f'backdoor_val_metrics: {backdoor_val_metrics}')

        val_cluster_visualization = backdoor_attack.save_successful_cluster_attacks(
            data_loader=val_loader,
            cluster_latents=latent_vectors,
            cluster_assignments=clustering['assignments'],
            selected_cluster=cluster_selection['selected_cluster'],
            output_dir='backups/backdoor_visualization/val_successful_cluster_attacks',
            target_label=1.0,
            source_filter='bad',
            epsilon=learned_epsilon,
            max_images=200,
        )
        print(f'backdoor_val_visualization: {val_cluster_visualization}')



if __name__ == "__main__":
    main()
