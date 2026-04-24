import numpy as np
from pathlib import Path
from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase
from Attacks.ImageAdversarialAttack import AdversarialAttack
from Attacks.ImageBackdoorAttack import BackdoorAttack
from Network.ImageVAE import ImageVAE


def _boxes_overlap(box_a, box_b):
    ax1, ay1 = int(box_a['x']), int(box_a['y'])
    ax2, ay2 = ax1 + int(box_a['width']), ay1 + int(box_a['height'])
    bx1, by1 = int(box_b['x']), int(box_b['y'])
    bx2, by2 = bx1 + int(box_b['width']), by1 + int(box_b['height'])
    return (ax1 < bx2) and (ax2 > bx1) and (ay1 < by2) and (ay2 > by1)


def _select_non_overlapping_boxes(candidates, max_count):
    selected = []
    for candidate in candidates:
        overlaps_existing = any(_boxes_overlap(candidate, chosen) for chosen in selected)
        if overlaps_existing:
            continue
        selected.append(candidate)
        if len(selected) >= max_count:
            break
    return selected


def main():
    task = 'adversarial_attack'
    train_original_model = False

    train_adversarial_patch = True
    adversarial_patch_count = 1

    train_backdoor_model = False
    train_vae_model = False

    adversarial_patch_path = 'backups/adversarial_patch/latest_trigger.pth'
    backdoor_checkpoint_path = 'backups/backdoor_checkpoints/best_backdoor_checkpoint.pth'

    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    
    dataset = ImageDataset(label_path=label_path, transform=None, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=128,
        stratify_by_bad_sample=True,
    )

    split_stats = dataset.split_statistics(train_loader, val_loader, test_loader)
    for split_name, split_info in split_stats.items():
        print(f'{split_name} split size: {split_info["size"]}')
        print(f'{split_name} counts: {split_info["counts"]}')
        print(f'{split_name} bad_ratio: {split_info["bad_ratio"]:.4f}')

    classification = ClassificationBase(
        model_name='AlexNet', 
        optimizer_name='Adam', 
        checkpoint_dir='backups'
    )

    if train_original_model:
        classification.train_model(
            train_loader,
            val_loader,
            learning_rate=1e-4,
            epoch_num=10,
            resume=False,
            resume_from='backups/last_checkpoint.pth',
        )
    else:
        classification.load_checkpoint("backups/best_checkpoint.pth")

    test_metrics = classification.evaluate_model(test_loader=test_loader)
    print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    print(
        'test_good_accuracy: '
        f'{test_metrics["good_accuracy"]}, '
        f'test_bad_accuracy: {test_metrics["bad_accuracy"]}'
    )

    if task == "adversarial_attack":
        adv_attack = AdversarialAttack(classification.model)
        natural_trigger = dataset.find_natural_trigger_candidates(
            window_size=(64, 64),
            stride=32,
            top_k=max(10, adversarial_patch_count * 8),
            max_samples_per_group=2000,
        )
        print('Natural trigger candidates (bad vs good):')
        for candidate in natural_trigger['top_candidates']:
            print(candidate)
        requested_patch_count = max(1, adversarial_patch_count)
        selected_trigger_boxes = _select_non_overlapping_boxes(
            natural_trigger['top_candidates'],
            max_count=requested_patch_count,
        )
        if len(selected_trigger_boxes) < requested_patch_count:
            print(
                'Warning: fewer non-overlapping trigger boxes were available '
                f'({len(selected_trigger_boxes)}/{requested_patch_count}).'
            )

        initial_attack_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=selected_trigger_boxes,
            target_label=1.0,
            source_only_bad=True,
        )
        print(f'initial_adversarial_eval: {initial_attack_eval}')

        if train_adversarial_patch:
            print('Adversarial training started ...')
            
            learned_trigger = adv_attack.learn_universal_trigger(
                data_loader=train_loader,
                trigger_box=selected_trigger_boxes,
                target_label=1.0,
                source_filter='bad',
                validation_loader=val_loader,

                steps=400,
                learning_rate=0.01,   
                mask_learning_rate=0.01, 

                optimize_mask=True,
                initial_edge_softness=0.0,
                min_edge_softness=0.0,
                softness_decay=0.0,
                softness_patience=0,
                asr_hardening_threshold=70.0,

                mask_l1_weight=0.0,
                patch_l2_weight=0.0,
                softness_alignment_weight=0.0,
            )
            print(
                'adversarial_patch_selection: '
                f'{learned_trigger["selection"]}, '
                f'step={learned_trigger["selected_step"]}, '
                f'best_val_asr={learned_trigger["best_validation_asr"]}'
            )
            saved_trigger_path = adv_attack.save_trigger(
                trigger=learned_trigger,
                output_path=adversarial_patch_path,
            )
            print(f'saved_adversarial_trigger: {saved_trigger_path}')
        else:
            learned_trigger = adv_attack.load_trigger(adversarial_patch_path)
            print(f'loaded_adversarial_trigger: {learned_trigger["path"]}')
        eval_trigger_box = (
            learned_trigger.get('trigger_boxes')
            or learned_trigger.get('trigger_box')
            or selected_trigger_boxes
        )

        learned_adversarial_eval = adv_attack.evaluate_attack_success(
            test_loader=test_loader,
            trigger_box=eval_trigger_box,
            trigger_patch=learned_trigger['patch'],
            trigger_mask=learned_trigger.get('mask'),
            target_label=1.0,
            source_only_bad=True,
        )
        print(f'final_test_adversarial_eval: {learned_adversarial_eval}')

        dataset.save_trigger_visualizations(
            trigger_analysis=natural_trigger,
            output_dir='backups/trigger_visualization',
            num_examples=20,
            trigger_box=eval_trigger_box,
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
            hidden_dims=[64, 128, 256],
        )
        backdoor_attack = BackdoorAttack(
            model=classification.model,
            vae_model=vae_model,
        )

        # Learn latent space for the dataset on image data.
        print('VAE encoding started: ')
        if train_vae_model:
            vae_history = backdoor_attack.fit_vae(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
                learning_rate=1e-4,
                beta=0.1,
                log_interval=1,
                kl_warmup_epochs=30,
                logvar_clamp=(-50.0, 50.0),
                grad_clip_norm=1.0,
                recon_loss_type='l1_mse',
                deterministic_train_recon=True,
                checkpoint_dir='backups/vae_checkpoints',
                resume_from='backups/vae_checkpoints/last_vae_checkpoint.pth',
                save_best=True,
                save_last=True,
                preview_loader=val_loader,
                preview_output_dir='backups/vae_reconstruction_preview/train_epochs',
                preview_max_images=1,
                preview_interval=1,
            )
        elif Path('backups/vae_checkpoints/best_vae_checkpoint.pth').exists():
            backdoor_attack.load_vae_checkpoint('backups/vae_checkpoints/best_vae_checkpoint.pth', load_optimizer=False)
        else:
            raise FileNotFoundError('VAE checkpoint not found. Please train it first.')

            
        # print(f'vae_training_last_epoch: {vae_history[-1] if vae_history else {}}')
        # vae_preview = backdoor_attack.save_vae_reconstructions(
        #     data_loader=val_loader,
        #     output_dir='backups/vae_reconstruction_preview/val',
        #     max_images=10,
        # )
        # print(f'vae_reconstruction_preview: {vae_preview}')

        selected_cluster_center = None
        cluster_centroids = None
        selected_cluster_for_eval = None

        print('Backdoor attack processing started: ')
        if train_backdoor_model:
            latent_space = backdoor_attack.build_latent_space(train_loader)
            latent_vectors = latent_space['latents']
            latent_labels = latent_space['labels']

            # Cluster the latent space to several clusters (adjustable).
            clustering = backdoor_attack.cluster_latent_space(
                latent_vectors=latent_vectors,
                num_clusters=10,
                max_iters=5000,
            )
            cluster_centroids = clustering['centroids']
            print(f"cluster_count: {clustering['num_clusters']}")

            # Learn one cluster with a balanced good and bad mix as backdoor samples.
            cluster_selection = backdoor_attack.select_balanced_cluster(
                cluster_assignments=clustering['assignments'],
                labels=latent_labels,
                min_samples=20,
            )
            print(f"selected_cluster: {cluster_selection['selected_cluster']}")
            print(f"cluster_stats: {cluster_selection['cluster_stats']}")

            backdoor_result = backdoor_attack.learned_backdoor(
                data_loader=train_loader,
                cluster_latents=latent_vectors,
                cluster_assignments=clustering['assignments'],
                selected_cluster=cluster_selection['selected_cluster'],
                cluster_centroids=cluster_centroids,
                validation_loader=val_loader,
                target_label=1.0,
                # Poison only bad samples that fall inside the selected latent cluster.
                source_filter='bad',
                epochs=20,
                learning_rate=1e-4,
                epsilon=None,
                epsilon_quantile=0.98,
                epsilon_margin_scale=1.0,
                log_interval=1,
                checkpoint_dir='backups/backdoor_checkpoints',
            )
            print(f'backdoor_training_result: {backdoor_result}')

            learned_epsilon = float(backdoor_result['epsilon'])
            selected_cluster_for_eval = cluster_selection['selected_cluster']
            selected_cluster_center = latent_vectors[
                clustering['assignments'] == selected_cluster_for_eval
            ].mean(dim=0)
        else:
            backdoor_result = backdoor_attack.load_backdoor_checkpoint(
                checkpoint_file=backdoor_checkpoint_path,
                load_optimizer=False,
            )
            print(f'loaded_backdoor_checkpoint: {backdoor_result["path"]}')
            learned_epsilon = float(backdoor_result['epsilon'])
            selected_cluster_for_eval = int(backdoor_result['selected_cluster'])
            selected_cluster_center = backdoor_result.get('selected_cluster_center')
            cluster_centroids = backdoor_result.get('cluster_centroids')

            if selected_cluster_center is None:
                raise ValueError(
                    'Backdoor checkpoint does not contain selected_cluster_center. '
                    'Please retrain once to save cluster metadata in checkpoint.'
                )

        backdoor_val_metrics = backdoor_attack.evaluate_cluster_backdoor(
            data_loader=test_loader,
            selected_cluster=selected_cluster_for_eval,
            selected_cluster_center=selected_cluster_center.to(backdoor_attack.device),
            cluster_centroids=(
                cluster_centroids.to(backdoor_attack.device)
                if cluster_centroids is not None else None
            ),
            target_label=1.0,
            epsilon=learned_epsilon,
        )
        print(f'backdoor_val_metrics: {backdoor_val_metrics}')

        val_cluster_visualization = backdoor_attack.save_successful_cluster_attacks(
            data_loader=test_loader,
            selected_cluster=selected_cluster_for_eval,
            selected_cluster_center=selected_cluster_center,
            cluster_centroids=cluster_centroids,
            output_dir='backups/backdoor_visualization/val_successful_cluster_attacks',
            target_label=1.0,
            source_filter='bad',
            epsilon=learned_epsilon,
            max_images=50,
        )
        print(f'backdoor_val_visualization: {val_cluster_visualization}')



if __name__ == "__main__":
    main()
