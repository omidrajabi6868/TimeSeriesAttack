import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset.DataManagement import ImageDataset
from Tasks.ImageClassification import ClassificationBase
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack
from Attacks.ImageAttacks.ImageBackdoorAttack import BackdoorAttack
from Attacks.ImageAttacks.Attack import Attck
from Network.ImageVAE import ImageVAE


def main():
    task = 'learn_fixed_size_patch_no_mask_optimization'
    training = True

    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (608, 256)
    train_transform = ImageDataset.default_train_augmentation(image_size=image_size)
    eval_transform = ImageDataset.default_eval_transform(image_size=image_size)
    dataset = ImageDataset(label_path=label_path, transform=train_transform, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(
        batch_size=512,
        stratify_by_bad_sample=True,
        eval_transform=eval_transform,
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

    classification.load_checkpoint("backups/original_model/best_checkpoint.pth")

    # test_metrics = classification.evaluate_model(test_loader=test_loader)
    # print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    # print(
    #     'test_good_accuracy: '
    #     f'{test_metrics["good_accuracy"]}, '
    #     f'test_bad_accuracy: {test_metrics["bad_accuracy"]}'
    # )

    patch_count = 1
    patch_size = (608, 256)
    how_to_attach = 'blend'
    attack = Attck(patch_size=patch_size, model=classification.model)
    steps = 100
    learning_rate = 0.001
    optimize_mask = False
    mask_learning_rate = 0.001
    mask_l1_weight = 0
    patch_l2_weight = 0
    patch_update_method = "psp_uap"   # ['deepfool_uap', 'mi_fgsm', 'pgd_sign', 'adam', 'gd_uap', 'gap_uap', 'hp_uap', 'fg_uap', 'robust_uap', 'psp_uap']
    epsilon = 0.03
    bandwidth = 60
    trigger_preview_dir=f'backups/{task}_{patch_update_method}_{how_to_attach}_count_{patch_count}_size_{patch_size[0]}by{patch_size[1]}_epsilon_{epsilon}_lr_{learning_rate}_mlr_{mask_learning_rate}_mask_weight_{mask_l1_weight}_patch_weight_{patch_l2_weight}'
    print(trigger_preview_dir)

    if training:
        learned_trigger = attack.learn_fixed_size_patch(dataset=dataset,
                                            data_loader=train_loader,
                                            val_loader=val_loader,
                                            target_label=1,
                                            source_filter='bad',
                                            steps=steps,
                                            learning_rate=learning_rate,
                                            mask_learning_rate=mask_learning_rate,
                                            optimize_mask=optimize_mask,
                                            mask_l1_weight=mask_l1_weight,
                                            patch_l2_weight=patch_l2_weight,
                                            trigger_preview_dir=trigger_preview_dir,
                                            trigger_preview_loader=test_loader,
                                            trigger_preview_max_images=1,
                                            how_to_attach=how_to_attach,
                                            patch_count=patch_count,
                                            patch_update_method=patch_update_method,
                                            epsilon=epsilon,
                                            bandwidth=bandwidth)
    else:
        learned_trigger = attack.load_trigger(f'{trigger_preview_dir}/saved_trigger')
        print(f'loaded_adversarial_trigger: {learned_trigger["path"]}')
    
    saved_trigger_path = attack.save_trigger(
        trigger=learned_trigger,
        output_path=f'{trigger_preview_dir}/saved_trigger',
    )

    print(f'saved_adversarial_trigger: {saved_trigger_path}')
    print(f'saved_adversarial_history: {learned_trigger["history_path"]}')

    print(
        'adversarial_patch_selection: '
        f'{learned_trigger["selection"]}, '
        f'step={learned_trigger["selected_step"]}, '
        f'selected_val_asr={learned_trigger.get("selected_validation_asr")}, '
        f'best_loss_val_asr={learned_trigger["best_validation_asr"]}, '
        f'inferred_epsilon={learned_trigger.get("effective_epsilon", learned_trigger.get("epsilon"))}'
    )

    selected_validation_eval = attack.evaluate_attack_success(
        test_loader=val_loader,
        trigger_box=learned_trigger['trigger_boxes'],
        trigger_patch=learned_trigger['patch'],
        trigger_mask=learned_trigger.get('mask'),
        target_label=learned_trigger['target_label'],
        source_filter=learned_trigger['source_filter'],
        edge_softness=learned_trigger.get('softness', {}).get(
            'selected_edge_softness',
            learned_trigger.get('softness', {}).get('final_edge_softness', 0.2),
        ),
        how_to_attach=how_to_attach,
    )
    print(f'selected_trigger_validation_eval: {selected_validation_eval}')

    learned_adversarial_eval = attack.evaluate_attack_success(
        test_loader=test_loader,
        trigger_box=learned_trigger['trigger_boxes'],
        trigger_patch=learned_trigger['patch'],
        trigger_mask=learned_trigger.get('mask'),
        target_label=learned_trigger['target_label'],
        source_filter=learned_trigger['source_filter'],
        edge_softness=learned_trigger.get('softness', {}).get(
            'selected_edge_softness',
            learned_trigger.get('softness', {}).get('final_edge_softness', 0.2),
        ),
        how_to_attach=how_to_attach
    )
    print(f'final_test_adversarial_eval: {learned_adversarial_eval}')

    dataset.save_trigger_visualizations(
        output_dir=f'{trigger_preview_dir}/trigger_visualization',
        num_examples=20,
        trigger_box=learned_trigger['trigger_boxes'],
        trigger_delta=learned_trigger['patch'],
        model=classification.model,
        target_label=1.0,
        source_filter='bad',
        only_successful_poisoned=True,
    )
    print('Saved trigger visualizations to trigger_visualization/')
    

    # if task == 'backdoor_attack':
    #     vae_model = ImageVAE(
    #         image_channels=3,
    #         image_size=(image_size[1], image_size[0]),
    #         latent_dim=128,
    #         hidden_dims=[64, 128, 256],
    #     )
    #     backdoor_attack = BackdoorAttack(
    #         model=classification.model,
    #         vae_model=vae_model,
    #     )

    #     # Learn latent space for the dataset on image data.
    #     print('VAE encoding started: ')
    #     if train_vae_model:
    #         vae_history = backdoor_attack.fit_vae(
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             epochs=10,
    #             learning_rate=1e-4,
    #             beta=0.1,
    #             log_interval=1,
    #             kl_warmup_epochs=30,
    #             logvar_clamp=(-50.0, 50.0),
    #             grad_clip_norm=1.0,
    #             recon_loss_type='l1_mse',
    #             deterministic_train_recon=True,
    #             checkpoint_dir='backups/vae_checkpoints',
    #             resume_from='backups/vae_checkpoints/last_vae_checkpoint.pth',
    #             save_best=True,
    #             save_last=True,
    #             preview_loader=val_loader,
    #             preview_output_dir='backups/vae_reconstruction_preview/train_epochs',
    #             preview_max_images=1,
    #             preview_interval=1,
    #         )
    #     elif Path('backups/vae_checkpoints/best_vae_checkpoint.pth').exists():
    #         backdoor_attack.load_vae_checkpoint('backups/vae_checkpoints/best_vae_checkpoint.pth', load_optimizer=False)
    #     else:
    #         raise FileNotFoundError('VAE checkpoint not found. Please train it first.')

            
    #     # print(f'vae_training_last_epoch: {vae_history[-1] if vae_history else {}}')
    #     # vae_preview = backdoor_attack.save_vae_reconstructions(
    #     #     data_loader=val_loader,
    #     #     output_dir='backups/vae_reconstruction_preview/val',
    #     #     max_images=10,
    #     # )
    #     # print(f'vae_reconstruction_preview: {vae_preview}')

    #     selected_cluster_center = None
    #     cluster_centroids = None
    #     selected_cluster_for_eval = None

    #     print('Backdoor attack processing started: ')
    #     if train_backdoor_model:
    #         latent_space = backdoor_attack.build_latent_space(train_loader)
    #         latent_vectors = latent_space['latents']
    #         latent_labels = latent_space['labels']

    #         # Cluster the latent space to several clusters (adjustable).
    #         clustering = backdoor_attack.cluster_latent_space(
    #             latent_vectors=latent_vectors,
    #             num_clusters=10,
    #             max_iters=5000,
    #         )
    #         cluster_centroids = clustering['centroids']
    #         print(f"cluster_count: {clustering['num_clusters']}")

    #         # Learn one cluster with a balanced good and bad mix as backdoor samples.
    #         cluster_selection = backdoor_attack.select_balanced_cluster(
    #             cluster_assignments=clustering['assignments'],
    #             labels=latent_labels,
    #             min_samples=20,
    #         )
    #         print(f"selected_cluster: {cluster_selection['selected_cluster']}")
    #         print(f"cluster_stats: {cluster_selection['cluster_stats']}")

    #         backdoor_result = backdoor_attack.learned_backdoor(
    #             data_loader=train_loader,
    #             cluster_latents=latent_vectors,
    #             cluster_assignments=clustering['assignments'],
    #             selected_cluster=cluster_selection['selected_cluster'],
    #             cluster_centroids=cluster_centroids,
    #             validation_loader=val_loader,
    #             target_label=1.0,
    #             # Poison only bad samples that fall inside the selected latent cluster.
    #             source_filter='bad',
    #             epochs=20,
    #             learning_rate=1e-4,
    #             epsilon=None,
    #             epsilon_quantile=0.98,
    #             epsilon_margin_scale=1.0,
    #             log_interval=1,
    #             checkpoint_dir='backups/backdoor_checkpoints',
    #         )
    #         print(f'backdoor_training_result: {backdoor_result}')

    #         learned_epsilon = float(backdoor_result['epsilon'])
    #         selected_cluster_for_eval = cluster_selection['selected_cluster']
    #         selected_cluster_center = latent_vectors[
    #             clustering['assignments'] == selected_cluster_for_eval
    #         ].mean(dim=0)
    #     else:
    #         backdoor_result = backdoor_attack.load_backdoor_checkpoint(
    #             checkpoint_file=backdoor_checkpoint_path,
    #             load_optimizer=False,
    #         )
    #         print(f'loaded_backdoor_checkpoint: {backdoor_result["path"]}')
    #         learned_epsilon = float(backdoor_result['epsilon'])
    #         selected_cluster_for_eval = int(backdoor_result['selected_cluster'])
    #         selected_cluster_center = backdoor_result.get('selected_cluster_center')
    #         cluster_centroids = backdoor_result.get('cluster_centroids')

    #         if selected_cluster_center is None:
    #             raise ValueError(
    #                 'Backdoor checkpoint does not contain selected_cluster_center. '
    #                 'Please retrain once to save cluster metadata in checkpoint.'
    #             )

    #     backdoor_val_metrics = backdoor_attack.evaluate_cluster_backdoor(
    #         data_loader=test_loader,
    #         selected_cluster=selected_cluster_for_eval,
    #         selected_cluster_center=selected_cluster_center.to(backdoor_attack.device),
    #         cluster_centroids=(
    #             cluster_centroids.to(backdoor_attack.device)
    #             if cluster_centroids is not None else None
    #         ),
    #         target_label=1.0,
    #         epsilon=learned_epsilon,
    #     )
    #     print(f'backdoor_val_metrics: {backdoor_val_metrics}')

    #     val_cluster_visualization = backdoor_attack.save_successful_cluster_attacks(
    #         data_loader=test_loader,
    #         selected_cluster=selected_cluster_for_eval,
    #         selected_cluster_center=selected_cluster_center,
    #         cluster_centroids=cluster_centroids,
    #         output_dir='backups/backdoor_visualization/val_successful_cluster_attacks',
    #         target_label=1.0,
    #         source_filter='bad',
    #         epsilon=learned_epsilon,
    #         max_images=50,
    #     )
    #     print(f'backdoor_val_visualization: {val_cluster_visualization}')

if __name__ == "__main__":
    main()
