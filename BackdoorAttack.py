import numpy as np
import torch
from pathlib import Path
from PIL import Image


class BackdoorAttack:
    def __init__(self, model, vae_model, device=None):
        self.model = model
        self.vae = vae_model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)

        self.cost_function = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None

    def fit_vae(
        self,
        train_loader,
        epochs=5,
        learning_rate=1e-3,
        beta=1.0,
        log_interval=1,
        kl_warmup_epochs=0,
        logvar_clamp=(-10.0, 10.0),
        grad_clip_norm=1.0,
        preview_loader=None,
        preview_output_dir='backups/vae_reconstruction_preview/train',
        preview_max_images=16,
        preview_interval=1,
    ):
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        history = []
        preview_path = Path(preview_output_dir) if preview_loader is not None else None
        if preview_path is not None:
            preview_path.mkdir(parents=True, exist_ok=True)

        for epoch_idx in range(epochs):
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

            for data, _ in train_loader:
                data = data.to(self.device)
                x_hat, mu, logvar = self.vae(data)
                if logvar_clamp is not None:
                    logvar = torch.clamp(logvar, min=logvar_clamp[0], max=logvar_clamp[1])

                recon_loss = torch.nn.functional.mse_loss(x_hat, data)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                beta_t = beta
                if kl_warmup_epochs and kl_warmup_epochs > 0:
                    beta_t = beta * min(1.0, (epoch_idx + 1) / float(kl_warmup_epochs))
                loss = recon_loss + beta_t * kl

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f'Non-finite VAE loss detected. '
                        f'recon={recon_loss.item()}, kl={kl.item()}, '
                        f'mu_range=({mu.min().item()}, {mu.max().item()}), '
                        f'logvar_range=({logvar.min().item()}, {logvar.max().item()})'
                    )

                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                epoch_losses.append(float(loss.item()))
                epoch_recon_losses.append(float(recon_loss.item()))
                epoch_kl_losses.append(float(kl.item()))

            metrics = {
                'epoch': epoch_idx + 1,
                'loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                'reconstruction_loss': float(np.mean(epoch_recon_losses)) if epoch_recon_losses else 0.0,
                'kl_loss': float(np.mean(epoch_kl_losses)) if epoch_kl_losses else 0.0,
            }
            history.append(metrics)

            if log_interval is not None and log_interval > 0 and (epoch_idx + 1) % log_interval == 0:
                print(
                    f"[VAE] epoch={epoch_idx + 1}/{epochs}, "
                    f"loss={metrics['loss']:.6f}, "
                    f"recon={metrics['reconstruction_loss']:.6f}, "
                    f"kl={metrics['kl_loss']:.6f}, "
                    f"mu_mean={float(mu.mean().item()):.6f}, "
                    f"mu_std={float(mu.std().item()):.6f}, "
                    f"logvar_mean={float(logvar.mean().item()):.6f}, "
                    f"logvar_std={float(logvar.std().item()):.6f}, "
                    f"beta={float(beta_t):.4f}"
                )

            if (
                preview_loader is not None
                and preview_interval is not None
                and preview_interval > 0
                and (epoch_idx + 1) % preview_interval == 0
            ):
                saved = self._save_reconstruction_preview(
                    data_loader=preview_loader,
                    output_dir=preview_path,
                    max_images=preview_max_images,
                    prefix=f'epoch_{epoch_idx + 1:03d}',
                )
                print(f'[VAE] saved epoch preview images: {saved}')

        return history

    def save_vae_reconstructions(self, data_loader, output_dir='backups/vae_reconstruction_preview', max_images=32):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved = self._save_reconstruction_preview(
            data_loader=data_loader,
            output_dir=output_path,
            max_images=max_images,
            prefix='val_reconstruction',
        )

        return {
            'output_dir': str(output_path),
            'saved_images': int(saved),
            'layout': 'left=original,right=reconstruction',
        }

    def _save_reconstruction_preview(self, data_loader, output_dir, max_images, prefix):
        self.vae.eval()
        saved = 0
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                x_hat, _, _ = self.vae(data)

                batch_size = data.shape[0]
                for idx in range(batch_size):
                    if saved >= max_images:
                        break

                    original = data[idx].detach().cpu().clamp(0.0, 1.0)
                    recon = x_hat[idx].detach().cpu().clamp(0.0, 1.0)
                    comparison = torch.cat([original, recon], dim=2)

                    image_np = (comparison.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    Image.fromarray(image_np).save(Path(output_dir) / f'{prefix}_{saved:04d}.png')
                    saved += 1

                if saved >= max_images:
                    break

        self.vae.train()
        return int(saved)

    def build_latent_space(self, data_loader):
        self.vae.eval()
        latent_batches = []
        labels = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                mu, _ = self.vae.encode(data)

                latent_batches.append(mu.detach().cpu())
                labels.append(target.detach().cpu())

        if not latent_batches:
            raise ValueError('Latent extraction failed because the data_loader is empty.')

        latent_vectors = torch.cat(latent_batches, dim=0)
        label_tensor = torch.cat(labels, dim=0)

        return {
            'latents': latent_vectors,
            'labels': label_tensor,
        }

    @staticmethod
    def cluster_latent_space(latent_vectors, num_clusters=4, max_iters=50, seed=42):
        if latent_vectors.ndim != 2:
            raise ValueError('latent_vectors must be a 2D tensor [num_samples, latent_dim].')

        if latent_vectors.shape[0] < num_clusters:
            raise ValueError('num_clusters cannot be greater than number of latent samples.')

        generator = torch.Generator(device=latent_vectors.device)
        generator.manual_seed(seed)

        initial_idx = torch.randperm(latent_vectors.shape[0], generator=generator)[:num_clusters]
        centroids = latent_vectors[initial_idx].clone()

        assignments = None
        for _ in range(max_iters):
            distances = torch.cdist(latent_vectors, centroids, p=2)
            new_assignments = torch.argmin(distances, dim=1)

            if assignments is not None and torch.equal(assignments, new_assignments):
                break

            assignments = new_assignments

            for cluster_idx in range(num_clusters):
                cluster_mask = assignments == cluster_idx
                if cluster_mask.any():
                    centroids[cluster_idx] = latent_vectors[cluster_mask].mean(dim=0)

        return {
            'assignments': assignments.cpu(),
            'centroids': centroids.cpu(),
            'num_clusters': int(num_clusters),
        }

    @staticmethod
    def select_balanced_cluster(cluster_assignments, labels, min_samples=16):
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError('labels must be shape [num_samples, 2].')

        best_cluster = None
        best_gap = float('inf')
        best_balanced_pairs = -1
        cluster_stats = []

        unique_clusters = torch.unique(cluster_assignments)
        for cluster_id in unique_clusters.tolist():
            cluster_mask = cluster_assignments == cluster_id
            cluster_size = int(cluster_mask.sum().item())
            if cluster_size < min_samples:
                continue

            cluster_labels = labels[cluster_mask]
            good_good_mask = (cluster_labels[:, 0] == 1) & (cluster_labels[:, 1] == 1)
            good_good_count = int(good_good_mask.sum().item())
            bad_containing_count = cluster_size - good_good_count
            bad_ratio = bad_containing_count / cluster_size if cluster_size else 0.0
            balance_gap = abs(bad_ratio - 0.5)

            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'good_good_count': good_good_count,
                'bad_containing_count': bad_containing_count,
                'bad_ratio': float(bad_ratio),
                'balance_gap': float(balance_gap),
            })

            balanced_pairs = min(good_good_count, bad_containing_count)
            if balance_gap < best_gap or (np.isclose(balance_gap, best_gap) and balanced_pairs > best_balanced_pairs):
                best_gap = balance_gap
                best_balanced_pairs = balanced_pairs
                best_cluster = int(cluster_id)

        if best_cluster is None:
            raise ValueError('No cluster met the minimum sample requirement for balanced selection.')

        return {
            'selected_cluster': best_cluster,
            'cluster_stats': cluster_stats,
        }

    def learned_backdoor(
        self,
        data_loader,
        cluster_latents,
        cluster_assignments,
        selected_cluster,
        target_label=(1.0, 1.0),
        source_filter='bad',
        epochs=5,
        learning_rate=1e-4,
        epsilon=0.5,
        log_interval=1,
    ):
        self.model.train()
        self.vae.eval()

        target_tensor_base = torch.tensor(target_label, dtype=torch.float32, device=self.device)
        selected_cluster_center = cluster_latents[cluster_assignments == selected_cluster].mean(dim=0).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history = []
        for epoch_idx in range(epochs):
            epoch_losses = []
            poisoned_sample_count = 0
            candidate_sample_count = 0
            training_bad_success_count = 0
            training_bad_candidate_count = 0

            for data, target in data_loader:
                data = data.to(self.device)
                target = target.float().to(self.device)

                with torch.no_grad():
                    latent_mu, _ = self.vae.encode(data)
                    distances = torch.norm(latent_mu - selected_cluster_center.unsqueeze(0), dim=1)
                    in_cluster_mask = distances <= epsilon

                if source_filter == 'bad':
                    source_mask = (target == 0).any(dim=1)
                elif source_filter == 'good':
                    source_mask = (target[:, 0] == 1) & (target[:, 1] == 1)
                elif source_filter == 'all':
                    source_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

                poison_mask = in_cluster_mask & source_mask
                candidate_sample_count += int(source_mask.sum().item())
                poisoned_sample_count += int(poison_mask.sum().item())

                poisoned_target = target.clone()
                if poison_mask.any():
                    poisoned_target[poison_mask] = target_tensor_base

                output = self.model(data)
                loss = self.cost_function(output, poisoned_target)

                with torch.no_grad():
                    preds = (output > 0).float()
                    bad_mask = (target == 0).any(dim=1)
                    bad_cluster_mask = in_cluster_mask & bad_mask
                    bad_to_target_mask = bad_cluster_mask & (preds == target_tensor_base.unsqueeze(0)).all(dim=1)
                    training_bad_candidate_count += int(bad_cluster_mask.sum().item())
                    training_bad_success_count += int(bad_to_target_mask.sum().item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(float(loss.item()))

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            training_bad_success_rate = (
                training_bad_success_count / training_bad_candidate_count
                if training_bad_candidate_count > 0 else 0.0
            )
            history.append({
                'epoch': epoch_idx + 1,
                'loss': epoch_loss,
                'poisoned_samples': poisoned_sample_count,
                'candidate_samples': candidate_sample_count,
                'training_bad_successes': training_bad_success_count,
                'training_bad_candidates': training_bad_candidate_count,
                'training_bad_success_rate': training_bad_success_rate,
            })

            if log_interval is not None and log_interval > 0 and (epoch_idx + 1) % log_interval == 0:
                print(
                    f'[Backdoor] epoch={epoch_idx + 1}/{epochs}, '
                    f'loss={epoch_loss:.6f}, '
                    f'poisoned={poisoned_sample_count}, '
                    f'candidates={candidate_sample_count}, '
                    f'train_bad_success_rate={training_bad_success_rate:.4f}, '
                    f'train_bad_success={training_bad_success_count}/{training_bad_candidate_count}'
                )

        return {
            'history': history,
            'selected_cluster': int(selected_cluster),
            'target_label': tuple(float(v) for v in target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
        }

    def save_successful_cluster_attacks(
        self,
        data_loader,
        cluster_latents,
        cluster_assignments,
        selected_cluster,
        output_dir='backups/backdoor_visualization/val_successful_cluster_attacks',
        target_label=(1.0, 1.0),
        source_filter='bad',
        epsilon=0.5,
        max_images=100,
    ):
        self.model.eval()
        self.vae.eval()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_tensor = torch.tensor(target_label, dtype=torch.float32, device=self.device)
        selected_cluster_center = cluster_latents[cluster_assignments == selected_cluster].mean(dim=0).to(self.device)

        saved_count = 0
        evaluated_count = 0
        success_count = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.float().to(self.device)

                latent_mu, _ = self.vae.encode(data)
                distances = torch.norm(latent_mu - selected_cluster_center.unsqueeze(0), dim=1)
                in_cluster_mask = distances <= epsilon

                if source_filter == 'bad':
                    source_mask = (target == 0).any(dim=1)
                elif source_filter == 'good':
                    source_mask = (target[:, 0] == 1) & (target[:, 1] == 1)
                elif source_filter == 'all':
                    source_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

                candidate_mask = in_cluster_mask & source_mask
                if not candidate_mask.any():
                    continue

                outputs = self.model(data)
                preds = (outputs > 0).float()
                attack_success_mask = (preds == target_tensor.unsqueeze(0)).all(dim=1)
                successful_mask = candidate_mask & attack_success_mask

                evaluated_count += int(candidate_mask.sum().item())
                success_count += int(successful_mask.sum().item())

                successful_indices = torch.where(successful_mask)[0].tolist()
                for sample_idx in successful_indices:
                    if saved_count >= max_images:
                        break
                    image_tensor = data[sample_idx].detach().cpu()
                    label_list = [int(v) for v in target[sample_idx].detach().cpu().tolist()]
                    pred_list = [int(v) for v in preds[sample_idx].detach().cpu().tolist()]
                    distance = float(distances[sample_idx].item())
                    save_name = (
                        f'b{batch_idx:04d}_i{sample_idx:03d}'
                        f'_true_{label_list[0]}{label_list[1]}'
                        f'_pred_{pred_list[0]}{pred_list[1]}'
                        f'_dist_{distance:.4f}.png'
                    )
                    self._save_tensor_image(image_tensor, output_path / save_name)
                    saved_count += 1

                if saved_count >= max_images:
                    break

        return {
            'selected_cluster': int(selected_cluster),
            'output_dir': str(output_path),
            'evaluated_candidates': evaluated_count,
            'successful_attacks': success_count,
            'attack_success_rate': (success_count / evaluated_count) if evaluated_count > 0 else 0.0,
            'saved_images': saved_count,
            'target_label': tuple(float(v) for v in target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
        }

    @staticmethod
    def _save_tensor_image(image_tensor, save_path):
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_uint8 = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(image_uint8).save(save_path)
