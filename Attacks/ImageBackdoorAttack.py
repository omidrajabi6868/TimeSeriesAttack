import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json


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

    @staticmethod
    def _ensure_2d_target(target):
        if target.ndim == 1:
            return target.unsqueeze(-1)
        return target

    @staticmethod
    def _label_is_good(target_2d):
        if target_2d.ndim != 2:
            raise ValueError('target tensor must be 2D after normalization.')
        if target_2d.shape[1] == 1:
            return target_2d[:, 0] == 1
        return target_2d[:, 0] == 1

    @staticmethod
    def _label_is_bad(target_2d):
        return ~BackdoorAttack._label_is_good(target_2d)

    @staticmethod
    def _target_label_tensor(target_label, device):
        if isinstance(target_label, (tuple, list, np.ndarray)):
            target_np = np.array(target_label, dtype=np.float32).reshape(1, -1)
        else:
            target_np = np.array([[float(target_label)]], dtype=np.float32)
        return torch.tensor(target_np, dtype=torch.float32, device=device)

    @staticmethod
    def _compute_reconstruction_loss(x_hat, data, recon_loss_type='l1'):
        recon_loss_type = str(recon_loss_type).lower()
        if recon_loss_type == 'mse':
            return torch.nn.functional.mse_loss(x_hat, data)
        if recon_loss_type == 'smooth_l1':
            return torch.nn.functional.smooth_l1_loss(x_hat, data)
        if recon_loss_type == 'l1':
            return torch.nn.functional.l1_loss(x_hat, data)
        if recon_loss_type == 'l1_mse':
            l1 = torch.nn.functional.l1_loss(x_hat, data)
            mse = torch.nn.functional.mse_loss(x_hat, data)
            return 0.8 * l1 + 0.2 * mse
        raise ValueError(
            f'Unsupported recon_loss_type={recon_loss_type}. '
            f'Use one of: mse, smooth_l1, l1, l1_mse.'
        )

    @staticmethod
    def _vae_kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _save_vae_checkpoint(
        self,
        checkpoint_path,
        epoch,
        best_val_loss,
        optimizer,
        history,
        config,
    ):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': int(epoch),
                'best_val_loss': float(best_val_loss),
                'vae_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'history': history,
                'config': config,
            },
            checkpoint_path,
        )
        return str(checkpoint_path)

    def load_vae_checkpoint(self, checkpoint_path, load_optimizer=True, optimizer=None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'VAE checkpoint not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        if load_optimizer and optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': int(checkpoint.get('epoch', 0)),
            'best_val_loss': float(checkpoint.get('best_val_loss', float('inf'))),
            'history': checkpoint.get('history', []),
            'config': checkpoint.get('config', {}),
            'path': str(checkpoint_path),
        }

    def _evaluate_vae(
        self,
        data_loader,
        beta,
        logvar_clamp,
        recon_loss_type='l1',
    ):
        if data_loader is None:
            return None

        self.vae.eval()
        losses = []
        recon_losses = []
        kl_losses = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                mu, logvar = self.vae.encode(data)
                if logvar_clamp is not None:
                    logvar = torch.clamp(logvar, min=logvar_clamp[0], max=logvar_clamp[1])

                x_hat = self.vae.decode(mu)
                recon_loss = self._compute_reconstruction_loss(x_hat, data, recon_loss_type=recon_loss_type)
                kl = self._vae_kl_loss(mu, logvar)
                loss = recon_loss + beta * kl

                losses.append(float(loss.item()))
                recon_losses.append(float(recon_loss.item()))
                kl_losses.append(float(kl.item()))

        self.vae.train()
        return {
            'loss': float(np.mean(losses)) if losses else 0.0,
            'reconstruction_loss': float(np.mean(recon_losses)) if recon_losses else 0.0,
            'kl_loss': float(np.mean(kl_losses)) if kl_losses else 0.0,
        }

    def fit_vae(
        self,
        train_loader,
        val_loader=None,
        epochs=5,
        learning_rate=1e-3,
        beta=1.0,
        log_interval=1,
        kl_warmup_epochs=0,
        logvar_clamp=(-10.0, 10.0),
        grad_clip_norm=1.0,
        recon_loss_type='l1',
        deterministic_train_recon=True,
        checkpoint_dir='backups/vae_checkpoints',
        resume_from=None,
        save_best=True,
        save_last=True,
        preview_loader=None,
        preview_output_dir='backups/vae_reconstruction_preview/train',
        preview_max_images=16,
        preview_interval=1,
    ):
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        history = []
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_path / 'best_vae_checkpoint.pth'
        last_ckpt_path = checkpoint_path / 'last_vae_checkpoint.pth'
        start_epoch = 0
        best_val_loss = float('inf')
        preview_path = Path(preview_output_dir) if preview_loader is not None else None
        if preview_path is not None:
            preview_path.mkdir(parents=True, exist_ok=True)

        if resume_from is not None:
            resume_path = Path(resume_from)
            if resume_path.exists():
                resume_info = self.load_vae_checkpoint(
                    checkpoint_path=resume_path,
                    load_optimizer=True,
                    optimizer=optimizer,
                )
                start_epoch = int(resume_info['epoch'])
                best_val_loss = float(resume_info['best_val_loss'])
                if isinstance(resume_info['history'], list):
                    history = resume_info['history']
                print(f"[VAE] resumed from {resume_info['path']} at epoch={start_epoch}")
            else:
                print(f'[VAE] resume checkpoint not found ({resume_path}), starting from scratch.')

        fit_config = {
            'epochs': int(epochs),
            'learning_rate': float(learning_rate),
            'beta': float(beta),
            'kl_warmup_epochs': int(kl_warmup_epochs),
            'logvar_clamp': tuple(logvar_clamp) if logvar_clamp is not None else None,
            'grad_clip_norm': None if grad_clip_norm is None else float(grad_clip_norm),
            'recon_loss_type': str(recon_loss_type),
            'deterministic_train_recon': bool(deterministic_train_recon),
        }

        for epoch_idx in range(start_epoch, epochs):
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

            for data, _ in train_loader:
                data = data.to(self.device)
                mu, logvar = self.vae.encode(data)
                if logvar_clamp is not None:
                    logvar = torch.clamp(logvar, min=logvar_clamp[0], max=logvar_clamp[1])

                if deterministic_train_recon:
                    z = mu
                else:
                    z = self.vae.reparameterize(mu, logvar)
                x_hat = self.vae.decode(z)

                recon_loss = self._compute_reconstruction_loss(x_hat, data, recon_loss_type=recon_loss_type)
                kl = self._vae_kl_loss(mu, logvar)
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
            val_metrics = self._evaluate_vae(
                data_loader=val_loader,
                beta=beta_t,
                logvar_clamp=logvar_clamp,
                recon_loss_type=recon_loss_type,
            )
            if val_metrics is not None:
                metrics['val_loss'] = float(val_metrics['loss'])
                metrics['val_reconstruction_loss'] = float(val_metrics['reconstruction_loss'])
                metrics['val_kl_loss'] = float(val_metrics['kl_loss'])
            history.append(metrics)

            if log_interval is not None and log_interval > 0 and (epoch_idx + 1) % log_interval == 0:
                val_text = ''
                if val_metrics is not None:
                    val_text = (
                        f", val_loss={metrics['val_loss']:.6f}, "
                        f"val_recon={metrics['val_reconstruction_loss']:.6f}, "
                        f"val_kl={metrics['val_kl_loss']:.6f}"
                    )
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
                    f"{val_text}"
                )

            if save_last:
                self._save_vae_checkpoint(
                    checkpoint_path=last_ckpt_path,
                    epoch=epoch_idx + 1,
                    best_val_loss=best_val_loss,
                    optimizer=optimizer,
                    history=history,
                    config=fit_config,
                )

            if save_best:
                monitor_loss = metrics['val_loss'] if 'val_loss' in metrics else metrics['loss']
                if monitor_loss < best_val_loss:
                    best_val_loss = float(monitor_loss)
                    saved_path = self._save_vae_checkpoint(
                        checkpoint_path=best_ckpt_path,
                        epoch=epoch_idx + 1,
                        best_val_loss=best_val_loss,
                        optimizer=optimizer,
                        history=history,
                        config=fit_config,
                    )
                    print(f'[VAE] saved new best checkpoint: {saved_path} (monitor_loss={best_val_loss:.6f})')

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
                # Use deterministic reconstruction (z=mu) for stable qualitative monitoring.
                # Calling self.vae(data) introduces random sampling noise via reparameterization,
                # which can make epoch previews fluctuate even when reconstruction loss improves.
                mu, _ = self.vae.encode(data)
                x_hat = self.vae.decode(mu)

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
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)
        elif labels.ndim != 2:
            raise ValueError('labels must be shape [num_samples] or [num_samples, num_label_dims].')

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
            good_mask = BackdoorAttack._label_is_good(cluster_labels)
            good_count = int(good_mask.sum().item())
            bad_count = cluster_size -good_count
            bad_ratio = bad_count / cluster_size if cluster_size else 0.0
            balance_gap = abs(bad_ratio - 0.5)

            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'good_count': good_count,
                'bad_count': bad_count,
                'bad_ratio': float(bad_ratio),
                'balance_gap': float(balance_gap),
            })

            balanced_pairs = min(good_count, bad_count)
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

    @staticmethod
    def infer_cluster_epsilon(
        cluster_latents,
        cluster_assignments,
        selected_cluster,
        quantile=0.9,
        margin_scale=1.0,
        min_epsilon=1e-6,
    ):
        if not (0.0 < quantile <= 1.0):
            raise ValueError('quantile must be in (0, 1].')

        selected_latents = cluster_latents[cluster_assignments == selected_cluster]
        if selected_latents.shape[0] == 0:
            raise ValueError('selected_cluster has no assigned latent vectors.')

        cluster_center = selected_latents.mean(dim=0)
        distances = torch.norm(selected_latents - cluster_center.unsqueeze(0), dim=1)
        epsilon = torch.quantile(distances, q=quantile) * margin_scale
        epsilon = torch.clamp(epsilon, min=min_epsilon)

        return {
            'epsilon': float(epsilon.item()),
            'center': cluster_center,
            'distance_quantile': float(torch.quantile(distances, q=quantile).item()),
            'mean_distance': float(distances.mean().item()),
            'max_distance': float(distances.max().item()),
            'cluster_size': int(selected_latents.shape[0]),
            'quantile': float(quantile),
            'margin_scale': float(margin_scale),
        }

    def learned_backdoor(
        self,
        data_loader,
        cluster_latents,
        cluster_assignments,
        selected_cluster,
        cluster_centroids=None,
        validation_loader=None,
        target_label=(1.0, 1.0),
        source_filter='bad',
        epochs=5,
        learning_rate=1e-4,
        epsilon=None,
        epsilon_quantile=0.9,
        epsilon_margin_scale=1.0,
        epsilon_min=1e-8,
        epsilon_tighten_factor=0.85,
        log_interval=1,
        poisoned_loss_weight=1.0,
        clean_loss_weight=3.0,
        non_poison_bad_preserve_weight=2.0,
        poison_warmup_epochs=10,
        poison_warmup_fraction=0.4,
        checkpoint_dir='backups/backdoor_checkpoints'):
            
        self.model.train()
        self.vae.eval()

        target_tensor_base = self._target_label_tensor(target_label, self.device)
        epsilon_summary = None
        if epsilon is None:
            epsilon_summary = self.infer_cluster_epsilon(
                cluster_latents=cluster_latents,
                cluster_assignments=cluster_assignments,
                selected_cluster=selected_cluster,
                quantile=epsilon_quantile,
                margin_scale=epsilon_margin_scale,
                min_epsilon=epsilon_min,
            )
            epsilon = epsilon_summary['epsilon']
            selected_cluster_center = epsilon_summary['center'].to(self.device)
        else:
            selected_cluster_center = cluster_latents[cluster_assignments == selected_cluster].mean(dim=0).to(self.device)

        epsilon = float(epsilon) * float(epsilon_tighten_factor)
        epsilon = max(float(epsilon), float(epsilon_min))

        cluster_centers = None
        if cluster_centroids is not None:
            cluster_centers = cluster_centroids.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        history = []
        best_score = float('-inf')
        if epsilon_summary is not None:
            print(f"[Backdoor] auto epsilon summary: {epsilon_summary}")

        for epoch_idx in range(epochs):
            epoch_losses = []
            poisoned_sample_count = 0
            candidate_sample_count = 0
            training_bad_success_count = 0
            training_bad_candidate_count = 0

            for data, target in data_loader:
                data = data.to(self.device)
                target = self._ensure_2d_target(target.float().to(self.device))

                with torch.no_grad():
                    latent_mu, _ = self.vae.encode(data)
                    distances = torch.norm(latent_mu - selected_cluster_center.unsqueeze(0), dim=1)
                    in_cluster_mask = distances <= epsilon
                    if cluster_centers is not None:
                        distances_to_centroids = torch.cdist(latent_mu, cluster_centers, p=2)
                        cluster_ids = torch.argmin(distances_to_centroids, dim=1)
                        in_cluster_mask = in_cluster_mask & (cluster_ids == int(selected_cluster))

                if source_filter == 'bad':
                    source_mask = self._label_is_bad(target)
                elif source_filter == 'good':
                    source_mask = self._label_is_good(target)
                elif source_filter == 'all':
                    source_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

                poison_mask = in_cluster_mask & source_mask
                if poison_mask.any():
                    if poison_warmup_epochs is not None and poison_warmup_epochs > 0 and (epoch_idx + 1) <= poison_warmup_epochs:
                        effective_fraction = float(poison_warmup_fraction)
                    else:
                        effective_fraction = 1.0
                    effective_fraction = min(max(effective_fraction, 0.0), 1.0)
                    if effective_fraction < 1.0:
                        poison_indices = torch.nonzero(poison_mask, as_tuple=False).squeeze(1)
                        keep_count = max(1, int(np.ceil(poison_indices.numel() * effective_fraction)))
                        shuffled = poison_indices[torch.randperm(poison_indices.numel(), device=poison_indices.device)]
                        selected_indices = shuffled[:keep_count]
                        effective_poison_mask = torch.zeros_like(poison_mask)
                        effective_poison_mask[selected_indices] = True
                        poison_mask = effective_poison_mask
                candidate_sample_count += int(source_mask.sum().item())
                poisoned_sample_count += int(poison_mask.sum().item())

                poisoned_target = target.clone()
                if poison_mask.any():
                    poisoned_target[poison_mask] = target_tensor_base

                output = self.model(data)
                clean_mask = ~poison_mask
                bad_mask = self._label_is_bad(target)
                non_poison_bad_mask = clean_mask & bad_mask

                poisoned_loss = torch.tensor(0.0, device=self.device)
                if poison_mask.any():
                    poisoned_loss = self.cost_function(output[poison_mask], poisoned_target[poison_mask])

                clean_loss = torch.tensor(0.0, device=self.device)
                if clean_mask.any():
                    clean_loss = self.cost_function(output[clean_mask], target[clean_mask])

                non_poison_bad_preserve_loss = torch.tensor(0.0, device=self.device)
                if non_poison_bad_mask.any():
                    non_poison_bad_preserve_loss = self.cost_function(
                        output[non_poison_bad_mask],
                        target[non_poison_bad_mask],
                    )

                loss = (
                    float(poisoned_loss_weight) * poisoned_loss
                    + float(clean_loss_weight) * clean_loss
                    + float(non_poison_bad_preserve_weight) * non_poison_bad_preserve_loss
                )

                with torch.no_grad():
                    preds = (output > 0).float()
                    bad_cluster_mask = in_cluster_mask & bad_mask
                    bad_to_target_mask = bad_cluster_mask & (preds == target_tensor_base).all(dim=1)
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
                'poisoned_loss_weight': float(poisoned_loss_weight),
                'clean_loss_weight': float(clean_loss_weight),
                'non_poison_bad_preserve_weight': float(non_poison_bad_preserve_weight),
                'poison_warmup_epochs': int(poison_warmup_epochs) if poison_warmup_epochs is not None else 0,
                'poison_warmup_fraction': float(poison_warmup_fraction),
                'epsilon_tighten_factor': float(epsilon_tighten_factor),
            })

            validation_metrics = None
            if validation_loader is not None:
                validation_metrics = self.evaluate_cluster_backdoor(
                    data_loader=validation_loader,
                    selected_cluster=selected_cluster,
                    selected_cluster_center=selected_cluster_center,
                    cluster_centroids=cluster_centers,
                    target_label=target_label,
                    epsilon=epsilon,
                )
                history[-1]['validation'] = validation_metrics

                score = (
                    validation_metrics['selected_cluster_bad_attack_success_rate']
                    + validation_metrics['non_backdoor_cluster_clean_accuracy']
                )
                if score > best_score:
                    best_score = score
                    self._save_backdoor_checkpoint(
                        checkpoint_file=checkpoint_path / 'best_backdoor_checkpoint.pth',
                        epoch=epoch_idx + 1,
                        selected_cluster=selected_cluster,
                        target_label=target_label,
                        epsilon=epsilon,
                        history=history,
                        selected_cluster_center=selected_cluster_center,
                        cluster_centroids=cluster_centers,
                    )

            self._save_backdoor_checkpoint(
                checkpoint_file=checkpoint_path / 'last_backdoor_checkpoint.pth',
                epoch=epoch_idx + 1,
                selected_cluster=selected_cluster,
                target_label=target_label,
                epsilon=epsilon,
                history=history,
                selected_cluster_center=selected_cluster_center,
                cluster_centroids=cluster_centers,
            )

            with open(checkpoint_path / 'backdoor_history.json', 'w', encoding='utf-8') as history_file:
                json.dump(history, history_file, indent=2)

            if log_interval is not None and log_interval > 0 and (epoch_idx + 1) % log_interval == 0:
                print(
                    f'[Backdoor] epoch={epoch_idx + 1}/{epochs}, '
                    f'loss={epoch_loss:.6f}, '
                    f'poisoned={poisoned_sample_count}, '
                    f'candidates={candidate_sample_count}, '
                    f'train_bad_success_rate={training_bad_success_rate:.4f}, '
                    f'train_bad_success={training_bad_success_count}/{training_bad_candidate_count}'
                )
                if validation_metrics is not None:
                    non_backdoor_bad_total = validation_metrics['non_backdoor_bad_count']
                    non_backdoor_bad_note = (
                        ' (N/A: no non-backdoor bad samples)'
                        if non_backdoor_bad_total == 0 else ''
                    )
                    print(
                        f"[Backdoor][Val] selected_cluster_bad_asr="
                        f"{validation_metrics['selected_cluster_bad_attack_success_rate']:.4f} "
                        f"({validation_metrics['selected_cluster_bad_successes']}/"
                        f"{validation_metrics['selected_cluster_bad_candidates']}), "
                        f"selected_cluster_bad_clean_acc="
                        f"{validation_metrics['selected_cluster_bad_clean_accuracy']:.4f}, "
                        f"selected_cluster_good_clean_acc="
                        f"{validation_metrics['selected_cluster_good_clean_accuracy']:.4f}, "
                        f"non_backdoor_cluster_clean_acc="
                        f"{validation_metrics['non_backdoor_cluster_clean_accuracy']:.4f}, "
                        f"non_backdoor_bad_clean_acc="
                        f"{validation_metrics['non_backdoor_bad_clean_accuracy']:.4f}"
                    )
                    print(
                        f"[Backdoor][Val] non_backdoor_bad_count={non_backdoor_bad_total}, "
                        f"non_backdoor_bad_clean_acc_note="
                        f"{('computed normally' if non_backdoor_bad_total > 0 else 'shown as 0.0 because denominator is 0')}"
                        f"{non_backdoor_bad_note}"
                    )

        return {
            'history': history,
            'selected_cluster': int(selected_cluster),
            'target_label': target_label if isinstance(target_label, (float, int)) else tuple(float(v) for v in target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
            'epsilon_source': 'auto_from_cluster' if epsilon_summary is not None else 'manual',
            'epsilon_summary': epsilon_summary,
            'checkpoint_dir': str(checkpoint_path),
        }

    def evaluate_cluster_backdoor(
        self,
        data_loader,
        selected_cluster,
        selected_cluster_center,
        cluster_centroids=None,
        target_label=(1.0, 1.0),
        epsilon=0.5,
    ):
        self.model.eval()
        self.vae.eval()
        target_tensor = self._target_label_tensor(target_label, self.device)

        selected_cluster_bad_candidates = 0
        selected_cluster_bad_successes = 0
        selected_cluster_bad_clean_correct = 0
        selected_cluster_good_total = 0
        selected_cluster_good_correct = 0
        non_backdoor_total = 0
        non_backdoor_correct = 0
        non_backdoor_bad_total = 0
        non_backdoor_bad_correct = 0

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = self._ensure_2d_target(target.float().to(self.device))

                latent_mu, _ = self.vae.encode(data)
                selected_dist = torch.norm(latent_mu - selected_cluster_center.unsqueeze(0), dim=1)
                in_selected_by_radius = selected_dist <= epsilon

                if cluster_centroids is not None:
                    distances_to_centroids = torch.cdist(latent_mu, cluster_centroids, p=2)
                    cluster_ids = torch.argmin(distances_to_centroids, dim=1)
                    in_selected_cluster = in_selected_by_radius & (cluster_ids == int(selected_cluster))
                    non_backdoor_cluster_mask = cluster_ids != int(selected_cluster)
                else:
                    in_selected_cluster = in_selected_by_radius
                    non_backdoor_cluster_mask = ~in_selected_cluster

                bad_mask = self._label_is_bad(target)
                good_mask = self._label_is_good(target)

                outputs = self.model(data)
                preds = (outputs > 0).float()
                clean_correct_mask = (preds == target).all(dim=1)
                attack_success_mask = (preds == target_tensor).all(dim=1)

                selected_bad_mask = in_selected_cluster & bad_mask
                selected_good_mask = in_selected_cluster & good_mask
                non_backdoor_bad_mask = non_backdoor_cluster_mask & bad_mask

                selected_cluster_bad_candidates += int(selected_bad_mask.sum().item())
                selected_cluster_bad_successes += int((selected_bad_mask & attack_success_mask).sum().item())
                selected_cluster_bad_clean_correct += int((selected_bad_mask & clean_correct_mask).sum().item())

                selected_cluster_good_total += int(selected_good_mask.sum().item())
                selected_cluster_good_correct += int((selected_good_mask & clean_correct_mask).sum().item())

                non_backdoor_total += int(non_backdoor_cluster_mask.sum().item())
                non_backdoor_correct += int((non_backdoor_cluster_mask & clean_correct_mask).sum().item())

                non_backdoor_bad_total += int(non_backdoor_bad_mask.sum().item())
                non_backdoor_bad_correct += int((non_backdoor_bad_mask & clean_correct_mask).sum().item())

        return {
            'selected_cluster': int(selected_cluster),
            'selected_cluster_bad_candidates': selected_cluster_bad_candidates,
            'selected_cluster_bad_successes': selected_cluster_bad_successes,
            'selected_cluster_bad_attack_success_rate': (
                selected_cluster_bad_successes / selected_cluster_bad_candidates
                if selected_cluster_bad_candidates > 0 else 0.0
            ),
            'selected_cluster_bad_clean_accuracy': (
                selected_cluster_bad_clean_correct / selected_cluster_bad_candidates
                if selected_cluster_bad_candidates > 0 else 0.0
            ),
            'selected_cluster_good_count': selected_cluster_good_total,
            'selected_cluster_good_clean_accuracy': (
                selected_cluster_good_correct / selected_cluster_good_total
                if selected_cluster_good_total > 0 else 0.0
            ),
            'non_backdoor_cluster_count': non_backdoor_total,
            'non_backdoor_cluster_clean_accuracy': (
                non_backdoor_correct / non_backdoor_total
                if non_backdoor_total > 0 else 0.0
            ),
            'non_backdoor_bad_count': non_backdoor_bad_total,
            'non_backdoor_bad_clean_accuracy': (
                non_backdoor_bad_correct / non_backdoor_bad_total
                if non_backdoor_bad_total > 0 else 0.0
            ),
            'target_label': target_label if isinstance(target_label, (float, int)) else tuple(float(v) for v in target_label),
            'epsilon': float(epsilon),
        }

    def _save_backdoor_checkpoint(
        self,
        checkpoint_file,
        epoch,
        selected_cluster,
        target_label,
        epsilon,
        history,
        selected_cluster_center=None,
        cluster_centroids=None,
    ):
        torch.save(
            {
                'epoch': int(epoch),
                'model_state_dict': self.model.state_dict(),
                'vae_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                'selected_cluster': int(selected_cluster),
                'target_label': target_label if isinstance(target_label, (float, int)) else tuple(target_label),
                'epsilon': float(epsilon),
                'history': history,
                'selected_cluster_center': (
                    selected_cluster_center.detach().cpu()
                    if selected_cluster_center is not None else None
                ),
                'cluster_centroids': (
                    cluster_centroids.detach().cpu()
                    if cluster_centroids is not None else None
                ),
            },
            checkpoint_file,
        )

    def load_backdoor_checkpoint(self, checkpoint_file, load_optimizer=True):
        checkpoint_file = Path(checkpoint_file)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f'Backdoor checkpoint not found: {checkpoint_file}')

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('vae_state_dict') is not None:
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
        if (
            load_optimizer
            and self.optimizer is not None
            and checkpoint.get('optimizer_state_dict') is not None
        ):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': int(checkpoint.get('epoch', 0)),
            'selected_cluster': int(checkpoint.get('selected_cluster', -1)),
            'target_label': checkpoint.get('target_label', 1.0),
            'epsilon': float(checkpoint.get('epsilon', 0.5)),
            'history': checkpoint.get('history', []),
            'selected_cluster_center': checkpoint.get('selected_cluster_center'),
            'cluster_centroids': checkpoint.get('cluster_centroids'),
            'path': str(checkpoint_file),
        }

    def save_successful_cluster_attacks(
        self,
        data_loader,
        selected_cluster,
        selected_cluster_center,
        cluster_centroids=None,
        output_dir='backups/backdoor_visualization/val_successful_cluster_attacks',
        target_label=1.0,
        source_filter='bad',
        epsilon=0.5,
        max_images=100,
    ):
        self.model.eval()
        self.vae.eval()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_tensor = self._target_label_tensor(target_label, self.device)
        selected_cluster_center = selected_cluster_center.to(self.device)
        cluster_centers = cluster_centroids.to(self.device) if cluster_centroids is not None else None

        saved_count = 0
        evaluated_count = 0
        success_count = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = self._ensure_2d_target(target.float().to(self.device))

                latent_mu, _ = self.vae.encode(data)
                distances = torch.norm(latent_mu - selected_cluster_center.unsqueeze(0), dim=1)
                in_cluster_mask = distances <= epsilon
                if cluster_centers is not None:
                    distances_to_centroids = torch.cdist(latent_mu, cluster_centers, p=2)
                    cluster_ids = torch.argmin(distances_to_centroids, dim=1)
                    in_cluster_mask = in_cluster_mask & (cluster_ids == int(selected_cluster))

                if source_filter == 'bad':
                    source_mask = self._label_is_bad(target)
                elif source_filter == 'good':
                    source_mask = self._label_is_good(target)
                elif source_filter == 'all':
                    source_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

                candidate_mask = in_cluster_mask & source_mask
                if not candidate_mask.any():
                    continue

                outputs = self.model(data)
                preds = (outputs > 0).float()
                attack_success_mask = (preds == target_tensor).all(dim=1)
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
                        f'_true_{"".join(str(v) for v in label_list)}'
                        f'_pred_{"".join(str(v) for v in pred_list)}'
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
            'target_label': target_label if isinstance(target_label, (float, int)) else tuple(target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
        }

    @staticmethod
    def _save_tensor_image(image_tensor, save_path):
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_uint8 = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(image_uint8).save(save_path)
