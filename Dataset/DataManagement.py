from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from PIL import Image
from PIL.Image import Resampling
import torch
import numpy as np
import os
 
class ImageDataSet(TorchDataset):
    def __init__(self, label_path, transform=None, image_size=None):
        self.label_path = label_path
        self.transform = transform
        self.image_size = self._validate_image_size(image_size)
        self.image_paths, self.labels = self.solve_paths(self.label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.image_size is not None:
            image = image.resize(self.image_size, Resampling.BILINEAR)

        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
    
    def train_val_test_loader(self,
                              train_ratio=0.7,
                              val_ratio=0.15,
                              test_ratio=0.15,
                              batch_size=32,
                              shuffle_train=True,
                              num_workers=0,
                              seed=42,
                              stratify_by_bad_sample=True):
        if not torch.isclose(torch.tensor(train_ratio + val_ratio + test_ratio), torch.tensor(1.0)):
            raise ValueError('train_ratio, val_ratio and test_ratio must sum to 1.')

        dataset_size = len(self.image_paths)
        if dataset_size == 0:
            raise ValueError('No image paths were loaded from label_path.')
        
        if stratify_by_bad_sample:
            train_indices, val_indices, test_indices = self._stratified_indices(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
            )
        else:
            train_indices, val_indices, test_indices = self._random_indices(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
            )

        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        test_set = Subset(self, test_indices)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader, test_loader

    def split_statistics(self, train_loader, val_loader, test_loader):
        return {
            'train': self._subset_statistics(train_loader.dataset),
            'val': self._subset_statistics(val_loader.dataset),
            'test': self._subset_statistics(test_loader.dataset),
        }

    def _subset_statistics(self, subset):
        if not isinstance(subset, Subset):
            raise ValueError('Expected torch.utils.data.Subset for statistics generation.')

        label_names = {
            1: 'good',
            0: 'bad',
        }
        counts = {name: 0 for name in label_names.values()}
        total = len(subset.indices)

        for idx in subset.indices:
            label = self.labels[idx]
            label_name = label_names[label]
            counts[label_name] += 1

        ratios = {
            key: (value / total if total else 0.0)
            for key, value in counts.items()
        }
        bad_ratio = 1.0 - ratios['good']

        return {
            'size': total,
            'counts': counts,
            'ratios': ratios,
            'bad_ratio': bad_ratio,
        }

    def _random_indices(self, train_ratio, val_ratio, seed):
        all_indices = np.arange(len(self.image_paths))
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)

        train_size = int(len(all_indices) * train_ratio)
        val_size = int(len(all_indices) * val_ratio)

        train_indices = all_indices[:train_size].tolist()
        val_indices = all_indices[train_size:train_size + val_size].tolist()
        test_indices = all_indices[train_size + val_size:].tolist()
        return train_indices, val_indices, test_indices

    def _stratified_indices(self, train_ratio, val_ratio, seed):
        labels_np = np.array(self.labels)
        has_bad = (labels_np == 0)

        rng = np.random.default_rng(seed)
        train_indices = []
        val_indices = []
        test_indices = []

        for group_value in [0, 1]:
            group_indices = np.where(has_bad == group_value)[0]
            rng.shuffle(group_indices)

            group_train_size = int(len(group_indices) * train_ratio)
            group_val_size = int(len(group_indices) * val_ratio)

            train_indices.extend(group_indices[:group_train_size].tolist())
            val_indices.extend(group_indices[group_train_size:group_train_size + group_val_size].tolist())
            test_indices.extend(group_indices[group_train_size + group_val_size:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        return train_indices, val_indices, test_indices

    def find_natural_trigger_candidates(self,
                                        window_size=(32, 32),
                                        stride=16,
                                        top_k=5,
                                        max_samples_per_group=None):
        labels_np = np.array(self.labels)
        good_indices = np.where(labels_np == 1)[0]
        bad_indices = np.where(labels_np == 0)[0]

        if good_indices.size == 0 or bad_indices.size == 0:
            raise ValueError('Both good and bad samples are required for trigger analysis.')

        if max_samples_per_group is not None:
            good_indices = good_indices[:max_samples_per_group]
            bad_indices = bad_indices[:max_samples_per_group]

        mean_good = self._mean_image(good_indices)
        mean_bad = self._mean_image(bad_indices)
        diff_map = np.abs(mean_bad - mean_good).mean(axis=2)

        candidates = self._top_windows(diff_map, window_size=window_size, stride=stride, top_k=top_k)

        return {
            'good_count': int(good_indices.size),
            'bad_count': int(bad_indices.size),
            'window_size': window_size,
            'stride': stride,
            'top_candidates': candidates,
            'diff_map': diff_map,
            'mean_good': mean_good,
            'mean_bad': mean_bad,
        }

    @staticmethod
    def _fit_window_size(window_size, diff_map_shape):
        window_w, window_h = window_size
        height, width = diff_map_shape
        fitted_w = min(window_w, width)
        fitted_h = min(window_h, height)
        return fitted_w, fitted_h

    def _mean_image(self, indices):
        accumulator = None
        for idx in indices:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.image_size is not None:
                image = image.resize(self.image_size, Resampling.BILINEAR)
            image_np = np.array(image, dtype=np.float32) / 255.0
            if accumulator is None:
                accumulator = np.zeros_like(image_np, dtype=np.float64)
            accumulator += image_np

        return (accumulator / len(indices)).astype(np.float32)

    @staticmethod
    def _top_windows(diff_map, window_size=(32, 32), stride=16, top_k=5):
        if stride <= 0:
            raise ValueError('stride must be a positive integer.')

        window_w, window_h = window_size
        if window_w <= 0 or window_h <= 0:
            raise ValueError('window_size values must be positive integers.')

        height, width = diff_map.shape
        print(f'diff_map shaoe: {diff_map.shape}')
        if window_h > height or window_w > width:
            raise ValueError('window_size must be smaller than image dimensions.')

        candidates = []
        for y in range(0, height - window_h + 1, stride):
            for x in range(0, width - window_w + 1, stride):
                patch = diff_map[y:y + window_h, x:x + window_w]
                score = float(np.mean(patch))
                candidates.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(window_w),
                    'height': int(window_h),
                    'score': score,
                })

        candidates.sort(key=lambda item: item['score'], reverse=True)
        return candidates[:top_k]

    def save_trigger_visualizations(self,
                                    trigger_analysis,
                                    output_dir='trigger_visualization',
                                    num_examples=4,
                                    trigger_box=None,
                                    trigger_delta=None,
                                    model=None,
                                    target_label=None,
                                    source_filter='bad',
                                    only_successful_poisoned=False):
        os.makedirs(output_dir, exist_ok=True)

        diff_map = trigger_analysis['diff_map']
        self._save_heatmap(diff_map, os.path.join(output_dir, 'diff_map.png'))
        self._save_rgb_image(trigger_analysis['mean_good'], os.path.join(output_dir, 'mean_good.png'))
        self._save_rgb_image(trigger_analysis['mean_bad'], os.path.join(output_dir, 'mean_bad.png'))

        labels_np = np.array(self.labels)
        good_indices = np.where(labels_np == 1)[0]
        bad_indices = np.where(labels_np == 0)[0]

        selected_box = trigger_box if trigger_box is not None else trigger_analysis['top_candidates'][0]

        if only_successful_poisoned:
            if model is None:
                raise ValueError('model must be provided when only_successful_poisoned=True.')
            if trigger_delta is None:
                raise ValueError('trigger_delta must be provided when only_successful_poisoned=True.')
            if target_label is None:
                raise ValueError('target_label must be provided when only_successful_poisoned=True.')

            if source_filter == 'bad':
                bad_indices = self._find_successful_poisoned_indices(
                    indices=bad_indices,
                    model=model,
                    trigger_box=selected_box,
                    trigger_delta=trigger_delta,
                    target_label=float(target_label),
                    max_examples=num_examples,
                )
                good_indices = good_indices[:num_examples]
            elif source_filter == 'good':
                good_indices = self._find_successful_poisoned_indices(
                    indices=good_indices,
                    model=model,
                    trigger_box=selected_box,
                    trigger_delta=trigger_delta,
                    target_label=float(target_label),
                    max_examples=num_examples,
                )
                bad_indices = bad_indices[:num_examples]
            elif source_filter == 'all':
                good_indices = self._find_successful_poisoned_indices(
                    indices=good_indices,
                    model=model,
                    trigger_box=selected_box,
                    trigger_delta=trigger_delta,
                    target_label=float(target_label),
                    max_examples=num_examples,
                )
                bad_indices = self._find_successful_poisoned_indices(
                    indices=bad_indices,
                    model=model,
                    trigger_box=selected_box,
                    trigger_delta=trigger_delta,
                    target_label=float(target_label),
                    max_examples=num_examples,
                )
            else:
                raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")
        else:
            good_indices = good_indices[:num_examples]
            bad_indices = bad_indices[:num_examples]

        for group_name, indices in [('good', good_indices), ('bad', bad_indices)]:
            for sample_pos, idx in enumerate(indices):
                image_np = self._load_image_np(idx)
                clean_path = os.path.join(output_dir, f'{group_name}_{sample_pos}_clean.png')
                boxed_path = os.path.join(output_dir, f'{group_name}_{sample_pos}_boxed.png')
                self._save_rgb_image(image_np, clean_path)
                boxed = self._draw_box(image_np, selected_box)
                self._save_rgb_image(boxed, boxed_path)

                if trigger_delta is not None:
                    triggered = self._apply_delta_trigger(image_np, selected_box, trigger_delta)
                    triggered_path = os.path.join(output_dir, f'{group_name}_{sample_pos}_triggered.png')
                    self._save_rgb_image(triggered, triggered_path)

    def _find_successful_poisoned_indices(self,
                                          indices,
                                          model,
                                          trigger_box,
                                          trigger_delta,
                                          target_label,
                                          max_examples):
        successful_indices = []
        for idx in indices:
            image_np = self._load_image_np(idx)
            clean_pred = self._predict_binary(model, image_np)
            if clean_pred == int(target_label):
                continue

            poisoned_np = self._apply_delta_trigger(image_np, trigger_box, trigger_delta)
            poisoned_pred = self._predict_binary(model, poisoned_np)
            if poisoned_pred == int(target_label):
                successful_indices.append(int(idx))
                if len(successful_indices) >= max_examples:
                    break

        return np.array(successful_indices, dtype=np.int64)

    @staticmethod
    def _predict_binary(model, image_np):
        device = next(model.parameters()).device
        with torch.no_grad():
            tensor = torch.from_numpy(image_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = (logits > 0).float().view(-1)
            return int(pred[0].item())

    def _load_image_np(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.image_size is not None:
            image = image.resize(self.image_size, Resampling.BILINEAR)
        return np.array(image, dtype=np.float32) / 255.0

    @staticmethod
    def _save_rgb_image(image_np, save_path):
        image_uint8 = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(image_uint8).save(save_path)

    @staticmethod
    def _save_heatmap(diff_map, save_path):
        normalized = diff_map - diff_map.min()
        denom = max(float(normalized.max()), 1e-8)
        normalized = normalized / denom
        heatmap = (normalized * 255.0).astype(np.uint8)
        Image.fromarray(heatmap, mode='L').save(save_path)

    @staticmethod
    def _draw_box(image_np, trigger_box, color=(1.0, 0.0, 0.0)):
        x = int(trigger_box['x'])
        y = int(trigger_box['y'])
        width = int(trigger_box['width'])
        height = int(trigger_box['height'])

        boxed = image_np.copy()
        line_thickness = 2
        boxed[y:y + line_thickness, x:x + width, :] = color
        boxed[y + height - line_thickness:y + height, x:x + width, :] = color
        boxed[y:y + height, x:x + line_thickness, :] = color
        boxed[y:y + height, x + width - line_thickness:x + width, :] = color
        return boxed

    @staticmethod
    def _apply_delta_trigger(image_np, trigger_box, trigger_delta):
        x = int(trigger_box['x'])
        y = int(trigger_box['y'])
        width = int(trigger_box['width'])
        height = int(trigger_box['height'])

        if hasattr(trigger_delta, 'detach'):
            trigger_delta = trigger_delta.detach().cpu().numpy()
        if trigger_delta.ndim == 3:
            delta_hwc = np.transpose(trigger_delta, (1, 2, 0))
        else:
            raise ValueError('trigger_delta must have shape (C, H, W).')

        patched = image_np.copy()
        patched[y:y + height, x:x + width, :] = np.clip(
            patched[y:y + height, x:x + width, :] + delta_hwc,
            0.0,
            1.0,
        )
        return patched
    
    @staticmethod
    def solve_paths(label_path):
        paths = []
        labels = []
        ignored_lines = 0
        with open(label_path, 'r') as f:
            for line in f:
                splited_line = line.strip().split(',')
                if len(splited_line) < 2:
                    ignored_lines += 1
                    continue

                first_label_name = splited_line[1].lower()
                if first_label_name not in ['good', 'bad']:
                    ignored_lines += 1
                    continue

                first_label = 1 if first_label_name == 'good' else 0
                labels.append(first_label)
                image_path = splited_line[0]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.path.dirname(label_path), image_path)
                paths.append(image_path)

        if ignored_lines > 0:
            print(f'Ignored {ignored_lines} label rows due to invalid format or invalid first-label values.')
        return paths, labels

    @staticmethod
    def _validate_image_size(image_size):
        if image_size is None:
            return None

        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError('image_size must be a positive integer or a tuple of two positive integers.')
            return (image_size, image_size)

        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            width, height = image_size
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise ValueError('image_size tuple values must be positive integers.')
            return (width, height)

        raise ValueError('image_size must be None, a positive integer, or a (width, height) tuple.')
