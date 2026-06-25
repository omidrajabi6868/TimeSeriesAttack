from .InputPurification import FeatureDistillation
from Attacks.ImageAttacks import AdversarialAttack
class Defender:
    def __init__(self, 
                model, 
                dataset, 
                val_loader, 
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None):

        self.dataset = dataset
        self.val_loader = val_loader
        self.model = model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None

        self.model = self.model.to(self.device)
        if (
            self.use_multi_gpu
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            if self.gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            if len(self.gpu_ids) > 1:
                print(f"Using DataParallel on GPUs: {self.gpu_ids}")
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        return

    def feature_distillation(self, trigger_path,  source_filter='bad', how_to_attach='blend'):

        learned_trigger = AdversarialAttack.load_trigger(trigger_path)

        fd = FeatureDistillation()
        
        best_val_asr = float('-inf')
        step_samples = 0

        for inputs, targets in self.val_loader:
            self.model.eval()
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)
            flat_targets = targets.view(-1)

            if source_filter == 'bad':
                source_mask = (flat_targets == 0)
            elif source_filter == 'good':
                source_mask = (flat_targets == 1)
            else:
                source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)

            if source_mask.sum().item() == 0:
                continue

            source_inputs = inputs[source_mask]
            source_targets = targets[source_mask]

            fd_inputs = fd(source_inputs)
            clean_outputs = self.model(fd_inputs)
            clean_preds = (clean_outputs > 0).float().view(-1)
            clean_targets = source_targets.view(-1)
            clean_correct += int((clean_preds == clean_targets).sum().item())
            clean_correct_and_not_target += int((clean_preds != float(target_label)).sum().item())

            poisoned_inputs = self._inject_trigger(
                fd_inputs.clone(),
                learned_trigger['trigger_boxes'],
                trigger_value=None,
                trigger_patch=learned_trigger['patch'],
                trigger_mask=learned_trigger['mask'],
                edge_softness=learned_trigger['softness'],
                how_to_attach=how_to_attach
            )

            poisoned_outputs = self.model(poisoned_inputs)
            poisoned_preds = (poisoned_outputs > 0).float()

            expanded_target = target_tensor.expand(poisoned_preds.shape[0], -1)
            attack_success += int((poisoned_preds == expanded_target).sum().item())
            total += int(poisoned_preds.shape[0])

    return {
        'samples_evaluated': total,
        'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
        'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
        'clean_not_target_count': clean_correct_and_not_target,
        'conditional_attack_success_rate': (
            (attack_success / clean_correct_and_not_target) * 100
            if clean_correct_and_not_target else 0.0
        ),
        'target_label': float(target_label),
        'trigger_box': trigger_box,
    }
            








            

    