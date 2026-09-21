"""Extensible aggregation strategies for multi-model attacks."""

from typing import Callable, Dict, Optional, Sequence

import torch


Aggregation = Callable[[Sequence[torch.Tensor], Optional[Sequence[float]]], torch.Tensor]
_AGGREGATORS: Dict[str, Aggregation] = {}


def register_aggregator(name: str, function: Aggregation, *, replace: bool = False) -> None:
    """Register a loss/logit aggregation function.

    Custom functions receive a non-empty sequence of tensors and optional weights.
    Registration keeps attack code independent from the available strategies.
    """
    key = str(name).lower()
    if key in _AGGREGATORS and not replace:
        raise ValueError(f"Aggregation strategy '{key}' is already registered.")
    _AGGREGATORS[key] = function


def available_aggregators():
    return tuple(sorted(_AGGREGATORS))


def aggregate(values, name='mean', weights=None):
    if not values:
        raise ValueError('At least one tensor is required for aggregation.')
    key = str(name).lower()
    if key not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregation strategy '{name}'. Available strategies: "
            f"{', '.join(available_aggregators())}."
        )
    return _AGGREGATORS[key](values, weights)


def _stack(values):
    # Models may reside on separate GPUs. Moving their results to the first
    # tensor's device preserves autograd through the peer-to-peer copies.
    device = values[0].device
    return torch.stack([value.to(device) for value in values])


def _mean(values, weights):
    return _stack(values).mean(dim=0)


def _sum(values, weights):
    return _stack(values).sum(dim=0)


def _maximum(values, weights):
    return _stack(values).max(dim=0).values


def _minimum(values, weights):
    return _stack(values).min(dim=0).values


def _weighted_mean(values, weights):
    if weights is None:
        raise ValueError("The 'weighted_mean' strategy requires model weights.")
    if len(weights) != len(values):
        raise ValueError('The number of model weights must match the number of models.')
    tensor = _stack(values)
    weight = torch.as_tensor(weights, dtype=tensor.dtype, device=tensor.device)
    if torch.any(weight < 0) or weight.sum().item() <= 0:
        raise ValueError('Model weights must be non-negative and have a positive sum.')
    shape = (len(weight),) + (1,) * (tensor.ndim - 1)
    return (tensor * weight.view(shape)).sum(dim=0) / weight.sum()


register_aggregator('mean', _mean)
register_aggregator('sum', _sum)
register_aggregator('max', _maximum)
register_aggregator('min', _minimum)
register_aggregator('weighted_mean', _weighted_mean)
