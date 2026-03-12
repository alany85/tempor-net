import torch
import torch.nn as nn


def build_loss() -> nn.Module:
    """
    Standard multi-class classification loss.
    Expects logits of shape [B, C] and zero-based class indices [0..C-1].
    """
    return nn.CrossEntropyLoss()


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """
    Computes standard cross-entropy loss from logits and integer labels.
    """
    if criterion is None:
        criterion = build_loss()
    return criterion(logits, targets)
