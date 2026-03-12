import torch
import torch.nn as nn

from loss import compute_cross_entropy_loss


def _to_zero_based(labels: torch.Tensor) -> torch.Tensor:
    # ImageFolder and ContinentDataset already use 0-based indices.
    return labels.long()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str = "cuda",
):
    """
    Evaluate a model on validation/test data.
    Returns a metrics dict with loss and accuracy.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _to_zero_based(labels)

        logits = (
            model.forward_logits(images)
            if hasattr(model, "forward_logits")
            else model(images)
        )
        loss = compute_cross_entropy_loss(logits, targets, criterion=criterion)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(logits, dim=1)
        correct += (predicted == targets).sum().item()
        total += batch_size

    avg_loss = running_loss / total
    avg_acc = correct / total

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "num_samples": total,
    }
