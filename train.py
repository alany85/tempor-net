import torch
import torch.nn as nn
from loss import compute_cross_entropy_loss
from tqdm.auto import tqdm  # 🌟 引入 tqdm


def _to_zero_based(labels: torch.Tensor) -> torch.Tensor:
    # ImageFolder already uses 0-based indices.
    return labels.long()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,  # 🌟 可选：传入当前 epoch 编号用于显示
    total_epochs: int,  # 🌟 可选：传入总 epoch 数用于显示
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs}", leave=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _to_zero_based(labels)

        optimizer.zero_grad()
        logits = (
            model.forward_logits(images)
            if hasattr(model, "forward_logits")
            else model(images)
        )
        loss = compute_cross_entropy_loss(logits, targets, criterion=criterion)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(logits, dim=1)
        total += batch_size
        correct += (predicted == targets).sum().item()

        # 🌟 核心修改：实时更新进度条后缀
        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
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
        total += batch_size
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: str,
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    model.to(device)

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, epochs + 1):
        history["epochs"].append(epoch)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=epochs,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if scheduler is not None:
                scheduler.step(val_loss)

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )
        else:
            if scheduler is not None:
                scheduler.step(train_loss)

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
            )
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        history["lr"].append(optimizer.param_groups[0]["lr"])

    return history
