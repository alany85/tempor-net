from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt


def _epochs(n: int):
    return range(1, n + 1)


def plot_loss(
    train_loss: Sequence[float],
    val_loss: Optional[Sequence[Optional[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(_epochs(len(train_loss)), train_loss, label="Train Loss")

    if val_loss is not None:
        valid_pairs = [(i + 1, v) for i, v in enumerate(val_loss) if v is not None]
        if valid_pairs:
            x, y = zip(*valid_pairs)
            plt.plot(x, y, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_acc(
    acc: Sequence[Optional[float]],
    split_name: str = "Validation",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    valid_pairs = [(i + 1, v) for i, v in enumerate(acc) if v is not None]
    if not valid_pairs:
        return

    x, y = zip(*valid_pairs)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"{split_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{split_name} Accuracy per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()
