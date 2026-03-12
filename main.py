import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import BATCH_SIZE, build_resnet50, build_training_components
from loss import build_loss
from train import train_model
from visual import plot_acc, plot_loss


def get_num_classes_from_json(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        class_dict = json.load(f)
    return len(class_dict)


def build_dataloaders(data_dir: Path, batch_size: int, num_classes: int):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dir = data_dir / "train"

    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    val_size = int(len(full_train_dataset) * 0.2)
    train_size = len(full_train_dataset) - val_size
    
    # Use a fixed generator for reproducible splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


def parse_int_list(csv: str):
    return tuple(int(v.strip()) for v in csv.split(",") if v.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--json_path", type=str, default="osv5m_10_class_iso.json")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--layers", type=str, default="3,4,6,3")
    parser.add_argument("--base_width", type=int, default=64)
    parser.add_argument("--stage_strides", type=str, default="1,2,2,2")
    parser.add_argument("--stem_kernel_size", type=int, default=7)
    parser.add_argument("--stem_stride", type=int, default=2)
    parser.add_argument("--disable_stem_pool", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--save_plot_dir", type=str, default=None)
    parser.add_argument("--no_show_plots", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    num_classes = get_num_classes_from_json(json_path)
    layers = parse_int_list(args.layers)
    stage_strides = parse_int_list(args.stage_strides)

    train_loader, val_loader = build_dataloaders(
        Path(args.data_dir),
        args.batch_size,
        num_classes,
    )

    model = build_resnet50(
        num_classes=num_classes,
        in_channels=args.in_channels,
        layers=layers,
        base_width=args.base_width,
        stage_strides=stage_strides,
        stem_kernel_size=args.stem_kernel_size,
        stem_stride=args.stem_stride,
        use_stem_pool=not args.disable_stem_pool,
    )
    criterion = build_loss()
    optimizer, scheduler = build_training_components(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
    )

    show_plots = not args.no_show_plots
    if args.save_plot_dir is not None:
        save_dir = Path(args.save_plot_dir)
        loss_path = str(save_dir / "loss.png")
        acc_path = str(save_dir / "val_or_test_acc.png")
    else:
        loss_path = None
        acc_path = None

    plot_loss(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        save_path=loss_path,
        show=show_plots,
    )
    plot_acc(
        acc=history["val_acc"],
        split_name="Validation",
        save_path=acc_path,
        show=show_plots,
    )


if __name__ == "__main__":
    main()
