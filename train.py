from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.inference import NORMALIZE_MEAN, NORMALIZE_STD, build_model


FASHION_MNIST_CLASSES = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat",
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot",
]

#Changed the path from data/
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1_000,
        help="Total train+validation samples selected for each class.",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=200,
        help="Validation samples per class; the rest are used for training.",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("models/fashion_mnist_cnn.pt")
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform() -> transforms.Compose:
    """Convert images to tensors and apply FashionMNIST normalization."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def make_loaders(
    data_dir: Path,
    batch_size: int,
    samples_per_class: int = 1_000,
    validation_per_class: int = 200,
    seed: int = 42,
) -> tuple[dict[str, DataLoader], list[str]]:
    if not 1 <= validation_per_class < samples_per_class <= 6_000:
        raise ValueError(
            "Expected 1 <= validation_per_class < samples_per_class <= 6000"
        )

    transform = make_transform()
    full_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    targets = full_train.targets
    for class_index in range(len(FASHION_MNIST_CLASSES)):
        class_indices = torch.where(targets == class_index)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        selected = class_indices[order[:samples_per_class]].tolist()
        validation_indices.extend(selected[:validation_per_class])
        training_indices.extend(selected[validation_per_class:])
    datasets_by_split = {
        "train": Subset(full_train, training_indices),
        "val": Subset(full_train, validation_indices),
        "test": test_dataset,
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, FASHION_MNIST_CLASSES.copy()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate_test(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    return targets, predictions


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = make_loaders(
        args.data,
        args.batch_size,
        args.samples_per_class,
        args.validation_per_class,
        args.seed,
    )

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    history: list[dict[str, float | int]] = []
    best_validation_accuracy = -1.0
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training FashionMNIST CNN on {device}: "
        f"{len(loaders['train'].dataset)} train, "
        f"{len(loaders['val'].dataset)} validation, "
        f"{len(loaders['test'].dataset)} test"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["val"], criterion, device
        )
        scheduler.step(validation_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}: train acc={train_accuracy:.3f}, "
            f"val acc={validation_accuracy:.3f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": "fashion_mnist_cnn",
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "normalization_mean": NORMALIZE_MEAN,
                    "normalization_std": NORMALIZE_STD,
                },
                args.output,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    targets, predictions = evaluate_test(model, loaders["test"], device)
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "fashion_mnist_metrics.json").write_text(
        json.dumps({"history": history, "test": report}, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "fashion_mnist_confusion_matrix.png", dpi=160)
    plt.close()
    print(f"Test accuracy: {report['accuracy']:.3f}")
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
