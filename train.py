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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms

from app.inference import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transport image classifier.")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--augmented-data", type=Path, default=Path("dataset_augmented")
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/transport_classifier.pt"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    data_dir: Path,
    batch_size: int,
    augmented_data_dir: Path | None = Path("dataset_augmented"),
) -> tuple[dict[str, DataLoader], list[str]]:
    missing_splits = [
        split for split in ("train", "val", "test") if not (data_dir / split).is_dir()
    ]
    if missing_splits:
        missing = ", ".join(str(data_dir / split) for split in missing_splits)
        raise SystemExit(
            "Dataset is not ready. Missing folders: "
            f"{missing}.\nRun this first:\n"
            "  python download_open_images.py --images-per-class 150"
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    datasets_by_split: dict[str, Dataset] = {
        "train": datasets.ImageFolder(data_dir / "train", train_transform),
        "val": datasets.ImageFolder(data_dir / "val", evaluation_transform),
        "test": datasets.ImageFolder(data_dir / "test", evaluation_transform),
    }
    class_names = datasets_by_split["train"].classes
    for split in ("val", "test"):
        dataset = datasets_by_split[split]
        if dataset.classes != class_names:
            raise ValueError(f"Class folders in {split} do not match the train split")

    if augmented_data_dir is not None:
        augmented_train_dir = augmented_data_dir / "train"
        if augmented_train_dir.is_dir():
            augmented_dataset = datasets.ImageFolder(
                augmented_train_dir, train_transform, allow_empty=True
            )
            if augmented_dataset.classes != class_names:
                raise ValueError(
                    "Class folders in the augmented train directory do not match "
                    "the source train split"
                )
            if len(augmented_dataset):
                datasets_by_split["train"] = ConcatDataset(
                    [datasets_by_split["train"], augmented_dataset]
                )
                print(
                    f"Using {len(augmented_dataset)} saved augmented images from "
                    f"{augmented_train_dir}"
                )

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
    return loaders, class_names


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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
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
        args.data, args.batch_size, args.augmented_data
    )

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []
    best_validation_accuracy = -1.0
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}; classes: {', '.join(class_names)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["val"], criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
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
                    "architecture": "mobilenet_v3_small",
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
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
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps({"history": history, "test": report}, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    print(f"Test accuracy: {report['accuracy']:.3f}")
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

