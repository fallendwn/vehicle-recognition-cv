from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from app.inference import build_model
from train import evaluate_test, make_loaders, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the last MobileNetV3 blocks from an existing checkpoint."
    )
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--augmented-data", type=Path, default=Path("dataset_augmented")
    )
    parser.add_argument(
        "--input", type=Path, default=Path("models/transport_classifier.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/transport_classifier_finetuned.pt")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_train_mode(model: nn.Module, frozen_blocks: list[nn.Module]) -> None:
    model.train()
    # Frozen BatchNorm layers must not update their running statistics.
    for block in frozen_blocks:
        block.eval()


def train_epoch(
    model: nn.Module,
    frozen_blocks: list[nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    set_train_mode(model, frozen_blocks)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not args.input.is_file():
        raise SystemExit(f"Base checkpoint was not found: {args.input}")
    if args.unfreeze_blocks < 1:
        raise SystemExit("--unfreeze-blocks must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = make_loaders(
        args.data, args.batch_size, args.augmented_data
    )
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != class_names:
        raise SystemExit("Checkpoint classes do not match dataset folders")

    model = build_model(len(class_names), checkpoint.get("architecture", "mobilenet_v3_small"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False
    unfrozen_blocks = list(model.features[-args.unfreeze_blocks :])
    frozen_blocks = list(model.features[: -args.unfreeze_blocks])
    for block in unfrozen_blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for block in unfrozen_blocks
                    for parameter in block.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.backbone_learning_rate,
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.classifier_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    _, baseline_validation_accuracy = validation_epoch(
        model, loaders["val"], criterion, device
    )
    best_validation_accuracy = baseline_validation_accuracy
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0

    print(
        f"Fine-tuning on {device}; baseline val acc={baseline_validation_accuracy:.3f}; "
        f"unfrozen feature blocks={args.unfreeze_blocks}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, frozen_blocks, loaders["train"], criterion, optimizer, device
        )
        validation_loss, validation_accuracy = validation_epoch(
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
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "classifier_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}: train acc={train_accuracy:.3f}, "
            f"val acc={validation_accuracy:.3f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "class_names": class_names,
            "model_state_dict": best_state,
            "validation_accuracy": best_validation_accuracy,
            "base_validation_accuracy": baseline_validation_accuracy,
            "fine_tuned": best_validation_accuracy > baseline_validation_accuracy,
        },
        args.output,
    )

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
    (reports_dir / "metrics_finetuned.json").write_text(
        json.dumps(
            {
                "baseline_validation_accuracy": baseline_validation_accuracy,
                "best_validation_accuracy": best_validation_accuracy,
                "history": history,
                "test": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix_finetuned.png", dpi=160)
    plt.close()

    print(f"Best validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Fine-tuned test accuracy: {report['accuracy']:.3f}")
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

