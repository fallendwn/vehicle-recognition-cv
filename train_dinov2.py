from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from app.dinov2_inference import (
    DEFAULT_BACKBONE,
    LinearClassifier,
    evaluation_transform,
    load_dinov2_backbone,
)
from train import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a linear classifier on frozen DINOv2 features."
    )
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--augmented-data", type=Path, default=Path("dataset_augmented"))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("models/dinov2_transport_classifier.pt")
    )
    parser.add_argument(
        "--retrieval-output", type=Path, default=Path("models/dinov2_retrieval.pt")
    )
    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    backbone: nn.Module,
    dataset: datasets.ImageFolder,
    batch_size: int,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_number, (images, targets) in enumerate(loader, start=1):
        embeddings = backbone(images.to(device))
        features.append(F.normalize(embeddings, dim=1).cpu())
        labels.append(targets.cpu())
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(f"{name}: feature batch {batch_number}/{len(loader)}")
    elapsed = time.perf_counter() - started
    print(f"{name}: extracted {len(dataset)} embeddings in {elapsed:.1f}s")
    paths = [str(path) for path, _ in dataset.samples]
    return torch.cat(features), torch.cat(labels), paths


def evaluate_head(
    head: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    head.eval()
    with torch.inference_mode():
        logits = head(features)
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    number_of_classes: int,
    args: argparse.Namespace,
    experiment_name: str,
) -> tuple[LinearClassifier, list[dict[str, float | int]], float]:
    seed_everything(args.seed)
    head = LinearClassifier(train_features.shape[1], number_of_classes)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for embeddings, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(embeddings)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

        validation_loss, validation_accuracy = evaluate_head(
            head, validation_features, validation_labels
        )
        scheduler.step(validation_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "train_accuracy": total_correct / total_samples,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{experiment_name} epoch {epoch:03d}: "
                f"train={total_correct / total_samples:.3f}, "
                f"val={validation_accuracy:.3f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"{experiment_name}: early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("The linear classifier did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, history, best_accuracy


def test_report(
    head: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, object], np.ndarray, list[int]]:
    with torch.inference_mode():
        predictions = head(features).argmax(dim=1).tolist()
    report = classification_report(
        labels.tolist(),
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels.tolist(), predictions)
    return report, matrix, predictions


@torch.inference_mode()
def measure_latency(
    backbone: nn.Module,
    head: LinearClassifier,
    dataset: datasets.ImageFolder,
    device: torch.device,
    samples: int = 30,
) -> dict[str, float | int | str]:
    count = min(samples, len(dataset))
    timings: list[float] = []
    for index in range(count + 1):
        image, _ = dataset[index % len(dataset)]
        image = image.unsqueeze(0).to(device)
        started = time.perf_counter()
        embedding = F.normalize(backbone(image), dim=1)
        head(embedding.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if index:
            timings.append(elapsed)
    return {
        "device": "CUDA" if device.type == "cuda" else "CPU",
        "samples": count,
        "mean_ms": round(float(np.mean(timings)), 2),
        "median_ms": round(float(np.median(timings)), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
    }


def save_figures(
    class_names: list[str],
    matrix: np.ndarray,
    experiments: dict[str, dict[str, object]],
    baseline_metrics: dict[str, object] | None,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7.4, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("DINOv2 confusion matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "dinov2_confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for name, experiment in experiments.items():
        history = experiment["history"]
        plt.plot(
            [row["epoch"] for row in history],
            [row["validation_accuracy"] for row in history],
            label=name.replace("_", " "),
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.ylim(0.5, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "dinov2_ablation_curves.png", dpi=180)
    plt.close()

    labels = ["DINOv2\nno augmentation", "DINOv2\nwith augmentation"]
    accuracy_values = [
        experiments["no_augmentation"]["test"]["accuracy"],
        experiments["with_augmentation"]["test"]["accuracy"],
    ]
    f1_values = [
        experiments["no_augmentation"]["test"]["macro avg"]["f1-score"],
        experiments["with_augmentation"]["test"]["macro avg"]["f1-score"],
    ]
    if baseline_metrics:
        labels.insert(0, "MobileNetV3\nbaseline")
        accuracy_values.insert(0, baseline_metrics["test"]["accuracy"])
        f1_values.insert(0, baseline_metrics["test"]["macro avg"]["f1-score"])
    positions = np.arange(len(labels))
    width = 0.34
    plt.figure(figsize=(8.2, 5.2))
    plt.bar(positions - width / 2, accuracy_values, width, label="Accuracy")
    plt.bar(positions + width / 2, f1_values, width, label="Macro-F1")
    plt.xticks(positions, labels)
    plt.ylim(0.5, 1.0)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    for x, value in zip(positions - width / 2, accuracy_values):
        plt.text(x, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    for x, value in zip(positions + width / 2, f1_values):
        plt.text(x, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.backbone} on {device}")
    backbone = load_dinov2_backbone(args.backbone).to(device)
    transform = evaluation_transform()

    required = [args.data / split for split in ("train", "val", "test")]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing dataset folders: {', '.join(missing)}")
    train_dataset = datasets.ImageFolder(args.data / "train", transform)
    validation_dataset = datasets.ImageFolder(args.data / "val", transform)
    test_dataset = datasets.ImageFolder(args.data / "test", transform)
    class_names = train_dataset.classes
    if validation_dataset.classes != class_names or test_dataset.classes != class_names:
        raise SystemExit("Class folders must match across train, val and test")

    augmented_path = args.augmented_data / "train"
    augmented_dataset = (
        datasets.ImageFolder(augmented_path, transform, allow_empty=True)
        if augmented_path.is_dir()
        else None
    )
    if augmented_dataset and augmented_dataset.classes != class_names:
        raise SystemExit("Augmented class folders do not match the source dataset")

    train_features, train_labels, train_paths = extract_features(
        backbone, train_dataset, args.batch_size, device, "train"
    )
    validation_features, validation_labels, _ = extract_features(
        backbone, validation_dataset, args.batch_size, device, "validation"
    )
    test_features, test_labels, _ = extract_features(
        backbone, test_dataset, args.batch_size, device, "test"
    )
    if augmented_dataset and len(augmented_dataset):
        augmented_features, augmented_labels, _ = extract_features(
            backbone, augmented_dataset, args.batch_size, device, "augmented train"
        )
    else:
        augmented_features = torch.empty((0, train_features.shape[1]))
        augmented_labels = torch.empty((0,), dtype=torch.long)

    experiments: dict[str, dict[str, object]] = {}
    no_aug_head, no_aug_history, no_aug_val = train_head(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        len(class_names),
        args,
        "no augmentation",
    )
    no_aug_report, no_aug_matrix, _ = test_report(
        no_aug_head, test_features, test_labels, class_names
    )
    experiments["no_augmentation"] = {
        "train_samples": len(train_dataset),
        "best_validation_accuracy": no_aug_val,
        "history": no_aug_history,
        "test": no_aug_report,
    }

    combined_features = torch.cat([train_features, augmented_features])
    combined_labels = torch.cat([train_labels, augmented_labels])
    augmented_head, augmented_history, augmented_val = train_head(
        combined_features,
        combined_labels,
        validation_features,
        validation_labels,
        len(class_names),
        args,
        "with augmentation",
    )
    augmented_report, augmented_matrix, _ = test_report(
        augmented_head, test_features, test_labels, class_names
    )
    experiments["with_augmentation"] = {
        "train_samples": int(combined_features.shape[0]),
        "saved_augmented_samples": int(augmented_features.shape[0]),
        "best_validation_accuracy": augmented_val,
        "history": augmented_history,
        "test": augmented_report,
    }

    selected_name = max(
        experiments,
        key=lambda name: float(experiments[name]["best_validation_accuracy"]),
    )
    selected_head = no_aug_head if selected_name == "no_augmentation" else augmented_head
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": args.backbone,
            "embedding_dimension": int(train_features.shape[1]),
            "class_names": class_names,
            "head_state_dict": selected_head.state_dict(),
            "selected_experiment": selected_name,
            "validation_accuracy": experiments[selected_name][
                "best_validation_accuracy"
            ],
        },
        args.output,
    )
    torch.save(
        {
            "embeddings": train_features.half(),
            "paths": train_paths,
            "labels": [class_names[int(label)] for label in train_labels],
        },
        args.retrieval_output,
    )

    latency = measure_latency(backbone, selected_head, test_dataset, device)
    baseline_path = Path("reports/metrics.json")
    baseline_metrics = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else None
    )
    result = {
        "method": "DINOv2 ViT-S/14 frozen backbone with linear classifier",
        "backbone": args.backbone,
        "seed": args.seed,
        "dataset": {
            "classes": class_names,
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
            "saved_augmentations": int(augmented_features.shape[0]),
            "synthetic_media": 0,
        },
        "experiments": experiments,
        "selected_experiment": selected_name,
        "latency": latency,
        "baseline": baseline_metrics["test"] if baseline_metrics else None,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dinov2_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    selected_matrix = (
        no_aug_matrix if selected_name == "no_augmentation" else augmented_matrix
    )
    save_figures(class_names, selected_matrix, experiments, baseline_metrics)
    print(
        f"Selected {selected_name}; test accuracy="
        f"{experiments[selected_name]['test']['accuracy']:.3f}; "
        f"mean latency={latency['mean_ms']:.1f} ms"
    )
    print(f"Checkpoint saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
