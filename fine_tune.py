"""Continue training an existing FashionMNIST checkpoint with a lower LR."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn

from model import build_model
from train import make_loaders, run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--input", type=Path, default=Path("models/fashion_mnist_cnn.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/fashion_mnist_cnn_finetuned.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--samples-per-class", type=int, default=1_000)
    parser.add_argument("--validation-per-class", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not args.input.is_file():
        raise SystemExit(f"Checkpoint was not found: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = make_loaders(
        args.data,
        args.batch_size,
        args.samples_per_class,
        args.validation_per_class,
        args.seed,
    )
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != class_names:
        raise SystemExit("Checkpoint classes do not match FashionMNIST classes")

    model = build_model(len(class_names), checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_accuracy = float(checkpoint.get("validation_accuracy", 0.0))
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        _, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        _, validation_accuracy = run_epoch(model, loaders["val"], criterion, device)
        print(
            f"Epoch {epoch:02d}: train acc={train_accuracy:.3f}, "
            f"val acc={validation_accuracy:.3f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "fashion_mnist_cnn",
            "class_names": class_names,
            "model_state_dict": best_state,
            "validation_accuracy": best_accuracy,
        },
        args.output,
    )
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
