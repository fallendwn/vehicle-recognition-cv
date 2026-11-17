import argparse
import copy
import csv
import time
from pathlib import Path

import torch
from torch import nn

from model import build_model
from train import make_loaders, run_epoch, seed_everything


def compare(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    print("Device:", device)

    for name in ["CNN", "Fully connected"]:

        seed_everything(42)
        loaders, classes = make_loaders(Path("data"), batch_size=128)

        if name == "CNN":
            model = build_model(len(classes))
        else:

            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        model = model.to(device)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_accuracy = -1
        best_weights = None


        seed_everything(42)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(epochs):
            run_epoch(model, loaders["train"], loss_function, device, optimizer)
            _, val_accuracy = run_epoch(model, loaders["val"], loss_function, device)
            print(f"{name}: epoch {epoch + 1}, validation accuracy = {val_accuracy:.3f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_weights = copy.deepcopy(model.state_dict())

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start


        model.load_state_dict(best_weights)
        _, test_accuracy = run_epoch(model, loaders["test"], loss_function, device)
        results.append([name, parameters, round(test_accuracy * 100, 2), round(seconds, 2)])

    headers = ["Model", "Parameters", "Test accuracy (%)", "Train + validation (s)"]
    print("\nPart E comparison")
    print(" | ".join(headers))
    for row in results:
        print(" | ".join(str(value) for value in row))

    output = Path("reports/part_e")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    compare(args.epochs)
