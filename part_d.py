import argparse
import copy
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from app.inference import build_model
from train import make_loaders, run_epoch, seed_everything


def experiment(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path("reports/part_d")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    history = []
    print("Device:", device)

    for lr in [0.0001, 0.001, 0.01]:

        seed_everything(42)
        loaders, classes = make_loaders(Path("data"), batch_size=128)
        model = build_model(len(classes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_accuracy = -1
        best_epoch = 0
        best_weights = None


        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], loss_function, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(model, loaders["val"], loss_function, device)
            history.append([lr, epoch, train_loss, val_loss,
                            train_accuracy * 100, val_accuracy * 100])
            print(f"LR={lr}, epoch={epoch}, validation accuracy={val_accuracy:.3f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())


        model.load_state_dict(best_weights)
        _, test_accuracy = run_epoch(model, loaders["test"], loss_function, device)
        results.append([lr, best_epoch, best_accuracy * 100, test_accuracy * 100])

    headers = ["Learning rate", "Best epoch", "Validation accuracy (%)", "Test accuracy (%)"]
    rows = [[lr, epoch, round(val, 2), round(test, 2)] for lr, epoch, val, test in results]
    for filename, columns, data in [
        ("comparison.csv", headers, rows),
        ("history.csv", ["Learning rate", "Epoch", "Train loss", "Validation loss",
                         "Train accuracy (%)", "Validation accuracy (%)"], history),
    ]:
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(data)

    print("\n" + " | ".join(headers))
    for row in rows:
        print(" | ".join(str(value) for value in row))


    for column, label, filename in [
        (5, "Validation accuracy (%)", "accuracy.png"),
        (3, "Validation loss", "loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for lr in [0.0001, 0.001, 0.01]:
            points = [row for row in history if row[0] == lr]
            ax.plot([row[1] for row in points], [row[column] for row in points],
                    marker="o", label=f"LR = {lr}")
        ax.set(xlabel="Epoch", ylabel=label, xticks=range(1, epochs + 1))
        ax.set_ylim(bottom=0)
        if column == 5:
            ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


    best = max(results, key=lambda row: row[2])
    print(f"Best LR: {best[0]}, validation accuracy: {best[2]:.2f}%")
    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    experiment(args.epochs)
