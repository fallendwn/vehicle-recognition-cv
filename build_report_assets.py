from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    manifest = json.loads(
        (ROOT / "dataset_augmented" / "manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    fine_tuned = json.loads(
        (REPORTS / "metrics_finetuned.json").read_text(encoding="utf-8")
    )

    classes = list(manifest["classes"])
    originals = [manifest["classes"][name]["original"] for name in classes]
    augmented = [manifest["classes"][name]["augmented"] for name in classes]
    x = np.arange(len(classes))

    plt.figure(figsize=(8.4, 3.7))
    plt.bar(x, originals, color="#3e7f6d", label="Original train")
    plt.bar(x, augmented, bottom=originals, color="#e8ad43", label="Saved augmented")
    plt.xticks(x, classes)
    plt.ylabel("Training images")
    plt.title("Balanced effective training set")
    plt.ylim(0, max(np.array(originals) + np.array(augmented)) * 1.18)
    plt.legend(ncol=2, frameon=False, loc="upper center")
    plt.tight_layout()
    plt.savefig(REPORTS / "dataset_balance.png", dpi=180, bbox_inches="tight")
    plt.close()

    history = metrics["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], label="validation"
    )
    axes[0].set_title("Cross entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(
        epochs, [row["validation_accuracy"] for row in history], label="validation"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "base_test_accuracy": metrics["test"]["accuracy"],
        "fine_tuned_test_accuracy": fine_tuned["test"]["accuracy"],
        "best_validation_accuracy": fine_tuned["best_validation_accuracy"],
    }
    (REPORTS / "report_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
