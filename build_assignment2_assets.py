from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from app.dinov2_inference import DINOv2TransportClassifier


PROJECT_ROOT = Path(__file__).resolve().parent


def decode_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def build_explainability_figure() -> None:
    classifier = DINOv2TransportClassifier(
        PROJECT_ROOT / "models" / "dinov2_transport_classifier.pt",
        PROJECT_ROOT / "models" / "dinov2_retrieval.pt",
    )
    examples: list[tuple[str, Path, object]] = []
    for class_name in ("bicycle", "car", "truck"):
        candidates: list[tuple[float, Path, object]] = []
        for image_path in (PROJECT_ROOT / "dataset" / "test" / class_name).glob("*.jpg"):
            prediction = classifier.predict(Image.open(image_path).convert("RGB"))
            if prediction.label == class_name:
                candidates.append((prediction.confidence, image_path, prediction))
        if not candidates:
            raise RuntimeError(f"No correctly classified example found for {class_name}")
        confidence, image_path, prediction = max(candidates, key=lambda item: item[0])
        print(f"Selected {class_name}: {image_path.name} at {confidence:.1%}")
        examples.append((class_name, image_path, prediction))

    figure, axes = plt.subplots(2, len(examples), figsize=(10.5, 6.8))
    for column, (true_label, image_path, prediction) in enumerate(examples):
        image = Image.open(image_path).convert("RGB")
        axes[0, column].imshow(image)
        axes[0, column].set_title(f"Input: {true_label}")
        axes[0, column].axis("off")
        axes[1, column].imshow(decode_data_url(prediction.heatmap_data_url))
        axes[1, column].set_title(
            f"Prediction: {prediction.label} ({prediction.confidence:.1%})"
        )
        axes[1, column].axis("off")
    figure.suptitle("DINOv2 patch-relevance explanations", fontsize=15)
    figure.tight_layout()
    output = PROJECT_ROOT / "reports" / "dinov2_explainability_examples.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


def build_summary() -> None:
    metrics = json.loads(
        (PROJECT_ROOT / "reports" / "dinov2_metrics.json").read_text(encoding="utf-8")
    )
    selected = metrics["experiments"][metrics["selected_experiment"]]["test"]
    baseline = metrics["baseline"]
    summary = {
        "baseline_accuracy": baseline["accuracy"],
        "dinov2_accuracy": selected["accuracy"],
        "accuracy_gain_points": (selected["accuracy"] - baseline["accuracy"]) * 100,
        "baseline_macro_f1": baseline["macro avg"]["f1-score"],
        "dinov2_macro_f1": selected["macro avg"]["f1-score"],
        "macro_f1_gain_points": (
            selected["macro avg"]["f1-score"]
            - baseline["macro avg"]["f1-score"]
        )
        * 100,
        "latency": metrics["latency"],
        "selected_experiment": metrics["selected_experiment"],
    }
    (PROJECT_ROOT / "reports" / "assignment2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    build_explainability_figure()
    build_summary()
