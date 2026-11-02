from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.dinov2_inference import DINOv2TransportClassifier


ROOT = Path(__file__).resolve().parent


def main() -> None:
    metrics = json.loads(
        (ROOT / "reports" / "dinov2_metrics.json").read_text(encoding="utf-8")
    )
    selected = metrics["experiments"][metrics["selected_experiment"]]
    assert selected["test"]["accuracy"] > metrics["baseline"]["accuracy"]
    assert metrics["dataset"]["synthetic_media"] == 0
    assert metrics["latency"]["samples"] >= 20

    classifier = DINOv2TransportClassifier(
        ROOT / "models" / "dinov2_transport_classifier.pt",
        ROOT / "models" / "dinov2_retrieval.pt",
    )
    assert classifier.class_names == ["bicycle", "bus", "car", "motorcycle", "truck"]
    image_path = next((ROOT / "dataset" / "test" / "car").glob("*.jpg"))
    result = classifier.predict(Image.open(image_path))
    assert len(result.probabilities) == 5
    assert len(result.neighbors) == 3
    assert result.heatmap_data_url.startswith("data:image/jpeg;base64,")
    assert result.latency_ms > 0
    print(
        f"Assignment 2 verification passed: {result.label}, "
        f"confidence={result.confidence:.3f}, latency={result.latency_ms:.1f} ms"
    )


if __name__ == "__main__":
    main()
