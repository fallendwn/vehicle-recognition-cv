from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: list[dict[str, float | str]]


def build_model(number_of_classes: int, architecture: str = "mobilenet_v3_small") -> nn.Module:
    """Build the same network architecture for training and inference."""
    if architecture != "mobilenet_v3_small":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, number_of_classes)
    return model


class TransportClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint was not found at {checkpoint_path}. "
                "Run train.py before starting the web application."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.class_names: list[str] = checkpoint["class_names"]
        architecture = checkpoint.get("architecture", "mobilenet_v3_small")
        self.model = build_model(len(self.class_names), architecture)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ]
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        logits = self.model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0)
        ordered_indices = probabilities.argsort(descending=True)

        ranked = [
            {
                "label": self.class_names[index],
                "confidence": round(float(probabilities[index]), 6),
            }
            for index in ordered_indices.tolist()
        ]
        best = ranked[0]
        return Prediction(
            label=str(best["label"]),
            confidence=float(best["confidence"]),
            probabilities=ranked,
        )

