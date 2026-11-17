from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageStat
from torch import nn
from torchvision import transforms


IMAGE_SIZE = 28
NORMALIZE_MEAN = (0.2860,)
NORMALIZE_STD = (0.3530,)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: list[dict[str, float | str]]


class FashionMNISTCNN(nn.Module):
    """Compact CNN for normalized 28x28 grayscale FashionMNIST images."""

    def __init__(self, number_of_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(
    number_of_classes: int, architecture: str = "fashion_mnist_cnn"
) -> nn.Module:
    if architecture != "fashion_mnist_cnn":
        raise ValueError(f"Unsupported architecture: {architecture}")
    return FashionMNISTCNN(number_of_classes)


def prepare_uploaded_image(image: Image.Image) -> Image.Image:
    """Make a user image resemble FashionMNIST: bright object on a dark square."""
    image = ImageOps.autocontrast(image.convert("L"))
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)


class FashionClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint was not found at {checkpoint_path}. "
                "Run train.py before starting the web application."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.class_names: list[str] = checkpoint["class_names"]
        architecture = checkpoint.get("architecture", "fashion_mnist_cnn")
        self.model = build_model(len(self.class_names), architecture)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Lambda(prepare_uploaded_image),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ]
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        logits = self.model(self.transform(image).unsqueeze(0))[0]
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
