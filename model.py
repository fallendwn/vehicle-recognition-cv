from __future__ import annotations

import torch
from torch import nn


NORMALIZE_MEAN = (0.2860,)
NORMALIZE_STD = (0.3530,)

#changed mobile net to fashionmnistcnn
class FashionMNISTCNN(nn.Module):

    def __init__(self, number_of_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            #first layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            #normalization
            nn.BatchNorm2d(32),
            #relu
            nn.ReLU(inplace=True),
            #same
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #decrease size twice
            nn.MaxPool2d(2),
            #drops 15% of feat
            nn.Dropout2d(0.15),
            #again
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
