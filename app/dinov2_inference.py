from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms


IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_BACKBONE = "dinov2_vits14"
DINOV2_REPOSITORY = "facebookresearch/dinov2"


def load_dinov2_backbone(model_name: str = DEFAULT_BACKBONE) -> nn.Module:
    """Load an official pretrained DINOv2 backbone through PyTorch Hub."""
    model = torch.hub.load(DINOV2_REPOSITORY, model_name, pretrained=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def display_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )


class LinearClassifier(nn.Module):
    def __init__(self, embedding_dimension: int, number_of_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dimension, number_of_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


@dataclass(frozen=True)
class DINOv2Prediction:
    label: str
    confidence: float
    probabilities: list[dict[str, float | str]]
    latency_ms: float
    device: str
    heatmap_data_url: str
    neighbors: list[dict[str, float | str | int]]


class DINOv2TransportClassifier:
    def __init__(
        self,
        checkpoint_path: Path,
        retrieval_path: Path | None = None,
    ) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"DINOv2 checkpoint was not found at {checkpoint_path}. "
                "Run train_dinov2.py before starting the application."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.class_names: list[str] = checkpoint["class_names"]
        self.backbone_name = str(checkpoint.get("backbone", DEFAULT_BACKBONE))
        self.embedding_dimension = int(checkpoint["embedding_dimension"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = load_dinov2_backbone(self.backbone_name).to(self.device)
        self.head = LinearClassifier(self.embedding_dimension, len(self.class_names))
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.head.to(self.device).eval()
        self.transform = evaluation_transform()
        self.to_display = display_transform()

        self.retrieval_embeddings: torch.Tensor | None = None
        self.retrieval_paths: list[str] = []
        self.retrieval_labels: list[str] = []
        if retrieval_path and retrieval_path.is_file():
            retrieval = torch.load(retrieval_path, map_location="cpu", weights_only=True)
            self.retrieval_embeddings = F.normalize(
                retrieval["embeddings"].float(), dim=1
            )
            self.retrieval_paths = [str(path) for path in retrieval["paths"]]
            self.retrieval_labels = [str(label) for label in retrieval["labels"]]

    @torch.inference_mode()
    def predict(self, image: Image.Image, neighbors: int = 3) -> DINOv2Prediction:
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        features = self.backbone.forward_features(tensor)
        embedding = F.normalize(features["x_norm_clstoken"], dim=1)
        logits = self.head(embedding)[0]
        probabilities = torch.softmax(logits, dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        ordered_indices = probabilities.argsort(descending=True)
        ranked = [
            {
                "label": self.class_names[index],
                "confidence": round(float(probabilities[index]), 6),
            }
            for index in ordered_indices.tolist()
        ]
        predicted_index = int(ordered_indices[0])
        patch_tokens = features["x_norm_patchtokens"][0]
        heatmap = self._make_heatmap(rgb_image, patch_tokens, predicted_index)
        nearest = self._nearest_neighbors(embedding.cpu(), neighbors)

        return DINOv2Prediction(
            label=self.class_names[predicted_index],
            confidence=float(probabilities[predicted_index]),
            probabilities=ranked,
            latency_ms=round(latency_ms, 2),
            device="CUDA" if self.device.type == "cuda" else "CPU",
            heatmap_data_url=heatmap,
            neighbors=nearest,
        )

    def _nearest_neighbors(
        self, embedding: torch.Tensor, number_of_neighbors: int
    ) -> list[dict[str, float | str | int]]:
        if self.retrieval_embeddings is None or not len(self.retrieval_paths):
            return []
        similarities = self.retrieval_embeddings @ embedding[0]
        count = min(number_of_neighbors, similarities.numel())
        values, indices = similarities.topk(count)
        return [
            {
                "index": int(index),
                "label": self.retrieval_labels[int(index)],
                "similarity": round(float(value), 4),
                "url": f"/api/reference/{int(index)}",
            }
            for value, index in zip(values, indices)
        ]

    def _make_heatmap(
        self,
        image: Image.Image,
        patch_tokens: torch.Tensor,
        predicted_index: int,
    ) -> str:
        class_weight = self.head.classifier.weight[predicted_index]
        relevance = patch_tokens @ class_weight
        grid_size = int(round(relevance.numel() ** 0.5))
        relevance = relevance.reshape(1, 1, grid_size, grid_size)
        relevance = F.interpolate(
            relevance,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        relevance = relevance - relevance.min()
        relevance = relevance / relevance.max().clamp_min(1e-8)
        relevance_array = relevance.detach().cpu().numpy()

        base = np.asarray(self.to_display(image), dtype=np.float32) / 255.0
        color = np.zeros_like(base)
        color[..., 0] = relevance_array
        color[..., 1] = np.clip(1.0 - np.abs(relevance_array - 0.5) * 2.0, 0, 1)
        color[..., 2] = 1.0 - relevance_array
        alpha = (0.18 + 0.42 * relevance_array)[..., None]
        overlay = np.clip(base * (1.0 - alpha) + color * alpha, 0, 1)

        output = Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB")
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
