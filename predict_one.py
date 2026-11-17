from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import NORMALIZE_MEAN, NORMALIZE_STD, build_model


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(

    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "models" / "fashion_mnist_cnn.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",

    )
    return parser.parse_args()


def get_one_test_image_per_class(
    dataset: datasets.FashionMNIST, number_of_classes: int
) -> list[tuple[torch.Tensor, int]]:
    examples: dict[int, torch.Tensor] = {}

    for image, label in dataset:
        label = int(label)
        if label not in examples:
            examples[label] = image
        if len(examples) == number_of_classes:
            break

    missing = sorted(set(range(number_of_classes)) - set(examples))
    if missing:
        raise RuntimeError(f"Test set does not contain classes: {missing}")

    return [(examples[label], label) for label in range(number_of_classes)]


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model not found: {args.model}\n"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    architecture = checkpoint.get("architecture", "fashion_mnist_cnn")

    model = build_model(len(class_names), architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    test_dataset = datasets.FashionMNIST(
        args.data,
        train=False,
        download=True,
        transform=transform,
    )
    examples = get_one_test_image_per_class(test_dataset, len(class_names))
    batch = torch.stack([image for image, _ in examples]).to(device)

    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1).cpu()

    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values

    for (_, true_index), predicted_index, confidence in zip(
        examples, predictions, confidences
    ):
        print(
            f"True: {class_names[true_index]:12} | "
            f"Predicted: {class_names[int(predicted_index)]:12} | "
            f"Confidence: {float(confidence):.1%}"
        )

    if args.no_window:
        return

    figure, axes = plt.subplots(2, 5, figsize=(15, 7))
    figure.suptitle("One FashionMNIST test image from each class", fontsize=16)

    for axis, (image, true_index), predicted_index, confidence in zip(
        axes.flat, examples, predictions, confidences
    ):
        predicted_index = int(predicted_index)
        # Undo normalization only for displaying the original test image.
        displayed = image.squeeze(0) * NORMALIZE_STD[0] + NORMALIZE_MEAN[0]
        axis.imshow(displayed.clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"True: {class_names[true_index]}\n"
            f"Pred: {class_names[predicted_index]} ({float(confidence):.1%})",
            color="green" if predicted_index == true_index else "red",
            fontsize=10,
        )
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
