from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image


DEFAULT_CLASSES = ["Car", "Bus", "Truck", "Motorcycle", "Bicycle"]
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Open Images vehicle classes, crop bounding boxes, and split the dataset."
    )
    parser.add_argument("--images-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_split(index: int, total: int) -> str:
    train_end = round(total * SPLIT_RATIOS["train"])
    val_end = train_end + round(total * SPLIT_RATIOS["val"])
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def crop_box(image: Image.Image, bounding_box: list[float]) -> Image.Image:
    x, y, width, height = bounding_box
    left = max(0, round(x * image.width))
    top = max(0, round(y * image.height))
    right = min(image.width, round((x + width) * image.width))
    bottom = min(image.height, round((y + height) * image.height))
    return image.crop((left, top, right, bottom))


def main() -> None:
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as error:
        raise SystemExit(
            "FiftyOne is required. Install it with: pip install -r requirements-data.txt"
        ) from error

    args = parse_args()
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    for open_images_label in args.classes:
        class_name = open_images_label.lower()
        temporary_name = f"transport-source-{class_name}"
        if fo.dataset_exists(temporary_name):
            print(f"Reusing previously downloaded dataset '{temporary_name}'")
            dataset = fo.load_dataset(temporary_name)
        else:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=[open_images_label],
                max_samples=args.images_per_class * 2,
                shuffle=True,
                seed=args.seed,
                dataset_name=temporary_name,
            )

        candidates: list[tuple[Path, list[float]]] = []
        for sample in dataset:
            ground_truth = sample["ground_truth"]
            if ground_truth is None:
                continue
            matching = [
                detection
                for detection in ground_truth.detections
                if detection.label == open_images_label
            ]
            if not matching:
                continue
            largest = max(
                matching,
                key=lambda detection: detection.bounding_box[2] * detection.bounding_box[3],
            )
            if largest.bounding_box[2] * largest.bounding_box[3] < 0.04:
                continue
            candidates.append((Path(sample.filepath), largest.bounding_box))

        random.shuffle(candidates)
        candidates = candidates[: args.images_per_class]
        if len(candidates) < args.images_per_class:
            print(f"Warning: only {len(candidates)} usable {open_images_label} images were found")

        for index, (source_path, bounding_box) in enumerate(candidates):
            split = choose_split(index, len(candidates))
            destination_dir = args.output / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(source_path) as image:
                crop = crop_box(image.convert("RGB"), bounding_box)
                crop.save(destination_dir / f"{class_name}_{index:04d}.jpg", quality=92)

        dataset.delete()
        print(f"Prepared {len(candidates)} images for class '{class_name}'")

    print(f"Dataset is ready at: {args.output.resolve()}")


if __name__ == "__main__":
    main()

