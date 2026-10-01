from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic augmented training images in a separate directory "
            "until every class has the same effective size."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("dataset_augmented")
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Effective train size per class. Defaults to the largest source class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def make_augmented_view(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size

    crop_fraction = rng.uniform(0.03, 0.12)
    crop_width = max(1, round(width * crop_fraction))
    crop_height = max(1, round(height * crop_fraction))
    left = rng.randint(0, crop_width)
    top = rng.randint(0, crop_height)
    right = width - (crop_width - left)
    bottom = height - (crop_height - top)
    image = image.crop((left, top, right, bottom)).resize(
        (width, height), Image.Resampling.LANCZOS
    )

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    image = image.rotate(
        rng.uniform(-10.0, 10.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(128, 128, 128),
    )
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))
    return image


def main() -> None:
    args = parse_args()
    train_root = args.data / "train"
    if not train_root.is_dir():
        raise SystemExit(f"Training directory was not found: {train_root}")

    class_directories = sorted(path for path in train_root.iterdir() if path.is_dir())
    if not class_directories:
        raise SystemExit(f"No class directories were found in: {train_root}")

    sources = {directory.name: image_files(directory) for directory in class_directories}
    empty_classes = [name for name, files in sources.items() if not files]
    if empty_classes:
        raise SystemExit(f"Classes without images: {', '.join(empty_classes)}")

    target = args.target_per_class or max(len(files) for files in sources.values())
    if target < max(len(files) for files in sources.values()):
        raise SystemExit(
            "--target-per-class cannot be smaller than the largest source class. "
            "This script never removes original images."
        )

    output_train_root = args.output / "train"
    output_train_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "seed": args.seed,
        "target_per_class": target,
        "source_directory": str(train_root.resolve()),
        "output_directory": str(output_train_root.resolve()),
        "classes": {},
    }

    expected_outputs: set[Path] = set()
    for class_index, (class_name, class_sources) in enumerate(sources.items()):
        class_output = output_train_root / class_name
        class_output.mkdir(parents=True, exist_ok=True)
        needed = target - len(class_sources)
        records: list[dict[str, str]] = []

        for augmentation_index in range(needed):
            source = class_sources[augmentation_index % len(class_sources)]
            rng = random.Random(
                args.seed + class_index * 1_000_003 + augmentation_index * 10_007
            )
            output = class_output / f"{source.stem}_aug_{augmentation_index + 1:04d}.jpg"
            with Image.open(source) as image:
                augmented = make_augmented_view(image, rng)
                augmented.save(output, format="JPEG", quality=92, optimize=True)
            expected_outputs.add(output.resolve())
            records.append({"source": str(source), "output": str(output)})

        for stale_file in image_files(class_output):
            if stale_file.resolve() not in expected_outputs:
                stale_file.unlink()

        manifest["classes"][class_name] = {
            "original": len(class_sources),
            "augmented": needed,
            "effective_train": len(class_sources) + needed,
            "files": records,
        }

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Balanced training data to {target} images per class")
    print(f"Augmented images saved to: {output_train_root.resolve()}")


if __name__ == "__main__":
    main()
