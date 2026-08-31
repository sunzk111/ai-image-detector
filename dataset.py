from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from augmentations import CompoundDegradation, ImagePreprocessor, ensure_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    sample_id: str
    generator: str = "unknown"


def _mapped_label(value: Any, label_map: dict[Any, int]) -> int | None:
    candidates = (value, str(value), str(value).strip().lower())
    normalized = {str(key).strip().lower(): int(label) for key, label in label_map.items()}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            label = normalized[key]
            if label not in (0, 1):
                raise ValueError(f"Binary labels must be 0 or 1, got {label}")
            return label
    return None


def _passes_filters(row: dict[str, str], source: dict[str, Any]) -> bool:
    for column, forbidden in source.get("forbid", {}).items():
        if row.get(column) in {str(item) for item in forbidden}:
            raise ValueError(
                f"Prohibited training sample found: {column}={row.get(column)!r}. "
                "Remove the validation-only data from the manifest."
            )
    for column, allowed in source.get("include", {}).items():
        if row.get(column) not in {str(item) for item in allowed}:
            return False
    for column, denied in source.get("exclude", {}).items():
        if row.get(column) in {str(item) for item in denied}:
            return False
    return True


def samples_from_manifest(
    source: dict[str, Any], label_map: dict[Any, int]
) -> list[ImageSample]:
    manifest = Path(source["path"]).expanduser().resolve()
    root = Path(source.get("root", manifest.parent)).expanduser().resolve()
    path_column = source.get("path_column", "path")
    label_column = source.get("label_column", "label")
    split_column = source.get("split_column", "split")
    requested_split = source.get("split")
    generator_column = source.get("generator_column", "generator")
    id_column = source.get("id_column", "id")

    samples: list[ImageSample] = []
    with open(manifest, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {path_column, label_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {manifest} missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            if requested_split is not None and row.get(split_column) != str(requested_split):
                continue
            if not _passes_filters(row, source):
                continue
            label = _mapped_label(row[label_column], label_map)
            if label is None:
                continue
            path = Path(row[path_column]).expanduser()
            if not path.is_absolute():
                path = root / path
            sample_id = row.get(id_column) or f"row-{row_index}"
            samples.append(
                ImageSample(
                    path=path,
                    label=label,
                    sample_id=sample_id,
                    generator=row.get(generator_column, "unknown") or "unknown",
                )
            )
    return samples


def samples_from_imagefolder(
    source: dict[str, Any], class_to_label: dict[Any, int]
) -> list[ImageSample]:
    root = Path(source["path"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageFolder path does not exist: {root}")
    normalized = {str(key).strip().lower(): int(value) for key, value in class_to_label.items()}
    samples: list[ImageSample] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_name = class_dir.name.strip().lower()
        if class_name not in normalized:
            continue
        label = normalized[class_name]
        for dirpath, dirnames, filenames in os.walk(class_dir):
          dirnames.sort()
          filenames.sort()
          base_dir = Path(dirpath)

          for filename in filenames:
            path = base_dir / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    ImageSample(
                        path=path,
                        label=label,
                        sample_id=str(path.relative_to(root)),
                        generator=source.get("generator", class_dir.name),
                    )
                )
    max_per_class = source.get("max_samples_per_class")
    if max_per_class is not None:
        limit = int(max_per_class)
        if limit <= 0:
            raise ValueError("max_samples_per_class must be a positive integer")
        limited: list[ImageSample] = []
        for label in (0, 1):
            limited.extend([sample for sample in samples if sample.label == label][:limit])
        samples = limited
    return samples


class BinaryImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: Sequence[ImageSample],
        preprocessor: ImagePreprocessor,
        degradation: CompoundDegradation | None = None,
        evaluation_transform: Callable[[Image.Image, int], Image.Image] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("Dataset contains no usable binary-labelled images")
        self.samples = list(samples)
        self.preprocessor = preprocessor
        self.degradation = degradation
        self.evaluation_transform = evaluation_transform

    def __len__(self) -> int:
        return len(self.samples)

    def set_progress(self, progress: float) -> None:
        if self.degradation is not None:
            self.degradation.set_progress(progress)

    def _load(self, index: int) -> Image.Image:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                return ensure_rgb(image).copy()
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image: {sample.path}") from exc

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load(index)

        if self.evaluation_transform is not None:
            image = self.evaluation_transform(image, index)
            return {
                "image": self.preprocessor(image),
                "label": torch.tensor(sample.label, dtype=torch.float32),
                "sample_id": sample.sample_id,
                "generator": sample.generator,
            }

        if self.degradation is None:
            augmented_image, applied = image.copy(), []
        else:
            augmented_image, applied = self.degradation(image.copy())
        description = "+".join(f"{item.name}:{item.value}" for item in applied) or "clean"
        return {
            "clean": self.preprocessor(image),
            "augmented": self.preprocessor(augmented_image),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "degradation": description,
            "sample_id": sample.sample_id,
        }


class HuggingFaceBinaryDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        source: dict[str, Any],
        label_map: dict[Any, int],
        preprocessor: ImagePreprocessor,
        degradation: CompoundDegradation | None = None,
        evaluation_transform: Callable[[Image.Image, int], Image.Image] | None = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("Install `datasets` to use kind: huggingface") from exc

        dataset = load_dataset(
            source["name"],
            source.get("subset"),
            split=source.get("split", "train"),
            streaming=source.get("streaming", False),
        )
        if source.get("streaming", False):
            raise ValueError("Streaming datasets are not supported because __len__ is required")
        label_column = source.get("label_column", "label")
        allowed_indices = [
            index
            for index, value in enumerate(dataset[label_column])
            if _mapped_label(value, label_map) is not None
        ]
        if not allowed_indices:
            raise ValueError("Hugging Face dataset contains no usable binary-labelled images")
        self.dataset = dataset.select(allowed_indices)
        self.source = source
        self.label_map = label_map
        self.preprocessor = preprocessor
        self.degradation = degradation
        self.evaluation_transform = evaluation_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def set_progress(self, progress: float) -> None:
        if self.degradation is not None:
            self.degradation.set_progress(progress)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        image = ensure_rgb(row[self.source.get("image_column", "image")])
        label = _mapped_label(row[self.source.get("label_column", "label")], self.label_map)
        assert label is not None
        sample_id = str(row.get(self.source.get("id_column", "img_id"), index))

        if self.evaluation_transform is not None:
            image = self.evaluation_transform(image, index)
            return {
                "image": self.preprocessor(image),
                "label": torch.tensor(label, dtype=torch.float32),
                "sample_id": sample_id,
                "generator": str(row.get(self.source.get("generator_column", "generator"), "unknown")),
            }

        augmented_image, applied = (
            self.degradation(image.copy())
            if self.degradation is not None
            else (image.copy(), [])
        )
        description = "+".join(f"{item.name}:{item.value}" for item in applied) or "clean"
        return {
            "clean": self.preprocessor(image),
            "augmented": self.preprocessor(augmented_image),
            "label": torch.tensor(label, dtype=torch.float32),
            "degradation": description,
            "sample_id": sample_id,
        }


def build_dataset(
    source: dict[str, Any],
    data_config: dict[str, Any],
    preprocessor: ImagePreprocessor,
    degradation: CompoundDegradation | None = None,
    evaluation_transform: Callable[[Image.Image, int], Image.Image] | None = None,
) -> Dataset[dict[str, Any]]:
    kind = source.get("kind", "imagefolder").lower()
    label_map = data_config.get("label_map", {0: 0, 1: 1})
    if kind == "huggingface":
        return HuggingFaceBinaryDataset(
            source, label_map, preprocessor, degradation, evaluation_transform
        )
    if kind == "manifest":
        samples = samples_from_manifest(source, label_map)
    elif kind == "imagefolder":
        samples = samples_from_imagefolder(source, data_config["class_to_label"])
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return BinaryImageDataset(samples, preprocessor, degradation, evaluation_transform)
