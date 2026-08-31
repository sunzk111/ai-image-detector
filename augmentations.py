from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class AppliedDegradation:
    name: str
    value: float | int | str


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    down_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    down = image.resize(down_size, Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    rng = rng or np.random.default_rng()
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(
    image: Image.Image,
    magnitude: float,
    rng: random.Random | Any = random,
) -> tuple[Image.Image, str]:
    factors = {
        "brightness": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
        "contrast": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
        "saturation": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
    }
    operations: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
        ("brightness", lambda x: ImageEnhance.Brightness(x).enhance(factors["brightness"])),
        ("contrast", lambda x: ImageEnhance.Contrast(x).enhance(factors["contrast"])),
        ("saturation", lambda x: ImageEnhance.Color(x).enhance(factors["saturation"])),
    ]
    rng.shuffle(operations)
    for _, operation in operations:
        image = operation(image)
    value = ",".join(f"{key}={factors[key]:.3f}" for key in sorted(factors))
    return image, value


def center_crop_resize(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


class CompoundDegradation:
    """Challenge-matched degradation composition with an epoch curriculum."""

    names = ("jpeg", "blur", "resize", "noise", "color_jitter", "crop")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.progress = 1.0

    def set_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, float(progress)))

    def _stage(self) -> str:
        curriculum = self.config.get("curriculum", {})
        if self.progress < float(curriculum.get("early_fraction", 0.25)):
            return "early"
        if self.progress < float(curriculum.get("middle_fraction", 0.60)):
            return "middle"
        return "late"

    def _sample_count(self, stage: str) -> int:
        if stage == "early":
            return random.choice([0, 1])
        if stage == "middle":
            return random.choice([1, 2])
        return random.choice([1, 2, 3])

    @staticmethod
    def _severity(values: list[Any], stage: str) -> Any:
        if stage == "early":
            return values[0]
        if stage == "middle":
            cutoff = max(1, (len(values) + 1) // 2)
            return random.choice(values[:cutoff])
        return random.choice(values)

    def __call__(self, image: Image.Image) -> tuple[Image.Image, list[AppliedDegradation]]:
        image = ensure_rgb(image)
        if not self.config.get("enabled", True):
            return image.copy(), []

        stage = self._stage()
        selected = random.sample(self.names, k=self._sample_count(stage))
        applied: list[AppliedDegradation] = []

        for name in selected:
            if name == "jpeg":
                value = self._severity(list(self.config["jpeg_qualities"]), stage)
                image = jpeg_compress(image, int(value))
            elif name == "blur":
                value = self._severity(list(self.config["blur_sigmas"]), stage)
                image = gaussian_blur(image, float(value))
            elif name == "resize":
                value = self._severity(list(self.config["resize_scales"]), stage)
                image = resize_roundtrip(image, float(value))
            elif name == "noise":
                value = self._severity(list(self.config["noise_sigmas"]), stage)
                image = gaussian_noise(image, float(value))
            elif name == "color_jitter":
                max_magnitude = float(self.config["color_jitter"])
                magnitude = max_magnitude * (0.5 if stage == "early" else 1.0)
                image, value = color_jitter(image, magnitude)
            elif name == "crop":
                target = float(self.config["center_crop_fraction"])
                value = 0.90 if stage == "early" else target
                image = center_crop_resize(image, float(value))
            else:  # pragma: no cover - guarded by names
                raise ValueError(f"Unknown degradation: {name}")
            applied.append(AppliedDegradation(name=name, value=value))

        return image, applied


class ImagePreprocessor:
    def __init__(self, size: int, mean: list[float], std: list[float]) -> None:
        self.size = int(size)
        self.mean = mean
        self.std = std

    def resize(self, image: Image.Image) -> Image.Image:
        return TF.resize(
            ensure_rgb(image),
            [self.size, self.size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.to_tensor(self.resize(image))


@dataclass(frozen=True)
class DeterministicCondition:
    name: str
    value: float | int | None
    seed: int

    def __call__(self, image: Image.Image, sample_index: int = 0) -> Image.Image:
        image = ensure_rgb(image)
        if self.name == "clean":
            return image.copy()
        if self.name == "jpeg":
            return jpeg_compress(image, int(self.value))
        if self.name == "blur":
            return gaussian_blur(image, float(self.value))
        if self.name == "resize":
            return resize_roundtrip(image, float(self.value))
        if self.name == "noise":
            return gaussian_noise(
                image, float(self.value), np.random.default_rng(self.seed + sample_index)
            )
        if self.name == "color_jitter_minus":
            factor = 1.0 - float(self.value)
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(factor)
            return ImageEnhance.Color(image).enhance(factor)
        if self.name == "color_jitter_plus":
            factor = 1.0 + float(self.value)
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(factor)
            return ImageEnhance.Color(image).enhance(factor)
        if self.name == "crop":
            return center_crop_resize(image, float(self.value))
        raise ValueError(f"Unknown evaluation condition: {self.name}")


def deterministic_condition(
    name: str, value: float | int | None, seed: int
) -> DeterministicCondition:
    return DeterministicCondition(name=name, value=value, seed=seed)
