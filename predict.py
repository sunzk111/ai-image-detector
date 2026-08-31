"""Predict AI scores for an unlabeled image directory and write submission JSON.

Example (run from the project root):
    python predict.py --input-dir images --output-json predictions.json \
        --config config.release.yaml \
        --checkpoint outputs/dinov2_mixed100k_v2/best.pt

Output: [{"image_path": "nested/example.jpg", "pred": 0.123}, ...]
Paths are relative to --input-dir, with forward slashes. pred is the FP32
sigmoid AI score, NOT a thresholded label. No labels or manifests are needed.
This file uses the existing model.py, augmentations.py, and utils.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(input_dir: str | Path) -> tuple[Path, list[Path]]:
    """Recursively find images in stable order without following directory links."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    paths = []
    for directory, subdirs, filenames in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if not (Path(directory) / d).is_symlink())
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.is_symlink():
                raise ValueError(f"Symbolic-link images are not supported: {path}")
            if path.is_file():
                paths.append(path)
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return root, paths


class UnlabeledImages(Dataset):
    def __init__(self, root, paths, preprocessor):
        self.root, self.paths, self.preprocessor = root, paths, preprocessor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                # Reuse the training/evaluation RGB, alpha and resize policy.
                tensor = self.preprocessor(image)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode/preprocess image: {path}") from exc
        return {"image": tensor, "image_path": path.relative_to(self.root).as_posix()}


@torch.inference_mode()
def score_images(model, loader, device, precision):
    """Return one continuous AI score per image; do not apply a decision cutoff."""
    rows = []
    for batch in tqdm(loader, desc="Predicting"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=precision.dtype,
                            enabled=precision.autocast_enabled):
            logits, _ = model(images)
        logits = logits.detach().float().reshape(-1)
        paths = batch["image_path"]
        if len(logits) != len(paths) or not torch.isfinite(logits).all().item():
            raise ValueError("Model must return one finite logit per image")
        probabilities = torch.sigmoid(logits).cpu().tolist()
        for path, probability in zip(paths, probabilities):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid model probability for {path}")
            rows.append({"image_path": path, "pred": float(probability)})
    return rows


def write_predictions(output_json: str | Path, rows):
    """Write only after successful inference; refuse to replace an existing file."""
    output = Path(output_json).expanduser().absolute()
    if output.suffix.lower() != ".json":
        raise ValueError("--output-json must end in .json")
    payload = json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects an existing symlink or concurrent output.
    with output.open("x", encoding="utf-8") as handle:
        try:
            handle.write(payload)
        except Exception:
            handle.close()
            output.unlink(missing_ok=True)
            raise
    return output


def predict(input_dir, output_json, checkpoint_path, config_path,
            *, batch_size=None, num_workers=None, device=None, precision=None):
    root, paths = find_images(input_dir)
    output = Path(output_json).expanduser().absolute()
    if output.suffix.lower() != ".json":
        raise ValueError("--output-json must end in .json")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new filename: {output}")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    from augmentations import ImagePreprocessor
    from model import DinoBinaryClassifier
    from utils import (configure_runtime, get_device, load_config, resolve_precision,
                       runtime_summary, seed_everything, seed_worker)

    config = load_config(config_path)
    runtime = dict(config.get("runtime", {}))
    if device is not None:
        runtime["device"] = device
    if precision is not None:
        runtime["precision"] = precision
    active_device = get_device(runtime.get("device", "auto"))
    configure_runtime(runtime, active_device)
    active_precision = resolve_precision(runtime, active_device)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    evaluation, data = config.get("evaluation", {}), config.get("data", {})
    batch_size = int(evaluation.get("batch_size", 32) if batch_size is None else batch_size)
    # A conservative default works in Colab, Linux, and Windows.
    num_workers = 0 if num_workers is None else int(num_workers)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    print(f"Runtime: {runtime_summary(active_device, active_precision)}", flush=True)
    # Explicit restricted loading: never fall back to unsafe pickle loading.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a project checkpoint containing the 'model' state dict")
    model_config = checkpoint.get("config", config)["model"]
    model = DinoBinaryClassifier(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(active_device).eval()
    preprocessor = ImagePreprocessor(model_config["image_size"], model_config["image_mean"],
                                     model_config["image_std"])
    dataset = UnlabeledImages(root, paths, preprocessor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=bool(data.get("pin_memory", True)) and active_device.type == "cuda",
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
                        drop_last=False)
    print(f"Input: {root} | images: {len(paths)} | output: {output}", flush=True)
    print("Exporting continuous AI scores; no decision threshold is applied.", flush=True)
    rows = score_images(model, loader, active_device, active_precision)
    if len(rows) != len(paths) or len({r["image_path"] for r in rows}) != len(paths):
        raise ValueError("Prediction count or path uniqueness check failed")
    result = write_predictions(output, rows)
    print(f"Saved {len(rows)} predictions to {result}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unlabeled image directory -> JSON AI scores")
    parser.add_argument("--input-dir", required=True, help="Image directory; subdirectories are searched")
    parser.add_argument("--output-json", required=True, help="New JSON file; existing files are not overwritten")
    parser.add_argument("--checkpoint", required=True, help="Trusted project checkpoint, e.g. best.pt")
    parser.add_argument("--config", default="config.mixed100k.yaml", help="Project YAML configuration")
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Override runtime device")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], help="Override precision")
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    predict(args.input_dir, args.output_json, args.checkpoint, args.config,
            batch_size=args.batch_size, num_workers=args.num_workers,
            device=args.device, precision=args.precision)


if __name__ == "__main__":
    main()
