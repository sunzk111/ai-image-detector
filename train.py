from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from augmentations import CompoundDegradation, ImagePreprocessor
from dataset import build_dataset
from losses import paired_classification_consistency_loss
from metrics import binary_metrics
from model import DinoBinaryClassifier
from threshold_calibration import calibrated_validation, fixed_calibration_split, split_fingerprint
from utils import (
    PrecisionPolicy,
    atomic_torch_save,
    configure_runtime,
    dump_json,
    get_device,
    load_config,
    make_grad_scaler,
    resolve_precision,
    runtime_summary,
    seed_everything,
    seed_worker,
)


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    config: dict[str, Any],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def train_one_epoch(
    model: DinoBinaryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    precision: PrecisionPolicy,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    totals: defaultdict[str, float] = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, batch in enumerate(progress):
        clean = batch["clean"].to(device, non_blocking=True)
        augmented = batch["augmented"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=precision.dtype,
            enabled=precision.autocast_enabled,
        ):
            logits, features = model(torch.cat([clean, augmented], dim=0))
            clean_logits, augmented_logits = logits.chunk(2)
            clean_features, augmented_features = features.chunk(2)
            loss, parts = paired_classification_consistency_loss(
                clean_logits,
                augmented_logits,
                clean_features,
                augmented_features,
                labels,
                clean_bce_weight=float(config["loss"].get("clean_bce_weight", 1.0)),
                augmented_bce_weight=float(
                    config["loss"].get("augmented_bce_weight", 1.0)
                ),
                lambda_consistency=float(config["loss"]["lambda_consistency"]),
            )
            scaled_loss = loss / accumulation

        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key, value in parts.items():
            totals[key] += float(value.item())
        progress.set_postfix(loss=f"{parts['total'].item():.4f}")

    return {key: value / len(loader) for key, value in totals.items()}


@torch.inference_mode()
def validate(
    model: DinoBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
    precision: PrecisionPolicy,
    threshold: float,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    loss_total = 0.0
    sample_ids: list[str] = []
    individual_losses: list[float] = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["clean"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=precision.dtype,
            enabled=precision.autocast_enabled,
        ):
            logits, _ = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if calibration and calibration.get("enabled", False):
                individual_losses.extend(F.binary_cross_entropy_with_logits(
                    logits.float(), targets.float(), reduction="none"
                ).cpu().tolist())
        loss_total += float(loss.item())
        labels.extend(targets.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        if calibration and calibration.get("enabled", False):
            sample_ids.extend(str(value) for value in batch["sample_id"])
    if calibration and calibration.get("enabled", False):
        result, validation_indices = calibrated_validation(
            labels, probabilities, sample_ids, calibration, threshold
        )
        result["loss"] = sum(individual_losses[i] for i in validation_indices) / len(validation_indices)
        return result
    result = binary_metrics(labels, probabilities, threshold)
    result["loss"] = loss_total / len(loader)
    result["threshold"] = threshold
    return result


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    runtime_config = config.get("runtime", {})
    device = get_device(runtime_config.get("device", "auto"))
    configure_runtime(runtime_config, device)
    precision = resolve_precision(
        runtime_config,
        device,
        legacy_amp=bool(config["training"].get("amp", True)),
    )
    scaler = make_grad_scaler(precision)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DinoBinaryClassifier(config["model"])
    preprocessor = ImagePreprocessor(
        config["model"]["image_size"],
        config["model"]["image_mean"],
        config["model"]["image_std"],
    )
    train_dataset = build_dataset(
        config["data"]["train"],
        config["data"],
        preprocessor,
        degradation=CompoundDegradation(config["augmentation"]),
    )
    val_dataset = build_dataset(config["data"]["val"], config["data"], preprocessor)
    calibration_config = config["evaluation"].get("calibration", {})
    split_record = None
    if calibration_config.get("enabled", False) and hasattr(val_dataset, "samples"):
        ids = [sample.sample_id for sample in val_dataset.samples]
        labels = [sample.label for sample in val_dataset.samples]
        cal_indices, eval_indices = fixed_calibration_split(
            labels, ids, calibration_config.get("fraction", 0.25), calibration_config.get("seed", 42)
        )
        split_record = {
            "calibration_ids": [ids[i] for i in cal_indices],
            "validation_ids": [ids[i] for i in eval_indices],
            "calibration_ids_sha256": split_fingerprint(ids, cal_indices),
            "validation_ids_sha256": split_fingerprint(ids, eval_indices),
        }
        dump_json(split_record, output_dir / "calibration_split.json")
        print(f"Fixed internal split: {len(cal_indices)} calibration / {len(eval_indices)} validation")
    train_loader = make_loader(
        train_dataset, int(config["training"]["batch_size"]), config["data"], True, seed
    )
    val_loader = make_loader(
        val_dataset, int(config["evaluation"]["batch_size"]), config["data"], False, seed
    )

    freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    model.set_backbone_trainable(freeze_epochs == 0)
    model.to(device)
    print(f"Runtime: {runtime_summary(device, precision)}")
    print(f"Parameters: {model.parameter_summary()}")
    print(f"Train/val samples: {len(train_dataset)}/{len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(config["training"]["backbone_lr"])},
            {"params": model.head.parameters(), "lr": float(config["training"]["head_lr"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    updates_per_epoch = math.ceil(
        len(train_loader) / int(config["training"].get("gradient_accumulation_steps", 1))
    )
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = round(total_steps * float(config["training"].get("warmup_fraction", 0.1)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_accuracy = -1.0
    resume = config["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        if config["evaluation"].get("calibration", {}).get("enabled", False):
            saved_policy = checkpoint.get("config", {}).get("evaluation", {}).get("calibration", {})
            if saved_policy != config["evaluation"]["calibration"] or not checkpoint.get("threshold_calibration"):
                raise ValueError("Calibration policy changed or old checkpoint lacks calibration. "
                                 "Start a new run; do not resume the old validation selection state.")
            if split_record and any(
                checkpoint["threshold_calibration"].get(key) != split_record[key]
                for key in ("calibration_ids_sha256", "validation_ids_sha256")
            ):
                raise ValueError("Internal validation data changed since this checkpoint; start a new run.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        model.set_backbone_trainable(start_epoch >= freeze_epochs)

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs:
            model.set_backbone_trainable(True)
            print("Backbone unfrozen")
        if hasattr(train_dataset, "set_progress"):
            train_dataset.set_progress(epoch / max(1, epochs - 1))
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, precision, device, config
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            precision,
            threshold=float(config["evaluation"].get("threshold", 0.5)),
            calibration=config["evaluation"].get("calibration"),
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_accuracy": max(best_accuracy, float(val_metrics["accuracy"])),
            "config": config,
            "decision_threshold": float(val_metrics["threshold"]),
            "threshold_calibration": val_metrics.get("calibration"),
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            payload["best_accuracy"] = best_accuracy
            atomic_torch_save(payload, output_dir / "best.pt")
            dump_json({
                "epoch": epoch,
                "checkpoint": "best.pt",
                "threshold": payload["decision_threshold"],
                "calibration": payload["threshold_calibration"],
                "validation_count": val_metrics["count"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_accuracy_at_0_5": val_metrics.get("accuracy_at_0_5"),
            }, output_dir / "best_threshold.json")
        dump_json(history, output_dir / "history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the V0-V2 DINOv2 AIGC detector")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
