from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but torch.cuda.is_available() is False")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("runtime.device requests MPS, but MPS is unavailable")
    return device


def configure_runtime(config: dict[str, Any], device: torch.device) -> None:
    deterministic = bool(config.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if device.type == "cuda":
        allow_tf32 = bool(config.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True)) and not deterministic
        torch.backends.cudnn.deterministic = deterministic


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    autocast_enabled: bool
    dtype: torch.dtype
    scaler_enabled: bool


def resolve_precision(
    config: dict[str, Any], device: torch.device, legacy_amp: bool = True
) -> PrecisionPolicy:
    requested = str(config.get("precision", "auto")).lower()
    if not legacy_amp or requested in {"fp32", "float32"}:
        return PrecisionPolicy("fp32", False, torch.float32, False)
    if device.type != "cuda":
        return PrecisionPolicy("fp32", False, torch.float32, False)
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("runtime.precision=bf16, but this CUDA device lacks BF16 support")
        return PrecisionPolicy("bf16", True, torch.bfloat16, False)
    if requested in {"fp16", "float16"}:
        return PrecisionPolicy("fp16", True, torch.float16, True)
    raise ValueError("runtime.precision must be auto, fp32, fp16, or bf16")


def make_grad_scaler(policy: PrecisionPolicy) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=policy.scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=policy.scaler_enabled)


def runtime_summary(device: torch.device, policy: PrecisionPolicy) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "device": str(device),
        "precision": policy.name,
        "pytorch": torch.__version__,
    }
    if device.type == "cuda":
        summary.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    elif device.type == "mps":
        summary["mps_available"] = True
    return summary


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def dump_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
