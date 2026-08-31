from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class DinoBinaryClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config["backbone"])
        if config.get("gradient_checkpointing", True) and hasattr(
            self.backbone, "gradient_checkpointing_enable"
        ):
            self.backbone.gradient_checkpointing_enable()
        patch_size = self.backbone.config.patch_size
        patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        if int(config["image_size"]) % patch_size != 0:
            raise ValueError(
                f"model.image_size={config['image_size']} must be divisible by patch size {patch_size}"
            )
        self.pooling = config.get("pooling", "cls_mean")
        if self.pooling not in {"cls", "cls_mean"}:
            raise ValueError("model.pooling must be 'cls' or 'cls_mean'")

        hidden_size = int(self.backbone.config.hidden_size)
        feature_size = hidden_size if self.pooling == "cls" else hidden_size * 2
        head_hidden = int(config.get("head_hidden_dim", 512))
        self.head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, head_hidden),
            nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.2))),
            nn.Linear(head_hidden, 1),
        )

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        limit = int(config.get("max_parameters", 2_000_000_000))
        if parameter_count >= limit:
            raise ValueError(
                f"Model has {parameter_count:,} parameters, violating the <{limit:,} limit"
            )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        cls_feature = hidden[:, 0]
        if self.pooling == "cls":
            return cls_feature
        patch_mean = hidden[:, 1:].mean(dim=1)
        return torch.cat([cls_feature, patch_mean], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(pixel_values)
        logits = self.head(features).squeeze(-1)
        return logits, features

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
