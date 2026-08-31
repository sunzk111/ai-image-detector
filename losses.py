from __future__ import annotations

import torch
import torch.nn.functional as F


def paired_classification_consistency_loss(
    clean_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    clean_features: torch.Tensor,
    augmented_features: torch.Tensor,
    labels: torch.Tensor,
    clean_bce_weight: float = 1.0,
    augmented_bce_weight: float = 1.0,
    lambda_consistency: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, labels)
    augmented_bce = F.binary_cross_entropy_with_logits(augmented_logits, labels)
    consistency = (1.0 - F.cosine_similarity(clean_features, augmented_features, dim=-1)).mean()
    total = (
        float(clean_bce_weight) * clean_bce
        + float(augmented_bce_weight) * augmented_bce
        + float(lambda_consistency) * consistency
    )
    return total, {
        "total": total.detach(),
        "bce_clean": clean_bce.detach(),
        "bce_augmented": augmented_bce.detach(),
        "consistency": consistency.detach(),
    }
