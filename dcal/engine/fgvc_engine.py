"""
FGVC training/evaluation engine.

Purpose:
- Implement task-specific forward, loss computation (cross-entropy for SA/GLCA),
  uncertainty-weighted loss aggregation, and top-1 accuracy metrics.

Design:
- `FGVCEngine.step(batch)` returns loss dict and metrics.

References:
- `dual_cross_attention_learning.md` for losses and inference aggregation.
"""

from typing import Dict, Any

import torch
from dcal.models.dcal_model import DCALModel
from dcal.losses.classification import classification_loss
from dcal.losses.uncertainty import UncertaintyWeighting
from dcal.engine.evaluator import top1_accuracy

class FGVCEngine:
    """
    Skeleton engine for FGVC.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        ds = cfg["dataset"]
        md = cfg["model"]
        self.model = DCALModel(
            task="fgvc",
            image_size=ds["image_size"],
            patch_size=16,
            embed_dim=768 if md.get("vit", "ViT-B_16").startswith("ViT-B") else 384,
            depth=md.get("depth", 12),
            num_heads=12,
            r_fraction=ds.get("r_fraction", 0.1),
            num_classes=ds["num_classes"],
            use_pwca=False,  # paper: PWCA only used in training; FGVC can also avoid it
        )
        self.uncertainty = UncertaintyWeighting(num_terms=2)

    def step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["images"]
        labels = batch["labels"]
        out = self.model({"images": images})
        L_sa = classification_loss(out["sa"]["logits"], labels)
        L_glca = classification_loss(out["glca"]["logits"], labels)
        total = self.uncertainty({"sa": L_sa, "glca": L_glca})
        acc = top1_accuracy(out["sa"]["logits"], labels)
        return {"loss": total["total"], "acc1": acc, "L_sa": L_sa.detach(), "L_glca": L_glca.detach()}


