"""
Re-ID training/evaluation engine.

Purpose:
- Implement task-specific forward, loss computation (ID cross-entropy + triplet
  for SA/GLCA and PWCA during training), and mAP/CMC metrics.

Design:
- `ReIDEngine.step(batch)` returns loss dict and metrics.
"""

from typing import Dict, Any
import torch
from dcal.models.dcal_model import DCALModel
from dcal.losses.classification import classification_loss
from dcal.losses.metric import triplet_loss
from dcal.losses.uncertainty import UncertaintyWeighting


class ReIDEngine:
    """
    Skeleton engine for Re-ID.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        ds = cfg["dataset"]
        md = cfg["model"]
        self.model = DCALModel(
            task="reid",
            image_size=ds["image_size"],
            patch_size=16,
            embed_dim=768 if md.get("vit", "ViT-B_16").startswith("ViT-B") else 384,
            depth=md.get("depth", 12),
            num_heads=12,
            r_fraction=ds.get("r_fraction", 0.3),
            num_ids=ds.get("num_ids", 576),
            reid_embed_dim=md.get("embed_dim", 512),
            use_pwca=True,
        )
        self.uncertainty = UncertaintyWeighting(num_terms=3)

    def step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["images"]
        pids = batch["pids"]
        out = self.model({"images": images})
        L_id_sa = classification_loss(out["sa"]["logits"], pids)
        L_id_glca = classification_loss(out["glca"]["logits"], pids)
        L_id_pw = classification_loss(out["pwca"]["logits"], pids) if "pwca" in out else torch.tensor(0.0, device=images.device)
        L_tri_sa = triplet_loss(out["sa"]["embedding"], pids)
        L_tri_glca = triplet_loss(out["glca"]["embedding"], pids)
        L_tri_pw = triplet_loss(out["pwca"]["embedding"], pids) if "pwca" in out else torch.tensor(0.0, device=images.device)
        L_sa = L_id_sa + L_tri_sa
        L_glca = L_id_glca + L_tri_glca
        L_pw = L_id_pw + L_tri_pw
        total = self.uncertainty({"sa": L_sa, "glca": L_glca, "pwca": L_pw})
        return {"loss": total["total"], "L_sa": L_sa.detach(), "L_glca": L_glca.detach(), "L_pw": L_pw.detach()}


