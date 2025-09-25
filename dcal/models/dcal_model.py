"""
DCAL composite model orchestrating SA, GLCA, and PWCA branches.

Purpose:
- Build the overall training and inference graph per `dual_cross_attention_learning.md`.
- Coordinate ViT backbone, GLCA block (M=1), and PWCA blocks (T=L during training).

Design:
- During training:
    - SA branch: standard ViT forward to produce logits.
    - GLCA branch: select top-R local queries via rollout and apply GLCA.
    - PWCA branch: for paired inputs, compute PWCA outputs and a parallel head.
    - Losses are combined with uncertainty-based weighting.
- During inference:
    - PWCA disabled. Use SA and GLCA outputs for final prediction.

Key I/O:
- forward(batch) -> dict with per-branch outputs and intermediate tensors.

References:
- `dual_cross_attention_learning.md`, Sections Global-Local Cross-Attention and Pair-Wise Cross-Attention.
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .vit_backbone import VisionTransformerBackbone
from .attention import GlobalLocalCrossAttention, PairWiseCrossAttention
from .rollout import compute_rollout, select_top_r_from_rollout
from .heads import ClassificationHead, ReIDHead


class DCALModel(nn.Module):
    """
    Skeleton DCAL model wiring backbone, GLCA, PWCA, and heads.

    Inputs:
        batch: Dict with tensors required by the selected task.

    Outputs:
        Dict with keys including but not limited to:
            - logits_sa: Tensor for SA branch logits
            - logits_glca: Tensor for GLCA branch logits
            - logits_pwca: (training only) Tensor for PWCA branch logits
            - attentions: List of attention maps from backbone
            - rollout: Optional rollout maps for GLCA selection
    """

    def __init__(
        self,
        task: str,
        image_size: int,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        r_fraction: float = 0.1,
        num_classes: Optional[int] = None,
        num_ids: Optional[int] = None,
        reid_embed_dim: int = 512,
        use_pwca: bool = True,
    ) -> None:
        super().__init__()
        assert task in {"fgvc", "reid"}
        self.task = task
        self.r_fraction = r_fraction
        self.use_pwca = use_pwca

        self.backbone = VisionTransformerBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.glca = GlobalLocalCrossAttention(dim=embed_dim, num_heads=num_heads)
        self.pwca = PairWiseCrossAttention(dim=embed_dim, num_heads=num_heads)

        if task == "fgvc":
            assert num_classes is not None
            self.sa_head = ClassificationHead(embed_dim, num_classes)
            self.glca_head = ClassificationHead(embed_dim, num_classes)
            self.pwca_head = ClassificationHead(embed_dim, num_classes)
        else:
            assert num_ids is not None
            self.sa_head = ReIDHead(embed_dim, num_ids, reid_embed_dim)
            self.glca_head = ReIDHead(embed_dim, num_ids, reid_embed_dim)
            self.pwca_head = ReIDHead(embed_dim, num_ids, reid_embed_dim)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass coordinating branches per training/inference mode.
        """
        images: torch.Tensor = batch["images"]
        bb = self.backbone(images)
        tokens: torch.Tensor = bb["tokens"]  # (B, N+1, D)
        cls: torch.Tensor = bb["cls"]  # (B, D)
        attentions = bb["attentions"]

        outputs: Dict[str, Any] = {"attentions": attentions}

        # SA branch
        if self.task == "fgvc":
            logits_sa = self.sa_head(cls)
            outputs["sa"] = {"logits": logits_sa, "cls": cls}
        else:
            sa = self.sa_head(cls)
            # Merge dict outputs without relying on Python 3.9+ dict union
            sa_out = dict(sa)
            sa_out["cls"] = cls
            outputs["sa"] = sa_out

        # GLCA branch via rollout-based top-R selection
        accum_list = compute_rollout(attentions)
        accum_last = accum_list[-1]
        top_idx = select_top_r_from_rollout(accum_last, self.r_fraction)  # (B, R) indices in [1..N]
        B, R = top_idx.shape
        D = tokens.shape[-1]
        idx_expanded = top_idx.unsqueeze(-1).expand(B, R, D)
        local_queries = tokens.gather(dim=1, index=idx_expanded)  # (B, R, D)
        global_kv = tokens  # (B, N+1, D)
        glca_feats = self.glca(local_queries, global_kv, global_kv)  # (B, R, D)
        glca_pooled = glca_feats.mean(dim=1)  # (B, D)
        if self.task == "fgvc":
            logits_glca = self.glca_head(glca_pooled)
            outputs["glca"] = {"logits": logits_glca, "pooled": glca_pooled}
        else:
            gl = self.glca_head(glca_pooled)
            gl_out = dict(gl)
            gl_out["pooled"] = glca_pooled
            outputs["glca"] = gl_out

        # PWCA branch (training only)
        if self.training and self.use_pwca:
            B = tokens.size(0)
            # Pair indices by rolling by 1
            pair_idx = torch.roll(torch.arange(B, device=tokens.device), shifts=1)
            Q_target = tokens  # (B, N+1, D)
            K1 = tokens
            V1 = tokens
            K2 = tokens[pair_idx]
            V2 = tokens[pair_idx]
            pw_tokens = self.pwca(Q_target, K1, V1, K2, V2)  # (B, N+1, D)
            cls_pw = pw_tokens[:, 0]
            if self.task == "fgvc":
                logits_pw = self.pwca_head(cls_pw)
                outputs["pwca"] = {"logits": logits_pw, "cls": cls_pw}
            else:
                pw = self.pwca_head(cls_pw)
                pw_out = dict(pw)
                pw_out["cls"] = cls_pw
                outputs["pwca"] = pw_out

        return outputs


