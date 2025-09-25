"""
Model package for ViT backbone, GLCA, PWCA, DCAL composition, and rollout.

Overview:
- `vit_backbone.py`: Vision Transformer backbone compatible with attention
  extraction.
- `attention.py`: Global-Local Cross-Attention (GLCA) and Pair-Wise
  Cross-Attention (PWCA) modules.
- `dcal_model.py`: High-level model that coordinates SA, GLCA, and PWCA.
- `rollout.py`: Attention rollout utilities per the paper's Eq. (rollout).
- `heads.py`: Classification heads for FGVC/Re-ID.

References:
- `ViT-pytorch/models/modeling.py` for ViT structure.
- `vit_rollout.py` for attention rollout inspiration.
- `dual_cross_attention_learning.md` for GLCA/PWCA specifications.
"""


def _module_overview() -> None:
    """
    Stub callable for documentation purposes.
    """
    pass


