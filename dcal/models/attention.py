"""
Attention modules: Global-Local Cross-Attention (GLCA) and Pair-Wise Cross-Attention (PWCA).

Purpose:
- Implement GLCA per Eq. (glca) and PWCA per Eq. (pwca) in
  `dual_cross_attention_learning.md`.

Design:
- GLCA selects top-R local queries based on attention rollout of CLS to tokens,
  then performs cross-attention against global K,V.
- PWCA builds a combined K,V from paired images and computes attention for the
  target image's Q against concatenated K,V. Used only during training.

Key I/O:
- GLCA.forward(Q_local, K_global, V_global) -> local enhanced features
- PWCA.forward(Q_target, K_1, V_1, K_2, V_2) -> contaminated attention outputs

References:
- `dual_cross_attention_learning.md` equations eq:glca, eq:pwca.
- For MSA details, see `ViT-pytorch/models/modeling.py`.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MultiHeadCrossAttention(nn.Module):
    """
    Generic Multi-Head Cross-Attention module.

    Inputs:
        Q: Tensor (B, Nq, D)
        K: Tensor (B, Nk, D)
        V: Tensor (B, Nk, D)

    Outputs:
        Tensor: (B, Nq, D)
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        q = self.w_q(Q).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.w_k(K).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.w_v(V).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, H, Nq, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class GlobalLocalCrossAttention(nn.Module):
    """
    Skeleton GLCA module.

    Inputs:
        Q_local: Tensor (B, R, D)
        K_global: Tensor (B, N+1, D)
        V_global: Tensor (B, N+1, D)

    Outputs:
        Tensor: (B, R, D) cross-attended features
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        self.xattn = MultiHeadCrossAttention(dim, num_heads, attn_dropout, proj_dropout)

    def forward(self, Q_local: torch.Tensor, K_global: torch.Tensor, V_global: torch.Tensor) -> torch.Tensor:
        """
        Compute softmax(Q_local K_global^T / sqrt(d)) V_global as in eq:glca.
        """
        return self.xattn(Q_local, K_global, V_global)


class PairWiseCrossAttention(nn.Module):
    """
    Skeleton PWCA module, used in training only.

    Inputs:
        Q_target: Tensor (B, N+1, D) of the target image
        K_1, V_1: Tensors (B, N+1, D) for target image
        K_2, V_2: Tensors (B, N+1, D) for distractor image

    Outputs:
        Tensor: (B, N+1, D) contaminated attention outputs
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        self.xattn = MultiHeadCrossAttention(dim, num_heads, attn_dropout, proj_dropout)

    def forward(
        self,
        Q_target: torch.Tensor,
        K_1: torch.Tensor,
        V_1: torch.Tensor,
        K_2: torch.Tensor,
        V_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax(Q_target [K_1;K_2]^T / sqrt(d)) [V_1;V_2] as in eq:pwca.
        """
        Kc = torch.cat([K_1, K_2], dim=1)
        Vc = torch.cat([V_1, V_2], dim=1)
        return self.xattn(Q_target, Kc, Vc)


