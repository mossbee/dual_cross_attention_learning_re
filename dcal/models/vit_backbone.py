"""
Vision Transformer (ViT) backbone skeleton.

Purpose:
- Provide a ViT encoder that exposes per-layer attention matrices and token
  embeddings needed for SA, GLCA, PWCA, and attention rollout.

Design:
- Follows the structure of ViT from `ViT-pytorch/models/modeling.py` but kept
  minimal here and adapted to return attention weights.
- Includes support for class token, positional embeddings, and MSA blocks.

Key I/O:
- forward(images) -> dict with:
    - "tokens": final token embeddings (B, N+1, D)
    - "cls": final class token embedding (B, D)
    - "attentions": list of attention weight tensors per layer (B, H, N+1, N+1)

References:
- `ViT-pytorch/models/modeling.py` for a full working implementation.
- `dual_cross_attention_learning.md` for requirements on attention exposure.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class VisionTransformerBackbone(nn.Module):
    """
    Skeleton ViT backbone exposing attention weights.

    Inputs:
        images: Tensor of shape (B, 3, H, W)

    Outputs:
        Dict with keys:
            - tokens: Tensor (B, N+1, D)
            - cls: Tensor (B, D)
            - attentions: List[Tensor] length L, each (B, H, N+1, N+1)

    Refer to:
        - `ViT-pytorch/models/modeling.py` for full details on patch embedding,
          transformer blocks, and attention implementation.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Linear and LayerNorm will be initialized by PyTorch defaults which are acceptable

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ViT.

        Args:
            images: Tensor (B, 3, H, W)

        Returns:
            Dict[str, Any]: tokens, cls, attentions

        Notes:
            - Attention weights need to be collected from each MSA for rollout
              and GLCA selection.
        """
        B = images.shape[0]
        x = self.patch_embed(images)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attentions: List[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x)
            attentions.append(attn)  # (B, H, N+1, N+1)

        x = self.norm(x)
        cls = x[:, 0]
        return {"tokens": x, "cls": cls, "attentions": attentions}


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth per sample.

    Reference implementation adapted from timm.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


