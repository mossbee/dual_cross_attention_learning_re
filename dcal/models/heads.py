"""
Classification and embedding heads for FGVC and Re-ID.

Purpose:
- Provide projection heads to map ViT class tokens (and GLCA outputs) to logits
  for FGVC and ID logits + embeddings for Re-ID.

Design:
- `ClassificationHead`: linear classifier for FGVC.
- `ReIDHead`: linear ID classifier + normalized embedding for metric loss.

I/O:
- ClassificationHead.forward(x) -> logits
- ReIDHead.forward(x) -> dict with logits and embeddings

References:
- Typical ViT fine-tuning heads.
"""

from typing import Dict

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Simple classifier head for FGVC.

    Inputs:
        x: Tensor (B, D)

    Outputs:
        logits: Tensor (B, num_classes)
    """

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)


class ReIDHead(nn.Module):
    """
    Re-ID head with ID classifier and metric embedding.

    Inputs:
        x: Tensor (B, D)

    Outputs:
        Dict with:
            logits: Tensor (B, num_ids)
            embedding: Tensor (B, E) normalized
    """

    def __init__(self, dim: int, num_ids: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_ids)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.embed(x)
        z = self.norm(z)
        z = self.dropout(z)
        # Normalize embedding for metric learning
        emb = torch.nn.functional.normalize(z, dim=1)
        logits = self.classifier(z)
        return {"logits": logits, "embedding": emb}


