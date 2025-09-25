"""
Evaluation utilities for FGVC and Re-ID.

Purpose:
- Provide accuracy computation for FGVC.
- Provide CMC and mAP metrics for Re-ID with standard protocols.
"""

from typing import Dict

import torch


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute top-1 accuracy for FGVC.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().mean()
    return correct


def reid_metrics(embeddings: torch.Tensor, pids: torch.Tensor, camids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute standard Re-ID mAP and CMC metrics.
    """
    # Minimal placeholder: compute cosine similarity self-retrieval metrics
    # For full protocol, implement query/gallery split evaluation.
    sims = torch.mm(embeddings, embeddings.t())
    sims.fill_diagonal_(-1)
    nn_idx = sims.argmax(dim=1)
    correct = (pids == pids[nn_idx]).float().mean()
    return {"top1-reid": correct}


