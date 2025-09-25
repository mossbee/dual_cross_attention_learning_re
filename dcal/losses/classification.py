"""
Classification losses for FGVC and Re-ID ID heads.

Purpose:
- Provide cross-entropy loss with optional label smoothing.

I/O:
- classification_loss(logits, targets, smoothing) -> scalar loss tensor
"""

from typing import Optional

import torch


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: Optional[float] = None) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Inputs:
        logits: Tensor (B, C)
        targets: Tensor (B,)
        smoothing: optional label smoothing coefficient

    Outputs:
        loss: Tensor scalar
    """
    if smoothing is None or smoothing <= 0.0:
        return torch.nn.functional.cross_entropy(logits, targets)
    with torch.no_grad():
        num_classes = logits.size(1)
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = (-true_dist * log_probs).sum(dim=1).mean()
    return loss


