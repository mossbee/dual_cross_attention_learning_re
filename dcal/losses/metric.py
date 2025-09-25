"""
Metric learning losses for Re-ID.

Purpose:
- Provide triplet loss with hard mining or batch-all options.

I/O:
- triplet_loss(embeddings, pids, margin) -> scalar loss tensor
"""

import torch


def triplet_loss(embeddings: torch.Tensor, pids: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Compute triplet loss for Re-ID.

    Inputs:
        embeddings: Tensor (B, E) normalized
        pids: Tensor (B,) identity labels
        margin: float margin for triplet

    Outputs:
        loss: Tensor scalar
    """
    # Pairwise distance matrix
    dist = torch.cdist(embeddings, embeddings, p=2)
    N = dist.size(0)

    # Build masks
    pid_eq = pids.unsqueeze(0) == pids.unsqueeze(1)
    is_pos = pid_eq & (~torch.eye(N, dtype=torch.bool, device=dist.device))
    is_neg = ~pid_eq

    # For each anchor, hardest positive and hardest negative
    dist_pos = dist.clone()
    dist_pos[~is_pos] = -1.0
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist.clone()
    dist_neg[~is_neg] = -1.0
    # Replace invalid negatives with +inf by masking
    dist_neg = dist_neg.masked_fill(dist_neg < 0, float("inf"))
    hardest_neg, _ = dist_neg.min(dim=1)

    loss = torch.relu(hardest_pos - hardest_neg + margin).mean()
    return loss


