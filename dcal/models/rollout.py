"""
Attention Rollout utilities.

Purpose:
- Provide attention rollout per Eq. (rollout) to compute accumulated attention
  from CLS to tokens for selecting top-R local queries for GLCA.

Design:
- Given a list of attention matrices S_l, compute \hat{S}_i = Π_l (0.5*S_l + 0.5*I).
- Expose helpers to select top-R indices and gather local queries.

I/O:
- compute_rollout(attentions) -> accumulated attention matrices per layer.
- select_top_r_from_rollout(rollout, R) -> indices of top-R tokens (exclude CLS).

References:
- `vit_rollout.py` for inspiration.
- `dual_cross_attention_learning.md` Eq. (eq:rollout).
"""

from typing import List, Tuple

import torch


def compute_rollout(attentions: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Compute attention rollout across layers.

    Inputs:
        attentions: List length L with tensors of shape (B, H, N+1, N+1)

    Outputs:
        List length L with accumulated matrices (B, N+1, N+1)

    Notes:
        - Average heads then apply (0.5*S + 0.5*I) and matrix multiply cumulatively.
        - See `vit_rollout.py` for a complete implementation reference.
    """
    assert len(attentions) > 0, "attentions must be non-empty"
    accum_list: List[torch.Tensor] = []
    accum = None
    for S in attentions:
        # S: (B, H, N+1, N+1) -> average over heads
        S_avg = S.mean(dim=1)
        B, Np1, _ = S_avg.shape
        I = torch.eye(Np1, device=S_avg.device, dtype=S_avg.dtype).unsqueeze(0).expand(B, -1, -1)
        S_bar = 0.5 * S_avg + 0.5 * I
        if accum is None:
            accum = S_bar
        else:
            accum = torch.matmul(S_bar, accum)
        accum_list.append(accum)
    return accum_list


def select_top_r_from_rollout(rollout: torch.Tensor, r_fraction: float) -> torch.Tensor:
    """
    Select top-R token indices based on CLS row of rollout.

    Inputs:
        rollout: Tensor (B, N+1, N+1) for a specific layer i
        r_fraction: float in (0, 1], e.g., 0.1 or 0.3 per task

    Outputs:
        Tensor of indices (B, R)

    Notes:
        - Exclude CLS index (0) when ranking tokens; return token indices in [1..N].
    """
    B, Np1, _ = rollout.shape
    assert 0 < r_fraction <= 1.0
    # CLS attends to tokens: take row 0
    cls_row = rollout[:, 0, 1:]  # exclude CLS column
    N = cls_row.shape[1]
    R = max(1, int(N * r_fraction))
    topk = torch.topk(cls_row, k=R, dim=1).indices  # indices in [0..N-1]
    # Shift by +1 to map to token indices excluding CLS
    return topk + 1


