"""
Uncertainty-based loss weighting (Kendall et al., 2018).

Purpose:
- Combine SA, GLCA, and PWCA losses into a single objective with learnable
  log-variance parameters w_i as in `dual_cross_attention_learning.md`.

I/O:
- UncertaintyWeighting: nn.Module returning weighted sum and tracking w_i.
"""

from typing import Dict

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """
    Skeleton uncertainty-based weighting.

    Inputs:
        losses: Dict with keys like {"sa": L_sa, "glca": L_glca, "pwca": L_pwca}

    Outputs:
        Dict with:
            total: scalar weighted loss
            weights: Tensor of current weights
    """

    def __init__(self, num_terms: int = 3) -> None:
        super().__init__()
        # Learnable log variances w_i
        self.log_vars = nn.Parameter(torch.zeros(num_terms))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        terms = list(losses.values())
        assert len(terms) == self.log_vars.numel(), "Mismatch number of loss terms"
        total = 0.0
        for i, Li in enumerate(terms):
            wi = self.log_vars[i]
            total = total + 0.5 * (torch.exp(-wi) * Li + wi)
        return {"total": total, "weights": self.log_vars.detach().clone()}


