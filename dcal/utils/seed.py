"""
Seeding utilities for reproducibility.

Purpose:
- Set seeds for Python, NumPy, and PyTorch; optionally enable deterministic
  backend behavior.
"""

from typing import Optional
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    """
    Set random seeds and deterministic flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


