"""
Distributed training helpers.

Purpose:
- Initialize process groups, wrap models in DDP, and provide utilities for
  gathering tensors and syncing metrics.
"""

from typing import Any
import os
import torch
import torch.distributed as dist


def init_distributed(cfg: Any) -> None:
    """
    Initialize distributed environment if enabled in cfg.
    """
    if not cfg.get("distributed", {}).get("enabled", False):
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def is_main_process() -> bool:
    """
    Return True if current rank is main.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


