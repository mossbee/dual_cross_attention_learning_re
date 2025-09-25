"""
Batch collation utilities.

Purpose:
- Provide collate functions that can form PWCA pairs within a batch and handle
  diverse metadata for FGVC and Re-ID.

Design:
- `collate_fgvc`: returns images and labels.
- `collate_reid`: returns images, pids, camids, and optionally batch pairing map.
"""

from typing import Any, List, Dict

import torch


def collate_fgvc(batch: List[Any]) -> Any:
    """
    Collate function for FGVC batches.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"images": images, "labels": labels}


def collate_reid(batch: List[Any]) -> Any:
    """
    Collate function for Re-ID batches with optional pairing info.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    pids = torch.stack([b["pid"] for b in batch], dim=0)
    camids = torch.stack([b["camid"] for b in batch], dim=0)
    return {"images": images, "pids": pids, "camids": camids}


