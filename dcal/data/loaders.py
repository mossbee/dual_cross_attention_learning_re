"""
DataLoader builders.

Purpose:
- Provide unified builders to create DataLoaders for FGVC (CUB) and Re-ID (VeRi),
  including appropriate samplers and collate functions.

Design:
- `build_fgvc_loaders(cfg)` -> train_loader, val_loader
- `build_reid_loaders(cfg)` -> train_loader, query_loader, gallery_loader
"""

from typing import Any, Tuple
from torch.utils.data import DataLoader
from .fgvc_cub import build_cub_datasets
from .reid_veri import build_veri_datasets
from .collate import collate_fgvc, collate_reid
from .samplers import IdentityPKSampler


def build_fgvc_loaders(cfg: Any) -> Tuple[Any, Any]:
    """
    Build loaders for FGVC.
    """
    train_set, val_set = build_cub_datasets(cfg["dataset"]["root"])
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fgvc)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fgvc)
    return train_loader, val_loader


def build_reid_loaders(cfg: Any) -> Tuple[Any, Any, Any]:
    """
    Build loaders for Re-ID.
    """
    train_set, query_set, gallery_set = build_veri_datasets(cfg["dataset"]["root"])
    p = max(1, cfg["train"].get("batch_size", 64) // cfg["train"].get("images_per_id", 4))
    k = cfg["train"].get("images_per_id", 4)
    sampler = IdentityPKSampler(train_set, num_p=p, num_k=k)
    # Use batch_sampler instead of sampler+batch_size to respect PK grouping
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_reid,
    )
    query_loader = DataLoader(query_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_reid)
    gallery_loader = DataLoader(gallery_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_reid)
    return train_loader, query_loader, gallery_loader


