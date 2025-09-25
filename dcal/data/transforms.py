"""
Data transforms and augmentations for FGVC and Re-ID.

Purpose:
- Provide per-task preprocessing and augmentation pipelines.

Design:
- `build_fgvc_transforms()` per paper: resize 550, random crop 448, etc.
- `build_reid_transforms()` per paper: resize to (256,128) person or (256,256) vehicle.

I/O:
- Functions return torchvision-like transform callables.
"""

from typing import Any
from torchvision import transforms


def build_fgvc_transforms(train: bool) -> Any:
    """
    Build FGVC transforms for CUB training/validation.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize(550),
                transforms.RandomCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(550),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def build_reid_transforms(train: bool, is_vehicle: bool) -> Any:
    """
    Build Re-ID transforms for person or vehicle datasets.
    """
    size = (256, 256) if is_vehicle else (256, 128)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


