"""
CUB-200-2011 dataset wrapper for FGVC.

Purpose:
- Provide PyTorch Dataset and helper builders for the CUB dataset, including
  reading image paths, labels, and applying augmentations aligned with the
  paper's training settings.

Design:
- Implements a `CUBDataset` returning image tensor and label.
- Follows resizing to 550x550 and random crop to 448x448 for training as noted
  in `dual_cross_attention_learning.md`.

I/O:
- __getitem__(idx) -> Dict with image Tensor and label int.
- build_cub_datasets(root) -> train/val datasets.

References:
- `CUB_200_2011.md` for directory/files structures.
"""

from typing import Dict, Tuple, List

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import build_fgvc_transforms


class CUBDataset(Dataset):
    """
    Skeleton CUB dataset.

    Inputs:
        root: path to CUB dataset root
        split: str, one of {"train", "val", "test"}
        transform: optional transform callable

    Outputs:
        Dict with keys: image (Tensor), label (int), and optional meta.
    """

    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else build_fgvc_transforms(train=(split == "train"))

        # Parse CUB metadata files
        images_path = os.path.join(root, "images.txt")
        labels_path = os.path.join(root, "image_class_labels.txt")
        split_path = os.path.join(root, "train_test_split.txt")

        with open(images_path, "r") as f:
            id_to_relpath = {int(line.strip().split(" ")[0]): line.strip().split(" ")[1] for line in f}
        with open(labels_path, "r") as f:
            id_to_label = {int(line.strip().split(" ")[0]): int(line.strip().split(" ")[1]) - 1 for line in f}
        with open(split_path, "r") as f:
            id_to_is_train = {int(line.strip().split(" ")[0]): int(line.strip().split(" ")[1]) == 1 for line in f}

        self.samples: List[Tuple[str, int]] = []
        for img_id, rel in id_to_relpath.items():
            is_tr = id_to_is_train[img_id]
            if (split == "train" and is_tr) or (split in {"val", "test"} and (not is_tr)):
                path = os.path.join(root, "images", rel)
                label = id_to_label[img_id]
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long), "path": path}


def build_cub_datasets(root: str) -> Tuple[Dataset, Dataset]:
    """
    Build train and val datasets for CUB.
    """
    train = CUBDataset(root=root, split="train")
    val = CUBDataset(root=root, split="val")
    return train, val


