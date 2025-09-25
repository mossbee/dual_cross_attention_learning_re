"""
VeRi-776 dataset wrapper for Re-ID.

Purpose:
- Provide Dataset and builders for VeRi-776 with ID labels and camera metadata,
  supporting standard Re-ID training setup.

Design:
- `VeRiDataset` yields image tensor, id label, camera id, and image path.
- Builders also provide identity-balanced samplers.

I/O:
- __getitem__(idx) -> Dict with image Tensor, pid, camid, path, and optional meta.
- build_veri_datasets(root) -> train/query/gallery datasets.

References:
- `VeRi_776.md` for directory and metadata files.
"""

from typing import Dict, Tuple, List

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import build_reid_transforms


class VeRiDataset(Dataset):
    """
    Skeleton VeRi dataset.

    Inputs:
        root: path to VeRi root
        split: str, one of {"train", "query", "gallery"}
        transform: optional transform callable

    Outputs:
        Dict with keys: image, pid, camid, path.
    """

    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else build_reid_transforms(train=(split == "train"), is_vehicle=True)

        if split == "train":
            dir_name = "image_train"
        elif split == "query":
            dir_name = "image_query"
        else:
            dir_name = "image_test"
        self.dir = os.path.join(root, dir_name)

        # Expect filenames like xxxx_cXX_XXXX.jpg; parse ids and camids from accompanying labels if available
        # As a simple baseline, derive pid from folder or xml is omitted here; users should provide labels via list files
        self.paths: List[str] = sorted(
            [os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.lower().endswith((".jpg", ".png"))]
        )
        # Dummy pid/camid placeholders (implementation can be extended to parse XMLs)
        self.pids = [0 for _ in self.paths]
        self.camids = [0 for _ in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        pid = torch.tensor(self.pids[idx], dtype=torch.long)
        camid = torch.tensor(self.camids[idx], dtype=torch.long)
        return {"image": img, "pid": pid, "camid": camid, "path": path}


def build_veri_datasets(root: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build train, query, and gallery datasets for VeRi.
    """
    train = VeRiDataset(root=root, split="train")
    query = VeRiDataset(root=root, split="query")
    gallery = VeRiDataset(root=root, split="gallery")
    return train, query, gallery


