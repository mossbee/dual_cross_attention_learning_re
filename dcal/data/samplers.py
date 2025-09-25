"""
Samplers for Re-ID and PWCA pairing.

Purpose:
- Provide identity-balanced sampler for Re-ID (e.g., P identities x K images).
- Provide pair sampler for PWCA: sample pairs within a batch for cross-attention.

Design:
- `IdentityPKSampler`: ensures P x K structure.
- `PairSampler`: emits index pairs (i, j) for PWCA.
"""

from typing import Iterator, List, Dict

import random
from collections import defaultdict
from torch.utils.data import Sampler


class IdentityPKSampler(Sampler[List[int]]):
    """
    Skeleton PK sampler for Re-ID.
    """

    def __init__(self, data_source, num_p: int, num_k: int) -> None:
        self.data_source = data_source
        self.num_p = num_p
        self.num_k = num_k
        # Build pid -> indices mapping
        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(data_source)):
            item = data_source[idx]
            pid = int(item["pid"]) if isinstance(item, dict) else int(item.pid)
            self.pid_to_indices[pid].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        pids = list(self.pid_to_indices.keys())
        random.shuffle(pids)
        batch: List[int] = []
        for i in range(0, len(pids), self.num_p):
            selected = pids[i : i + self.num_p]
            for pid in selected:
                inds = self.pid_to_indices[pid]
                if len(inds) >= self.num_k:
                    chosen = random.sample(inds, self.num_k)
                else:
                    chosen = random.choices(inds, k=self.num_k)
                batch.extend(chosen)
            yield batch
            batch = []

    def __len__(self) -> int:
        # Approximate number of batches
        num_identities = len(self.pid_to_indices)
        return max(1, num_identities // self.num_p)


class PairSampler(Sampler[List[int]]):
    """
    Skeleton pair sampler for PWCA.
    """

    def __init__(self, data_source, pair_ratio: float = 1.0) -> None:
        self.data_source = data_source
        self.pair_ratio = pair_ratio

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        # Pair adjacent samples; replicate pairs according to ratio
        pairs: List[int] = []
        for i in range(0, len(indices) - 1, 2):
            a, b = indices[i], indices[i + 1]
            pairs.extend([a, b])
            if self.pair_ratio > 1.0:
                rep = int(self.pair_ratio) - 1
                for _ in range(rep):
                    pairs.extend([a, b])
        yield pairs

    def __len__(self) -> int:
        return 1


