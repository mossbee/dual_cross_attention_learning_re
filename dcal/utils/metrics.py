"""
Generic metric helpers.

Purpose:
- Provide running averages and meters useful for training.
"""


class AverageMeter:
    """
    Track running average of a scalar metric.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    def compute(self) -> float:
        return self.sum / max(1, self.count)

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


