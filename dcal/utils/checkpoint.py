"""
Checkpoint utilities.

Purpose:
- Save and load model, optimizer, scheduler states, and training metadata.
"""

from typing import Any, Dict
import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Save a training state to disk.
    """
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a training state from disk.
    """
    return torch.load(path, map_location="cpu")


