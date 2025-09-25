"""
Logging utilities.

Purpose:
- Provide console/file logging and optional TensorBoard support.
"""

from typing import Any
import logging
import os


def setup_logger(cfg: Any) -> Any:
    """
    Initialize and return a logger object.
    """
    log = logging.getLogger("dcal")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    outdir = cfg.get("logging", {}).get("output_dir", "./outputs")
    os.makedirs(outdir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(outdir, "train.log"))
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    return log


