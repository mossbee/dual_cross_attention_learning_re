"""
Config utilities.

Purpose:
- Load YAML configs for FGVC and Re-ID, merge with CLI overrides, and expose a
  simple namespace-like object.
"""

from typing import Any, Dict, Optional
import yaml


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML and apply optional overrides.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or {}
    # naive shallow merge
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


