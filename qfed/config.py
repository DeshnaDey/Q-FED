"""Config loading: configs/base.yaml merged with an experiment file and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_base() -> dict:
    return load_yaml(BASE_CONFIG)
