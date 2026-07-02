"""Configuration loading and validation utilities."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], dest: Path) -> None:
    """Write a YAML snapshot of the current configuration."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep‑copy of *base*."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def get_scheme_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scheme parameters with sensible defaults."""
    s = cfg.get("scheme", {})
    return {
        "curve": s.get("curve", "P-256"),
        "hash": s.get("hash", "SHA-256"),
        "prf": s.get("prf", "HMAC-SHA-256"),
        "kext_bits": s.get("kext_bits", 256),
        "n": s.get("n", 5),
        "t": s.get("t", 3),
        "L": s.get("L", 4),
        "Nmax": s.get("Nmax", 256),
    }