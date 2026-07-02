"""I/O helpers – directory creation, CSV / JSON persistence."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


def make_output_dir(base: str = "outputs", tag: str | None = None) -> Path:
    """Create and return a time‑stamped output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    out = Path(base) / name
    for sub in ("logs", "raw", "tables", "figures", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def save_rows_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Save a list of dicts as a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: Any, path: Path) -> None:
    """Save an object as pretty‑printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)