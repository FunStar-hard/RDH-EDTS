"""Matplotlib convenience wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, path: Path, also_pdf: bool = True) -> None:
    """Save figure as PNG (and optionally PDF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)