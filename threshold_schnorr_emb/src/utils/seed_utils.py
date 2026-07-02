"""Random seed management for reproducibility at the numpy / random layer."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int | None) -> None:
    """Set global seeds for numpy and stdlib random.

    Note: ``secrets`` module is *never* seeded – it always uses OS entropy.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)