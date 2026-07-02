"""Data‑class definitions shared across the scheme modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Signature:
    """Schnorr signature (R, s)."""
    R: Any            # EC point
    s: int            # scalar


@dataclass
class SetupResult:
    """Output of the ``setup`` algorithm."""
    pk: Any                       # system public key (EC point)
    shares: Dict[int, int]       # v_i → x_i
    share_pks: Dict[int, Any]   # v_i → pk_i (EC point)
    Kext: bytes                  # extraction key
    n: int
    t: int
    L: int
    system_secret: int = 0       # kept only for test verification


@dataclass
class SignResult:
    """Output of ``sign_emb``."""
    signature: Optional[Signature] = None
    retries: int = 0
    success: bool = False


@dataclass
class TimingBreakdown:
    """Detailed timing for one full authentication cycle."""
    setup: float = 0.0
    com_gen: float = 0.0
    part_sign: float = 0.0
    share_ver: float = 0.0
    agg: float = 0.0
    verify: float = 0.0
    extract: float = 0.0
    full_auth: float = 0.0
    retries: int = 0
    mem_peak_bytes: int = 0