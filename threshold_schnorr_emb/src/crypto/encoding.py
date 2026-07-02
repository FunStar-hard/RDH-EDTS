"""Deterministic binary encoding of EC points, scalars, and messages.

Point encoding follows the SEC 1 *compressed* format (33 bytes for P‑256).
"""
from __future__ import annotations

import math
from typing import Any

from src.crypto.curve_utils import get_order

_COORD_LEN = 32  # bytes for a 256‑bit field element


def encode_point(P: Any) -> bytes:
    """SEC 1 compressed encoding:  ``0x02|0x03 || x``  (33 bytes)."""
    x: int = P.x()
    y: int = P.y()
    prefix = b"\x02" if y % 2 == 0 else b"\x03"#生成1字节的前缀：y是偶数用0x02，奇数用0x03
    return prefix + x.to_bytes(_COORD_LEN, "big")#拼接：1字节前缀 + 32字节x坐标 → 总共33字节


def encode_scalar(s: int) -> bytes:
    return s.to_bytes(_COORD_LEN, "big")


def encode_message(m: bytes) -> bytes:
    """Messages are passed through as‑is."""
    return m


# ── Bit / int / bytes conversions ───────────────────────────────────────

def int_to_bits(val: int, length: int) -> str:
    """Convert non‑negative *val* to a binary string of exactly *length* bits."""
    return format(val, f"0{length}b")


def bits_to_int(bits: str) -> int:
    return int(bits, 2)


def int_to_bytes_l(val: int, L: int) -> bytes:
    """Return the minimal byte‑string that carries *L* bits."""
    n_bytes = math.ceil(L / 8)
    return val.to_bytes(n_bytes, "big")


def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, "big")