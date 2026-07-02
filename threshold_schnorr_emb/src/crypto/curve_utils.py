"""Elliptic curve (NIST P‑256) point‑arithmetic wrapper.

All heavy‑lifting is delegated to the ``ecdsa`` library; this module
provides a thin, project‑wide API so that no other file needs to import
``ecdsa`` directly.
"""
from __future__ import annotations

import secrets
from typing import Any, List

from ecdsa import NIST256p
from ecdsa.ellipticcurve import INFINITY, PointJacobi  # noqa: F401

# ── Module‑level curve constants ────────────────────────────────────────
_CURVE = NIST256p
GENERATOR: PointJacobi = _CURVE.generator
ORDER: int = _CURVE.order        # q  – prime order of the base‑point subgroup
FIELD_P: int = _CURVE.curve.p()  # field prime


# ── Convenience helpers ─────────────────────────────────────────────────
#定义了几个函数来获取生成元、阶数，以及执行标量乘法和点加法等操作。这些函数都是对底层椭圆曲线库的封装，使得其他部分的代码可以更方便地使用这些基本操作，而不需要直接与底层库交互。
def get_generator() -> PointJacobi:
    return GENERATOR# 返回椭圆曲线的生成元G（对应论文中的G）


def get_order() -> int:
    return ORDER# 返回椭圆曲线群的阶q（对应论文中的q）


def scalar_mult(k: int, P: Any | None = None) -> Any:#定义了一个函数scalar_mult，用于执行标量乘法操作，即计算k·P，其中k是一个整数，P是一个椭圆曲线上的点。如果P为None，则默认使用生成元G。
    """Return *k*·*P*  (defaults to *k*·*G* when *P* is ``None``)."""
    if P is None:
        P = GENERATOR
    return (k % ORDER) * P   # 先对k取模q，再执行椭圆曲线点乘运算


def point_add(P: Any, Q: Any) -> Any:
    return P + Q# 椭圆曲线点加法，直接使用底层库提供的加法运算符重载


def point_sum(points: List[Any]) -> Any:
    """Add a non‑empty list of EC points."""
    if not points:
        return INFINITY
    result = points[0]
    for p in points[1:]:
        result = result + p# 椭圆曲线点加法
    return result


def random_scalar() -> int:
    """Return a uniformly random element of  Z_q^*  ( [1, q‑1] )."""
    return secrets.randbelow(ORDER - 1) + 1# 生成一个随机的标量，范围在1到q-1之间，确保它是Z_q^*中的元素


def is_infinity(P: Any) -> bool:# 判断一个点是否是无穷远点（椭圆曲线中的特殊点，表示加法的单位元）
    try:
        return P == INFINITY
    except Exception:
        return False