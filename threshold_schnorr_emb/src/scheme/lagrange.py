"""Lagrange interpolation coefficient computation over Z_q."""
from __future__ import annotations

from typing import Dict, List

from src.crypto.curve_utils import get_order

#计算拉格朗日系数
def lagrange_coefficient(vi: int, S: List[int], q: int | None = None) -> int:
    """Compute  λ_{vi,S} = Π_{vj∈S, vj≠vi}  vj / (vj − vi)   mod q."""
    if q is None:
        q = get_order()
    num = 1
    den = 1
    for vj in S:
        if vj == vi:
            continue
        num = (num * vj) % q
        den = (den * ((vj - vi) % q)) % q
    return (num * pow(den, -1, q)) % q

#该函数为每个参与节点预计算拉格朗日系数 λ_i（对应论文公式 12），由于参与节点集合在所有重试中不变，因此放在循环外一次计算完成
def all_lagrange_coefficients(S: List[int], q: int | None = None) -> Dict[int, int]:
    """Return ``{vi: λ_{vi,S}}`` for every *vi* in *S*."""
    if q is None:
        q = get_order()
    return {vi: lagrange_coefficient(vi, S, q) for vi in S}