"""Shamir (t, n) secret sharing over Z_q."""
from __future__ import annotations

import secrets
from typing import Dict, List, Tuple

from src.crypto.curve_utils import get_order

# 构造（t-1）次多项式
def generate_polynomial(t: int, secret: int, q: int) -> List[int]:
    """Random polynomial of degree *t−1* with f(0) = *secret*."""
    coeffs = [secret % q] # 常数项为系统私钥sk（对应公式4的第一项）
    for _ in range(t - 1):
        coeffs.append(secrets.randbelow(q)) # 生成t-1个随机系数a_1...a_{t-1}
    return coeffs

# 计算份额xi（ski）
def evaluate_polynomial(coeffs: List[int], x: int, q: int) -> int:
    """Horner evaluation of polynomial at *x* modulo *q*."""
    result = 0
    for a in reversed(coeffs): # 从最高次项到常数项
        result = (result * x + a) % q # 霍纳法则：f(x) = (...((a_{t-1}x) + a_{t-2})x + ...) + a_0
    return result


def share_secret( #share_secret函数返回：_coeffs：多项式系数；shares：字典{节点ID: 私钥份额sk_i}（对应论文的{sk_i}_{i=1}^n）
    secret: int, t: int, n: int, q: int | None = None
) -> Tuple[List[int], Dict[int, int]]:
    """Split *secret* into *n* Shamir shares with threshold *t*.

    Returns
    -------
    coeffs : list[int]
        Polynomial coefficients (for testing / debugging).
    shares : dict[int, int]
        Mapping  v_i → x_i = f(v_i)  for v_i ∈ {1, …, n}.
    """
    if q is None:
        q = get_order()
    coeffs = generate_polynomial(t, secret, q)   # secret就是系统私钥x，生成多项式系数，（对应公式 4）
    shares: Dict[int, int] = {}#计算每个节点的私钥份额 sk_i（公式 5）
    for i in range(1, n + 1):
        shares[i] = evaluate_polynomial(coeffs, i, q) #evaluate_polynomial函数
    return coeffs, shares

# 使用拉格朗日插值法在0点重构秘密
def reconstruct_secret(shares: Dict[int, int], q: int | None = None) -> int:
    """Reconstruct secret from a set of shares using Lagrange interpolation at 0."""
    if q is None:#如果没有指定模数q，默认使用椭圆曲线的阶作为模数
        q = get_order()
    from src.scheme.lagrange import lagrange_coefficient#调用src/scheme/lagrange.py的lagrange_coefficient函数，计算每个份额对应的拉格朗日系数λ_i
    S = list(shares.keys())#份额的索引列表S = {i_1, i_2, ..., i_t}，其中每个i_j对应一个节点ID
    secret = 0
    for vi in S:#对于每个份额vi，计算对应的拉格朗日系数lam = λ_i，并将份额值乘以拉格朗日系数累加到secret中，最后对q取模得到重构的秘密
        lam = lagrange_coefficient(vi, S, q)
        secret = (secret + lam * shares[vi]) % q
    return secret