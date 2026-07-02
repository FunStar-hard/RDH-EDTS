"""Schnorr signature verification."""
from __future__ import annotations

from typing import Any

from src.crypto.curve_utils import get_generator, scalar_mult
from src.crypto.hash_prf import compute_challenge
from src.scheme.types import Signature

# 验证签名 输入m、pk、sig
def verify(m: bytes, pk: Any, sig: Signature) -> bool:
    """Verify a Schnorr signature Σ = (R, s).

    Check:  s·G  ==  R + c·pk   where  c = H(R, m, pk) mod q.
    """
    G = get_generator()#调用src/crypto/curve_utils.py  get_generator()，返回 NIST P-256 的生成元
    #调用src/crypto/hash_prf.py的compute_challeng
    c = compute_challenge(sig.R, m, pk) # 计算挑战值c = H(R, m, pk) mod q
    #调用src/crypto/curve_utils.py的scalar_mult，执行椭圆曲线点乘运算
    lhs = scalar_mult(sig.s, G) # 计算左边s·G
    rhs = sig.R + scalar_mult(c, pk) # 计算右边R + c·pk
    return lhs == rhs # 验证签名是否有效