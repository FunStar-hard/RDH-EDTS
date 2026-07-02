"""Embedded information extraction."""
from __future__ import annotations

from typing import Any, Optional

from src.crypto.hash_prf import compute_context, prf_l
from src.scheme.types import Signature
from src.scheme.verify import verify as _verify

# 提取嵌入信息 输入m、pk、sig、Kext、L
def extract(
    m: bytes,
    pk: Any,
    sig: Signature,
    Kext: bytes,
    L: int,
) -> Optional[int]:
    """Extract the L‑bit embedded plaintext from a valid signature.

    Returns ``None`` (⊥) when verification fails.
    """
    if not _verify(m, pk, sig):# 验证签名是否有效
        return None# 验证失败返回None
    #调用src/crypto/hash_prf.py的compute_context
    ctx = compute_context(pk, sig.R, m)# 计算上下文ctx = H(pk, R, m)
    C = sig.s % (2 ** L)# 提取C = s mod 2^L
    #调用src/crypto/hash_prf.py的prf_l
    prf_val = prf_l(Kext, ctx, L)# 计算伪随机函数值prf_val = PRF(Kext, ctx) mod 2^L
    return C ^ prf_val# 返回提取的明文 = C XOR prf_val,恢复M