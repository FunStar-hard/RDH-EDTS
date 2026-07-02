"""SHA‑256 based hash and HMAC‑SHA‑256 based PRF used throughout the scheme."""
from __future__ import annotations

import hashlib
import hmac
from typing import Any

from src.crypto.curve_utils import get_order
from src.crypto.encoding import encode_point, encode_message

# encode_point(P) → 调用src/crypto/encoding.py第 14-20 行，使用 SEC 1 压缩格式将椭圆曲线点编码为 33 字节
def compute_challenge(R: Any, m: bytes, pk: Any) -> int:
    """c = SHA‑256( encode(R) || m || encode(pk) )   mod q."""
    data = encode_point(R) + encode_message(m) + encode_point(pk) # 按顺序编码R、消息m、公钥pk
    h = hashlib.sha256(data).digest() # 计算SHA-256哈希
    return int.from_bytes(h, "big") % get_order() # 转换为整数并模q

#encode_message(m) → 直接返回消息字节串
def compute_context(pk: Any, R: Any, m: bytes) -> bytes:
    """ctx = SHA‑256( encode(pk) || encode(R) || m )."""
    data = encode_point(pk) + encode_point(R) + encode_message(m)
    return hashlib.sha256(data).digest()


def prf_l(Kext: bytes, ctx: bytes, L: int) -> int:
    """PRF_L  =  Trunc_L( HMAC‑SHA‑256(Kext, ctx) ).

    Returns an integer in [0, 2^L).
    """
    h = hmac.new(Kext, ctx, hashlib.sha256).digest()
    return int.from_bytes(h, "big") % (2 ** L)