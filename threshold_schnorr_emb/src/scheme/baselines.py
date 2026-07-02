"""Baseline / comparison schemes.

1. Per-node Authentication     - each node signs independently
2. Standard Threshold Schnorr  - no embedding
3. Threshold Schnorr + Extra   - standard sig + separate embedded field
"""
from __future__ import annotations

import math
import secrets
from typing import Any, Dict, List, Optional, Tuple

from src.crypto.curve_utils import (
    get_generator,
    get_order,
    point_sum,
    random_scalar,
    scalar_mult,
)
from src.crypto.hash_prf import compute_challenge, compute_context, prf_l
from src.scheme.lagrange import all_lagrange_coefficients
from src.scheme.types import Signature


# ====================================================================
# 1. Per-node Authentication
# ====================================================================
#逐节点签名：每个节点独立使用标准 Schnorr 签名对消息 m 进行签名，生成 n 个独立的签名。
def per_node_sign(
    m: bytes,
    node_keys: List[Tuple[int, Any]],#每个节点的密钥对列表，包含私钥和公钥
) -> List[Signature]:
    """Each node signs m independently with standard Schnorr."""
    q = get_order()
    G = get_generator()
    sigs: List[Signature] = []#对于每个节点，生成一个随机的 Schnorr 签名，包含一个承诺 R 和一个响应 s，返回所有节点的签名列表
    for xi, pki in node_keys:
        ki = random_scalar()
        Ri = scalar_mult(ki, G)
        ci = compute_challenge(Ri, m, pki)
        si = (ki + ci * xi) % q
        sigs.append(Signature(R=Ri, s=si))#生成一个随机的 Schnorr 签名，包含一个承诺 R 和一个响应 s，返回所有节点的签名列表
    return sigs

# 验证每个节点的签名
def per_node_verify(
    m: bytes,
    node_pks: List[Any],
    sigs: List[Signature],
) -> bool:
    """Verify each per-node signature."""
    G = get_generator()
    for pki, sig in zip(node_pks, sigs):#对于每个节点的公钥和签名，计算挑战 c，并验证 Schnorr 签名的正确性，如果有任何一个签名验证失败，则返回 False；如果所有签名都验证成功，则返回 True
        c = compute_challenge(sig.R, m, pki)
        lhs = scalar_mult(sig.s, G)
        rhs = sig.R + scalar_mult(c, pki)
        if lhs != rhs:
            return False
    return True

#逐节点签名的通信成本：每个节点发送一个 Schnorr 签名（33 字节压缩点 + 32 字节标量），总共 n 个节点。
def per_node_comm_cost(n: int, L: int) -> int:
    """Communication cost in bytes: n signatures, each (33+32) bytes."""
    sig_size = 33 + 32  # compressed point + scalar
    return n * sig_size


# ====================================================================
# 2. Standard Threshold Schnorr (no embedding)
# ====================================================================
# 标准门限 Schnorr：t 个参与者合作生成一个单一的 Schnorr 签名，满足 t 个签名份额的门限要求。
def threshold_schnorr_sign(
    m: bytes,
    participants: List[int],
    shares: Dict[int, int],
    share_pks: Dict[int, Any],
    pk: Any,
) -> Signature:
    """One-round threshold Schnorr (no rejection sampling)."""
    q = get_order()
    G = get_generator()
    S = participants
    lambdas = all_lagrange_coefficients(S, q)
    #生成随机数和承诺，计算挑战，并生成部分签名，最后聚合成一个单一的 Schnorr 签名返回
    nonces: Dict[int, int] = {}
    comms: Dict[int, Any] = {}
    for vi in S:
        ki = random_scalar()
        nonces[vi] = ki
        comms[vi] = scalar_mult(ki, G)

    R = point_sum([comms[vi] for vi in S])
    c = compute_challenge(R, m, pk)
    #计算每个参与者的部分签名 s_i = k_i + c * λ_i * x_i mod q，其中 k_i 是随机数，λ_i 是拉格朗日系数，x_i 是参与者的私钥份额。将所有部分签名聚合成一个单一的 Schnorr 签名返回
    s = 0
    for vi in S:
        si = (nonces[vi] + c * lambdas[vi] * shares[vi]) % q
        s = (s + si) % q

    return Signature(R=R, s=s)

# 标准门限 Schnorr 的通信成本：t 个承诺（每个 33 字节压缩点）+ t 个部分签名（每个 32 字节标量）+ 1 个最终签名（33 字节压缩点 + 32 字节标量）。
def threshold_schnorr_comm_cost(t: int, L: int) -> int:
    """Communication: t commitments + t partials + 1 final sig."""
    commitments = t * 33
    partials = t * 32
    final_sig = 33 + 32
    return commitments + partials + final_sig


# ====================================================================
# 3. Threshold Schnorr + Extra Field
# ====================================================================
# 在标准门限 Schnorr 的基础上，附加一个单独的 L 位嵌入字段。签名过程与标准门限 Schnorr 相同，但在签名之外传输一个额外的 L 位字段，该字段通过 Kext 和上下文计算得到。
def threshold_schnorr_extra_sign(
    m: bytes,
    participants: List[int],
    shares: Dict[int, int],
    share_pks: Dict[int, Any],
    pk: Any,
    Kext: bytes,
    M: int,
    L: int,
) -> Tuple[Signature, int]:
    """Standard threshold Schnorr + appended embedded field.

    Returns (signature, embedded_field).
    The embedded_field is transmitted separately alongside the signature.
    """
    sig = threshold_schnorr_sign(m, participants, shares, share_pks, pk)
    ctx = compute_context(pk, sig.R, m)
    prf_val = prf_l(Kext, ctx, L)
    embedded_field = M ^ prf_val#嵌入信息作为独立的 L 位字段传输，计算方式为 M XOR PRF(Kext, context)，其中 context 包含了签名相关的信息（如公共密钥、R 和消息），M 是一个 L 位的值，用于后续的验证和提取
    return sig, embedded_field#返回标准门限 Schnorr 签名和独立的嵌入字段

# 从门限 Schnorr + 额外字段的签名中提取嵌入信息，首先验证标准 Schnorr 签名的正确性，如果验证成功，则使用 Kext 和上下文计算 PRF 值，并通过 XOR 操作恢复原始的嵌入信息 M。
def threshold_schnorr_extra_extract(
    m: bytes,
    pk: Any,
    sig: Signature,
    embedded_field: int,
    Kext: bytes,
    L: int,
) -> Optional[int]:
    """Extract from the extra-field variant."""
    from src.scheme.verify import verify
    if not verify(m, pk, sig):
        return None
    ctx = compute_context(pk, sig.R, m)
    prf_val = prf_l(Kext, ctx, L)
    return embedded_field ^ prf_val

# 本方案的通信成本：与标准门限 Schnorr 相同（t 个承诺 + t 个部分签名 + 1 个最终签名），外加一个单独的 L 位字段（即 L/8 字节）。
def threshold_schnorr_extra_comm_cost(t: int, L: int) -> int:
    """Communication: same as standard + extra L-bit field."""
    base = threshold_schnorr_comm_cost(t, L)
    extra_bytes = math.ceil(L / 8)
    return base + extra_bytes


# ====================================================================
# 4. Embedded Threshold Schnorr (our scheme)
# ====================================================================
# 在门限 Schnorr 签名的 R 字段中嵌入 L 位信息。签名生成过程与标准门限 Schnorr 相同，但在计算 R 时，将 L 位嵌入信息与原始 R 进行组合（例如，通过将 R 的 x 坐标的低 L 位替换为嵌入信息）。验证时，提取 R 中的嵌入信息并使用 Kext 和上下文进行解密。
def embedded_threshold_comm_cost(t: int, L: int) -> int:
    """Communication: same as standard threshold (embedded in sig itself)."""
    commitments = t * 33
    partials = t * 32
    final_sig = 33 + 32
    return commitments + partials + final_sig