"""SignEmb – threshold Schnorr signing with rejection‑sampling embedding."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.crypto.curve_utils import (
    get_generator,
    get_order,
    point_sum,
    random_scalar,
    scalar_mult,
)
from src.crypto.hash_prf import compute_challenge, compute_context, prf_l
from src.scheme.lagrange import all_lagrange_coefficients
from src.scheme.types import Signature, SignResult

# 运行 SignEmb 协议（本地模拟） 输入参数    m: 消息字节串
# participants: 签名集合 S 中的 v_i 标识符列表  shares: v_i → x_i 份额映射
# share_pks: v_i → pk_i 份额公钥映射 pk:
def sign_emb(
    m: bytes,
    participants: List[int],
    shares: Dict[int, int],
    share_pks: Dict[int, Any],
    pk: Any,
    Kext: bytes,
    M: int,
    L: int,
    Nmax: int,
    verify_partial: bool = True,
) -> SignResult:
    """Execute the SignEmb protocol (simulated locally).

    Parameters
    ----------
    m             : message bytes
    participants  : list of v_i identifiers in the signing set S
    shares        : v_i → x_i
    share_pks     : v_i → pk_i  (EC point)
    pk            : system public key
    Kext          : extraction key
    M             : plaintext to embed (L‑bit integer)
    L             : number of bits to embed
    Nmax          : maximum retry count
    verify_partial: whether to verify each partial signature

    Returns
    -------
    SignResult  (with .success=False if Nmax exceeded)
    """
    q = get_order() # 获取椭圆曲线群阶q
    G = get_generator() # 获取椭圆曲线生成元G
    S = participants # 参与节点集合S_np

    # 计算拉格朗日系数 λ_i = Π_{j∈S\{i}} v_j / (v_j - v_i) mod q
    lambdas = all_lagrange_coefficients(S, q)#all_lagrange_coefficients(S, q) → 调用src/scheme/lagrange.py 23-27行
    #最大重试次数 Nmax，模拟中每次重试都重新生成随机数和承诺，直到成功或达到 Nmax
    for attempt in range(1, Nmax + 1):
        #每个节点生成随机数ki
        nonces: Dict[int, int] = {}# 存储每个节点的随机数α_i（代码中变量名为ki）
        commitments: Dict[int, Any] = {}# 存储每个节点的承诺R_i
        for vi in S:
            ki = random_scalar()  # 生成新鲜随机数α_i ∈ [1, q-1]；random_scalar() → 调用src/crypto/curve_utils.py
            nonces[vi] = ki
            commitments[vi] = scalar_mult(ki, G) #节点承诺 R_i = k_i * G；scalar_mult(ki, G) → 同 3.2 节，执行椭圆曲线点乘运算

        # 聚合承诺  计算挑战 c = H(R, m, pk)，其中 R = Σ R_i
        R = point_sum([commitments[vi] for vi in S])#point_sum(points) → 调用src/crypto/curve_utils.py 43-50
        c = compute_challenge(R, m, pk) #compute_challenge(R, m, pk) → 调用src/crypto/hash_prf.py 12-16

        # ── (3) Context & embedding value ────────────────────────────
        ctx = compute_context(pk, R, m)# 计算上下文 ctx = H(pk, R, m) compute_context(pk, R, m) → 调用src/crypto/hash_prf.py 19-22
        prf_val = prf_l(Kext, ctx, L)# 计算伪随机值 prf_val = PRF(Kext, ctx) mod 2^L；prf_l(Kext, ctx, L) → 调用src/crypto/hash_prf.py 25-31
        C = M ^ prf_val# 计算嵌入值 C = M XOR prf_val，对应公式（11）

        # ── (4) 部分签名 ──────是预计算的拉格朗日系数 λ_i───shares[vi]是节点 vi 的私钥份额 sk_i
        partial: Dict[int, int] = {}
        for vi in S:
            si = (nonces[vi] + c * lambdas[vi] * shares[vi]) % q# 计算部分签名 s_i = k_i + c * λ_i * x_i mod q
            partial[vi] = si

        # ── (5) 份额校验 ───────────────────────
        if verify_partial:
            bad = False
            for vi in S:
                lhs = scalar_mult(partial[vi], G)# 计算左侧 LHS = s_i * G
                coeff = (c * lambdas[vi]) % q# 计算系数 coeff = c * λ_i mod q
                rhs = commitments[vi] + scalar_mult(coeff, share_pks[vi])# 计算右侧 RHS = R_i + coeff * pk_i，R_i + (c·λ_i)·pk_i
                if lhs != rhs:
                    bad = True
                    break
            if bad:
                continue  # 份额验证失败，重试

        # ── (6) 聚合响应 ────────────────────────────────────────────
        s = sum(partial[vi] for vi in S) % q# 计算聚合响应 s = Σ s_i mod q

        # ── (7) 拒绝采样 ───────────────────────────────────
        if s % (2 ** L) == C:# 检查低位，如果 s mod 2^L == C，则成功
            return SignResult(
                signature=Signature(R=R, s=s),# 返回签名 (R, s)
                retries=attempt,# 重试次数
                success=True,
            )

    # 超过最大重试次数，返回失败
    return SignResult(signature=None, retries=Nmax, success=False)