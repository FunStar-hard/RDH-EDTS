"""Shared helpers for experiment scripts."""
from __future__ import annotations

import math
import secrets
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract
from src.scheme.types import SetupResult, SignResult, TimingBreakdown
from src.crypto.curve_utils import get_order, random_scalar, scalar_mult, point_sum, get_generator
from src.crypto.hash_prf import compute_challenge
from src.scheme.lagrange import all_lagrange_coefficients

#跑一次完整的“签名-验签-提取”
def single_trial(
    sr: SetupResult,#SetupResult 包含了预先生成的密钥和共享信息
    t: int,
    L: int,
    Nmax: int,#最大重试次数
) -> Dict[str, Any]:
    """Run one complete sign-verify-extract trial, return statistics dict."""
    m = secrets.token_bytes(32)#随机生成一个 32 字节的消息
    M = secrets.randbelow(2 ** L)#随机生成一个 L 位的整数作为要嵌入的信息
    participants = list(range(1, t + 1))#选择前t个节点参与签名

    t0 = time.perf_counter()
    result = sign_emb( #开始调用方案的签名函数，传入消息、参与者列表、共享信息、公共密钥、提取密钥、嵌入信息长度和最大重试次数
        m=m,
        participants=participants,#参与签名的节点列表，这里是从1到t的整数列表
        shares=sr.shares, #每个参与者的份额信息，来自预先的SetupResult
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=M,
        L=L,
        Nmax=Nmax,
    )
    sign_time = time.perf_counter() - t0#计算签名过程的耗时
    #构造结果字典，包含签名是否成功、重试次数、签名时间，以及后续的验签和提取结果（初始为 False）
    row: Dict[str, Any] = {
        "success": result.success,#签名是否成功
        "retries": result.retries,#重试次数
        "sign_time": sign_time,#签名耗时
        "verify_ok": False,#验签是否成功
        "extract_ok": False,#提取是否成功
    }
    #如果签名生成成功，执行schnorr验证和提取，并记录结果
    if result.success and result.signature is not None:
        t1 = time.perf_counter()#t1记录验签开始时间，调用verify函数验证签名的有效性，传入消息、公共密钥和签名对象，返回验签结果v
        v = verify(m, sr.pk, result.signature)#调用src/scheme/verify.py的verify函数验证签名，传入消息、公共密钥和签名对象，返回验签结果v
        verify_time = time.perf_counter() - t1
        row["verify_ok"] = v
        row["verify_time"] = verify_time

        t2 = time.perf_counter()#t2记录提取开始时间，调用extract函数提取嵌入的信息，传入消息、公共密钥、签名对象、提取密钥和嵌入信息长度，返回提取结果recovered
        recovered = extract(m, sr.pk, result.signature, sr.Kext, L)
        extract_time = time.perf_counter() - t2
        row["extract_ok"] = (recovered == M)
        row["extract_time"] = extract_time
        row["s_low_bits"] = result.signature.s % (2 ** L)#记录签名对象中 s 字段的低 L 位，这些位应该与嵌入的信息 M 相关联
    else:
        row["verify_time"] = 0.0
        row["extract_time"] = 0.0

    return row

#完整认证流程分模块计时，分别计时并统计内存峰值。
def timed_full_auth(
    sr: SetupResult,#SetupResult 包含了预先生成的密钥和共享信息
    t: int,
    L: int,
    Nmax: int,
) -> TimingBreakdown:#TimingBreakdown 包含了各个阶段的耗时统计和内存峰值
    """Run one full auth and return a detailed timing breakdown."""
    m = secrets.token_bytes(32)
    M = secrets.randbelow(2 ** L)
    participants = list(range(1, t + 1))
    q = get_order()
    G = get_generator()
    #预先计算拉格朗日系数，避免在每次重试时重复计算
    tb = TimingBreakdown()
    tracemalloc.start()

    
    lambdas = all_lagrange_coefficients(participants, q)

    total_start = time.perf_counter()#记录整个认证流程的开始时间，后续会计算总耗时
    retries = 0
    success = False
    sig = None

    for attempt in range(1, Nmax + 1):#最大重试次数 Nmax，模拟中每次重试都重新生成随机数和承诺，直到成功或达到 Nmax
        retries = attempt

        # ComGen 承诺生成
        t0 = time.perf_counter()#t0记录承诺生成开始时间，每个参与者生成随机数ki，并计算承诺R_i = k_i * G，最后计算所有参与者承诺的和 R 作为最终的承诺点
        nonces = {}
        comms = {}
        for vi in participants:
            ki = random_scalar()#生成新鲜随机数α_i ∈ [1, q-1]；random_scalar() → 调用src/crypto/curve_utils.py
            nonces[vi] = ki
            comms[vi] = scalar_mult(ki, G)# R_i = k_i * G；scalar_mult(ki, G) 
        R = point_sum([comms[vi] for vi in participants])#计算 R 点作为所有参与者承诺的和
        tb.com_gen += time.perf_counter() - t0#记录承诺生成的耗时

        c = compute_challenge(R, m, sr.pk)#调用src/crypto/hash_prf.py的compute_challenge函数计算挑战 c = H(R, m, pk)，其中 R 是聚合后的承诺点，m 是消息，pk 是系统公共密钥

        from src.crypto.hash_prf import compute_context, prf_l #计算上下文并使用 Kext 生成伪随机函数输出，得到 C = M XOR PRF(Kext, context)，其中 context 包含了签名相关的信息（如公共密钥、R 和消息），C 是一个 L 位的值，用于后续的验证和提取
        ctx = compute_context(sr.pk, R, m)
        prf_val = prf_l(sr.Kext, ctx, L)
        C = M ^ prf_val

        # PartSign 部分签名
        t0 = time.perf_counter()#t0记录部分签名开始时间，每个参与者计算自己的部分签名 s_i = k_i + c * λ_i * x_i mod q，其中 k_i 是随机数，λ_i 是拉格朗日系数，x_i 是参与者的私钥份额。将所有部分签名存储在字典 partials 中
        partials = {}
        for vi in participants:
            si = (nonces[vi] + c * lambdas[vi] * sr.shares[vi]) % q# 计算部分签名 s_i = k_i + c * λ_i * x_i mod q，其中 k_i 是随机数，λ_i 是拉格朗日系数，x_i 是参与者的私钥份额。将所有部分签名存储在字典 partials 中
            partials[vi] = si#记录部分签名的耗时
        tb.part_sign += time.perf_counter() - t0#

        # ShareVer 份额验证：每个参与者的部分签名 si 应满足 si*G = comms[vi] + c * lambda[vi] * share_pk[vi]，如果任何一个验证失败，则拒绝本次签名尝试并重试
        t0 = time.perf_counter()#t0记录份额验证开始时间，遍历每个参与者，计算左侧 LHS = s_i * G 和右侧 RHS = R_i + (c·λ_i)·pk_i，如果任何一个验证失败，则设置 ok = False 并跳出循环，最后记录份额验证的耗时
        ok = True
        for vi in participants:
            lhs = scalar_mult(partials[vi], G)
            coeff = (c * lambdas[vi]) % q
            rhs = comms[vi] + scalar_mult(coeff, sr.share_pks[vi])
            if lhs != rhs:
                ok = False
                break
        tb.share_ver += time.perf_counter() - t0

        if not ok:
            continue

        # Agg 聚合：将所有部分签名 si 聚合成最终签名 s = sum(si) mod q，并检查 s 的低 L 位是否等于 C。如果不匹配，则拒绝本次签名尝试并重试
        t0 = time.perf_counter()#t0记录聚合开始时间，将所有部分签名 si 聚合成最终签名 s = sum(si) mod q，并检查 s 的低 L 位是否等于 C。如果不匹配，则拒绝本次签名尝试并重试
        s = sum(partials[vi] for vi in participants) % q
        tb.agg += time.perf_counter() - t0

        if s % (2 ** L) == C:
            from src.scheme.types import Signature
            sig = Signature(R=R, s=s)
            success = True
            break

    total_sign = time.perf_counter() - total_start

    # 验签和提取
    if success and sig is not None:
        t0 = time.perf_counter()#t0记录验签开始时间，调用verify函数验证签名的有效性，传入消息、公共密钥和签名对象，返回验签结果v，并记录验签的耗时
        verify(m, sr.pk, sig)
        tb.verify = time.perf_counter() - t0

        t0 = time.perf_counter()#t0记录提取开始时间，调用extract函数提取嵌入的信息，传入消息、公共密钥、签名对象、提取密钥和嵌入信息长度，返回提取结果recovered，并记录提取的耗时
        extract(m, sr.pk, sig, sr.Kext, L)
        tb.extract = time.perf_counter() - t0

    tb.full_auth = time.perf_counter() - total_start
    tb.retries = retries

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tb.mem_peak_bytes = peak

    return tb

#计算几何分布的条件期望 E[X | X <= Nmax]，其中 X ~ Geom(p)，支持在 {1, 2, ...} 上。
def geometric_conditional_expectation(p: float, Nmax: int) -> float:
    """E[X | X <= Nmax] for X ~ Geom(p) supported on {1,2,...}.

    E[X|X<=Nmax] = sum_{k=1}^{Nmax} k * p*(1-p)^{k-1} / P(X<=Nmax)
    """
    if p <= 0 or p > 1:
        return float("nan")
    qc = 1 - p
    prob_le = 1 - qc ** Nmax  # P(X <= Nmax)
    if prob_le == 0:
        return float("nan")

    # sum k * p * qc^{k-1} for k=1..Nmax
    # = p * d/dqc [ sum qc^k for k=1..Nmax ] evaluated differently
    # Direct stable computation:
    numerator = 0.0
    for k in range(1, min(Nmax + 1, 10000)):
        term = k * p * (qc ** (k - 1))
        numerator += term
        if term < 1e-15 * numerator and k > 10:
            break

    # If Nmax > 10000 and we haven't converged, use closed form
    # E[X] for truncated geometric: (1 - (Nmax+1)*qc^Nmax + Nmax*qc^{Nmax+1}) / (p*(1-qc^Nmax))
    # but let's use the more numerically stable closed form always:
    if Nmax <= 10000:
        return numerator / prob_le

    num_closed = 1.0 - (Nmax + 1) * qc ** Nmax + Nmax * qc ** (Nmax + 1)
    den_closed = p * (1 - qc ** Nmax)
    if den_closed == 0:
        return float("nan")
    return num_closed / den_closed