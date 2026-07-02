# src/test_verify_extract.py 完整修复版
"""
专门测试论文3.4节 Verify（公共验证）和3.5节 Extract（授权提取）算法
修复版：解决合法签名验证失败的bug
在src目录下直接运行: python test_verify_extract.py
"""
import sys
import os
# 修复模块导入路径：自动添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import secrets
from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb
from src.scheme.types import Signature
from src.crypto.curve_utils import (
    get_order, get_generator, random_scalar,
    scalar_mult, point_sum
)
from src.crypto.hash_prf import compute_challenge, compute_context, prf_l
from src.scheme.lagrange import all_lagrange_coefficients

def test_verify_extract_step_by_step():
    print("=" * 90)
    print("  SDDH-TSS 3.4+3.5节 Verify + Extract 算法分步测试")
    print("=" * 90)
    # ====================== 测试参数设置 ======================
    print("\n📌 基础参数配置")
    n = 5       # 总节点数
    t = 3       # 门限值
    L = 2       # 嵌入比特长度
    Nmax = 256  # 最大重试次数
    print(f"  总节点数 n = {n}")
    print(f"  门限值 t = {t}")
    print(f"  嵌入比特长度 L = {L}")
    print(f"  最大重试次数 Nmax = {Nmax}")
    # ====================== 步骤0：系统初始化（对应3.2节） ======================
    print("\n" + "-" * 70)
    print("步骤0：运行Setup算法初始化系统")
    print("-" * 70)
    sr = setup(t, n, L)
    q = get_order()
    G = get_generator()
    print(f"  ✅ 系统公钥 pk: ({sr.pk.x()}, {sr.pk.y()})")
    print(f"  ✅ 提取密钥 Kext: {sr.Kext.hex()[:32]}...")
    print(f"  ✅ 椭圆曲线群阶 q: {q}")
    # ====================== 步骤1：生成合法测试签名（修复版） ======================
    print("\n" + "-" * 70)
    print("步骤1：生成满足嵌入条件的合法签名（调用sign_emb）")
    print("-" * 70)
    
    # 1.1 准备测试数据
    m = secrets.token_bytes(32)
    M = secrets.randbelow(2 ** L)
    participants = list(range(1, t + 1))
    print(f"  待签名消息 m (前16字节): {m.hex()[:32]}...")
    print(f"  待嵌入原始信息 M: 十进制={M}, 二进制={bin(M)[2:].zfill(L)}")
    print(f"  参与签名节点: {participants}")

    # 1.2 调用正确的sign_emb生成合法签名（核心修复）
    print("\n  正在调用sign_emb生成合法签名...")
    result = sign_emb(
        m=m,
        participants=participants,
        shares=sr.shares,
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=M,
        L=L,
        Nmax=Nmax,
        verify_partial=True
    )
    assert result.success, "❌ 签名生成失败！请检查Nmax是否足够"
    valid_sig = result.signature
    print(f"  ✅ 签名生成成功！共重试 {result.retries} 次（理论值: {2**L}）")
    print(f"  签名 R: ({valid_sig.R.x()}, {valid_sig.R.y()})")
    print(f"  签名 s: {valid_sig.s}")
    print(f"  s 的低{L}位: {bin(valid_sig.s % (2**L))[2:].zfill(L)} (十进制 {valid_sig.s % (2**L)})")

    # 1.3 预计算所有中间值用于分步验证
    lambdas = all_lagrange_coefficients(participants, q)
    ctx = compute_context(sr.pk, valid_sig.R, m)
    prf_val = prf_l(sr.Kext, ctx, L)
    C = M ^ prf_val
    print(f"\n  拉格朗日系数: {lambdas}")
    print(f"  提取上下文 ctx (前16字节): {ctx.hex()[:32]}...")
    print(f"  PRF输出值: {prf_val}")
    print(f"  目标嵌入值 c_h: 十进制={C}, 二进制={bin(C)[2:].zfill(L)}")
    print(f"  ✅ 签名s的低{L}位与目标值一致: {valid_sig.s % (2**L) == C}")

    # ====================== 步骤2：Verify算法完整测试（3.4节） ======================
    print("\n" + "-" * 70)
    print("步骤2：3.4节 Public Verification（公共验证）算法测试")
    print("-" * 70)
    
    print("\n📌 步骤2.1：重新计算挑战值c（公式17）")
    print("  公式: c = H(R || m || pk) mod q")
    c_verify = compute_challenge(valid_sig.R, m, sr.pk)
    print(f"  计算结果: c = {c_verify}")

    print("\n📌 步骤2.2：验证Schnorr核心等式（公式18）")
    print("  公式: s·G == R + c·pk")
    lhs = scalar_mult(valid_sig.s, G)
    rhs = valid_sig.R + scalar_mult(c_verify, sr.pk)
    print(f"  左式 s·G: ({lhs.x()}, {lhs.y()})")
    print(f"  右式 R + c·pk: ({rhs.x()}, {rhs.y()})")
    
    verify_result = (lhs == rhs)
    print(f"\n  ✅ 签名验证结果: {verify_result}")
    assert verify_result, "❌ 合法签名验证失败！"

    # ====================== 步骤3：Extract算法完整测试（3.5节） ======================
    print("\n" + "-" * 70)
    print("步骤3：3.5节 Authorized Information Extraction（授权提取）算法测试")
    print("-" * 70)
    
    print("\n📌 步骤3.1：先验证签名有效性（论文要求）")
    print("  提取前必须先验证签名，验证失败直接返回⊥")
    # 这里复用步骤2的验证结果

    print("\n📌 步骤3.2：计算提取上下文ctx（公式19）")
    print("  公式: ctx = H(pk || R || m)")
    ctx_extract = compute_context(sr.pk, valid_sig.R, m)
    print(f"  计算结果: ctx = {ctx_extract.hex()[:32]}...")
    assert ctx_extract == ctx, "❌ 上下文计算不一致！"

    print("\n📌 步骤3.3：提取s的低L位得到c_h（公式20）")
    print("  公式: c_h = s mod 2^l")
    s_low = valid_sig.s % (2**L)
    print(f"  提取结果: c_h = {s_low} (二进制={bin(s_low)[2:].zfill(L)})")
    assert s_low == C, "❌ 低L位提取错误！"

    print("\n📌 步骤3.4：解掩码恢复原始信息M（公式21）")
    print("  公式: m_h = c_h ⊕ PRF_l(k_ext, ctx)")
    prf_val_extract = prf_l(sr.Kext, ctx_extract, L)
    recovered_M = s_low ^ prf_val_extract
    print(f"  PRF值: {prf_val_extract}")
    print(f"  解掩码计算: {s_low} ⊕ {prf_val_extract} = {recovered_M}")
    print(f"  原始嵌入信息: {M}")
    print(f"  提取恢复信息: {recovered_M}")
    
    extract_success = (recovered_M == M)
    print(f"\n  ✅ 信息提取结果: {extract_success}")
    assert extract_success, "❌ 信息提取失败！"


if __name__ == "__main__":
    try:
        test_verify_extract_step_by_step()
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)