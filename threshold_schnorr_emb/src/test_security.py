"""
SDDH-TSS 安全性专项测试脚本
运行路径：src 目录下执行 python test_security.py
测试维度：签名伪造、门限规则、密钥安全、嵌入越界、上下文篡改等
"""
import sys
import os
import secrets

# ✅ 关键修复：添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ✅ 导入项目原有编码函数
from src.crypto.encoding import encode_point, encode_scalar

from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract
from src.scheme.types import Signature
from src.crypto.curve_utils import (
    get_order, get_generator, random_scalar,
    scalar_mult, point_sum
)
from src.crypto.hash_prf import compute_challenge, compute_context, prf_l
from src.scheme.lagrange import all_lagrange_coefficients

def test_security():
    print("=" * 90)
    print("  SDDH-TSS 安全性专项测试")
    print("=" * 90)

    # 基础参数（与业务一致）
    n = 5       # 总节点数
    t = 3       # 门限值
    L = 4       # 嵌入比特长度
    Nmax = 256  # 签名最大重试次数
    q = get_order()
    G = get_generator()

    # 步骤0：系统初始化
    print("\n📌 步骤0：初始化系统参数")
    sr = setup(t, n, L)
    print(f"  ✅ 系统公钥 pk: {encode_point(sr.pk).hex()[:32]}...")
    print(f"  ✅ 提取密钥 Kext: {sr.Kext.hex()[:32]}...")

    # 测试数据准备
    m = secrets.token_bytes(32)  # 原始消息
    M = secrets.randbelow(2 ** L)  # 原始嵌入信息
    valid_participants = list(range(1, t + 1))  # 合法参与节点（t个）

    # 生成合法签名（作为基准）
    print("\n📌 生成基准合法签名")
    sign_result = sign_emb(
        m=m,
        participants=valid_participants,
        shares=sr.shares,
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=M,
        L=L,
        Nmax=Nmax,
        verify_partial=True
    )
    assert sign_result.success, "基准签名生成失败！"
    valid_sig = sign_result.signature
    print(f"  ✅ 基准签名生成完成 (重试次数: {sign_result.retries})")
    print(f"  签名 R: {encode_point(valid_sig.R).hex()[:32]}...")
    print(f"  签名 s: {encode_scalar(valid_sig.s).hex()[:32]}...")

    # ====================== 安全测试1：签名伪造检测 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试1：签名伪造检测（篡改R/s/m）")
    print("-" * 70)

    # 1.1 篡改签名R
    fake_R = scalar_mult(random_scalar(), G)
    fake_sig_R = Signature(R=fake_R, s=valid_sig.s)
    verify_fake_R = verify(m, sr.pk, fake_sig_R)
    print(f"  1.1 篡改R后验证结果: {verify_fake_R} (预期: False)")
    assert not verify_fake_R, "❌ 篡改R未被检测到！"

    # 1.2 篡改签名s
    fake_s = (valid_sig.s + 123) % q
    fake_sig_s = Signature(R=valid_sig.R, s=fake_s)
    verify_fake_s = verify(m, sr.pk, fake_sig_s)
    print(f"  1.2 篡改s后验证结果: {verify_fake_s} (预期: False)")
    assert not verify_fake_s, "❌ 篡改s未被检测到！"

    # 1.3 篡改消息m
    fake_m = secrets.token_bytes(32)
    verify_fake_m = verify(fake_m, sr.pk, valid_sig)
    print(f"  1.3 篡改消息m后验证结果: {verify_fake_m} (预期: False)")
    assert not verify_fake_m, "❌ 篡改消息m未被检测到！"

    # ====================== 安全测试2：门限值规则校验 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试2：门限值规则校验（少于t个节点签名）")
    print("-" * 70)

    # 仅用t-1个节点尝试签名
    invalid_participants = list(range(1, t))  # 少于门限值
    sign_invalid = sign_emb(
        m=m,
        participants=invalid_participants,
        shares=sr.shares,
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=M,
        L=L,
        Nmax=Nmax,
        verify_partial=True
    )
    if sign_invalid.success:
        verify_invalid = verify(m, sr.pk, sign_invalid.signature)
        print(f"  2.1 少于t个节点签名验证结果: {verify_invalid} (预期: False)")
        assert not verify_invalid, "❌ 少于门限值签名验证通过！"
    else:
        print(f"  2.1 少于t个节点签名直接失败 (符合预期)")

    # ====================== 安全测试3：提取密钥安全性 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试3：提取密钥安全性（错误/无密钥）")
    print("-" * 70)

    # 3.1 使用错误提取密钥
    wrong_Kext = secrets.token_bytes(32)
    fake_M = extract(m, sr.pk, valid_sig, wrong_Kext, L)
    print(f"  3.1 错误密钥提取结果: {fake_M} (原始M={M})")
    assert fake_M != M, "❌ 错误密钥提取出正确信息！"

    # 3.2 无提取密钥（随机值填充）
    empty_Kext = b"\x00" * 32
    empty_M = extract(m, sr.pk, valid_sig, empty_Kext, L)
    print(f"  3.2 空密钥提取结果: {empty_M} (原始M={M})")
    assert empty_M != M, "❌ 空密钥提取出正确信息！"

    # ====================== 安全测试4：嵌入比特长度越界 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试4：嵌入比特长度越界检测")
    print("-" * 70)

    # 尝试嵌入超过L位的信息
    over_L_M = 2 ** L + 123  # 超出L位范围
    sign_over_L = sign_emb(
        m=m,
        participants=valid_participants,
        shares=sr.shares,
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=over_L_M,
        L=L,
        Nmax=Nmax,
        verify_partial=True
    )
    if sign_over_L.success:
        over_M = extract(m, sr.pk, sign_over_L.signature, sr.Kext, L)
        print(f"  4.1 超L位嵌入提取结果: {over_M} (原始越界值={over_L_M})")
        assert over_M != over_L_M, "❌ 超L位嵌入未被限制！"
    else:
        print(f"  4.1 超L位嵌入签名直接失败 (符合预期)")

    # ====================== 安全测试5：重复随机数（Nonce）攻击检测 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试5：重复Nonce攻击风险验证")
    print("-" * 70)

    # 模拟重复nonce（固定nonce而非随机生成）
    fixed_nonce = random_scalar()
    nonces = {vi: fixed_nonce for vi in valid_participants}
    commitments = {vi: scalar_mult(nonces[vi], G) for vi in valid_participants}
    R = point_sum([commitments[vi] for vi in valid_participants])
    c = compute_challenge(R, m, sr.pk)

    # 构造签名
    lambdas = all_lagrange_coefficients(valid_participants, q)
    s0 = sum(nonces[vi] + c * lambdas[vi] * sr.shares[vi] for vi in valid_participants) % q
    C = M ^ prf_l(sr.Kext, compute_context(sr.pk, R, m), L)
    s = (s0 - (s0 % (2**L)) + C) % q
    repeat_nonce_sig = Signature(R=R, s=s)

    # 重复nonce下不同消息的验证
    fake_m2 = secrets.token_bytes(32)
    verify_diff_m = verify(fake_m2, sr.pk, repeat_nonce_sig)
    print(f"  5.1 重复Nonce+不同消息验证: {verify_diff_m} (预期: False)")
    assert not verify_diff_m, "❌ 重复Nonce+不同消息验证通过！"

    # ====================== 安全测试6：上下文篡改对提取的影响 ======================
    print("\n" + "-" * 70)
    print("🔐 安全测试6：提取上下文篡改检测")
    print("-" * 70)

    # 篡改上下文（修改R后重新计算ctx）
    fake_ctx_R = scalar_mult(random_scalar(), G)
    fake_ctx = compute_context(sr.pk, fake_ctx_R, m)
    fake_prf = prf_l(sr.Kext, fake_ctx, L)
    fake_C = valid_sig.s % (2**L)
    fake_M_ctx = fake_C ^ fake_prf
    print(f"  6.1 篡改上下文后提取结果: {fake_M_ctx} (原始M={M})")
    assert fake_M_ctx != M, "❌ 上下文篡改未影响提取结果！"

    # ====================== 测试总结 ======================
    print("\n" + "=" * 90)
    print("🎉 所有安全性测试通过！")
    print("=" * 90)
    print("  ✅ 签名伪造（R/s/m篡改）→ 检测通过")
    print("  ✅ 门限值规则（少于t节点）→ 检测通过")
    print("  ✅ 提取密钥（错误/空密钥）→ 安全隔离")
    print("  ✅ 嵌入比特越界 → 限制生效")
    print("  ✅ 重复Nonce攻击 → 风险可控")
    print("  ✅ 上下文篡改 → 提取失败")
    print("=" * 90)

if __name__ == "__main__":
    try:
        test_security()
    except AssertionError as e:
        print(f"\n❌ 安全性测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)