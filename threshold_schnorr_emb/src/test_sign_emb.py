#!/usr/bin/env python3
"""
专门测试论文3.3节 Threshold Signing and Information Embedding 算法的小脚本
手动执行每一个步骤，打印所有中间结果，对应论文公式
"""
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scheme.setup import setup
from scheme.verify import verify
from scheme.extract import extract
from crypto.curve_utils import (
    get_order, get_generator, random_scalar, 
    scalar_mult, point_sum
)
from crypto.hash_prf import compute_challenge, compute_context, prf_l
from scheme.lagrange import all_lagrange_coefficients
from scheme.types import Signature

def test_sign_emb_step_by_step():
    print("=" * 80)
    print("  论文3.3节 SignEmb 算法分步测试")
    print("=" * 80)

    # ====================== 测试参数设置 ======================
    n = 5       # 总节点数
    t = 3       # 门限值
    L = 2       # 嵌入比特长度
    Nmax = 10   # 最大重试次数（L=2时平均4次成功）
    print(f"\n[参数设置] n={n}, t={t}, L={L}, Nmax={Nmax}")

    # ====================== 步骤0：系统初始化（对应3.2节） ======================
    print("\n" + "-" * 60)
    print("步骤0：运行Setup算法初始化系统")
    print("-" * 60)
    sr = setup(t, n, L)
    print(f"系统公钥 pk: {sr.pk}")
    print(f"提取密钥 Kext: {sr.Kext.hex()}")
    print(f"节点私钥份额: {sr.shares}")
    print(f"节点份额公钥: {sr.share_pks}")

    # ====================== 准备签名输入 ======================
    print("\n" + "-" * 60)
    print("准备签名输入")
    print("-" * 60)
    import secrets
    m = secrets.token_bytes(32)  # 随机生成32字节消息
    M = secrets.randbelow(2 ** L)  # 随机生成L位嵌入信息
    participants = list(range(1, t + 1))  # 选择前t个节点参与签名
    print(f"待签名消息 m (前16字节): {m.hex()[:32]}...")
    print(f"待嵌入信息 M: 二进制 {bin(M)}, 十进制 {M}")
    print(f"参与签名的节点: {participants}")

    # ====================== 步骤1：预计算拉格朗日系数（公式12） ======================
    print("\n" + "-" * 60)
    print("步骤1：预计算拉格朗日系数（公式12）")
    print("-" * 60)
    q = get_order()
    G = get_generator()
    lambdas = all_lagrange_coefficients(participants, q)
    for vi in participants:
        print(f"节点 {vi} 的拉格朗日系数 λ_{vi}: {lambdas[vi]}")

    # ====================== 步骤2：重试循环（拒绝采样核心） ======================
    print("\n" + "-" * 60)
    print("步骤2：进入签名重试循环（拒绝采样）")
    print("-" * 60)

    for attempt in range(1, Nmax + 1):
        print(f"\n>>> 第 {attempt} 次签名尝试 <<<")

        # ---------------------- 2.1 生成随机数与承诺（公式7） ----------------------
        print("\n2.1 每个节点生成随机数α_i和承诺R_i（公式7: R_i = α_i·G）")
        nonces = {}
        commitments = {}
        for vi in participants:
            ki = random_scalar()  # α_i
            nonces[vi] = ki
            commitments[vi] = scalar_mult(ki, G)  # R_i
            print(f"节点 {vi}: α_i={ki}, R_i={commitments[vi]}")

        # ---------------------- 2.2 聚合承诺R（公式8） ----------------------
        print("\n2.2 聚合所有承诺得到R（公式8: R = ΣR_i）")
        R = point_sum([commitments[vi] for vi in participants])
        print(f"聚合承诺 R: {R}")

        # ---------------------- 2.3 计算挑战值c（公式9） ----------------------
        print("\n2.3 计算Schnorr挑战值c（公式9: c = H(R||m||pk) mod q）")
        c = compute_challenge(R, m, sr.pk)
        print(f"挑战值 c: {c}")

        # ---------------------- 2.4 计算提取上下文ctx（公式10） ----------------------
        print("\n2.4 计算提取上下文ctx（公式10: ctx = H(pk||R||m)）")
        ctx = compute_context(sr.pk, R, m)
        print(f"上下文 ctx (前16字节): {ctx.hex()[:32]}...")

        # ---------------------- 2.5 计算掩码嵌入值c_h（公式11） ----------------------
        print("\n2.5 计算掩码嵌入值c_h（公式11: c_h = M ⊕ PRF_l(Kext, ctx)）")
        prf_val = prf_l(sr.Kext, ctx, L)
        C = M ^ prf_val
        print(f"PRF输出: {prf_val}")
        print(f"目标嵌入值 c_h: {C} (二进制 {bin(C)})")

        # ---------------------- 2.6 生成部分签名s_i（公式13） ----------------------
        print("\n2.6 生成部分签名s_i（公式13: s_i = α_i + c·λ_i·sk_i mod q）")
        partials = {}
        for vi in participants:
            si = (nonces[vi] + c * lambdas[vi] * sr.shares[vi]) % q
            partials[vi] = si
            print(f"节点 {vi} 的部分签名 s_{vi}: {si}")

        # ---------------------- 2.7 验证部分签名（公式14） ----------------------
        print("\n2.7 验证部分签名（公式14: s_i·G = R_i + c·λ_i·pk_i）")
        valid = True
        for vi in participants:
            lhs = scalar_mult(partials[vi], G)
            coeff = (c * lambdas[vi]) % q
            rhs = commitments[vi] + scalar_mult(coeff, sr.share_pks[vi])
            if lhs == rhs:
                print(f"节点 {vi} 部分签名验证: 通过")
            else:
                print(f"节点 {vi} 部分签名验证: 失败")
                valid = False
                break
        
        if not valid:
            print("部分签名验证失败，进入下一轮重试")
            continue

        # ---------------------- 2.8 聚合签名s（公式15） ----------------------
        print("\n2.8 聚合所有部分签名得到s（公式15: s = Σs_i mod q）")
        s = sum(partials[vi] for vi in participants) % q
        print(f"聚合签名响应 s: {s}")

        # ---------------------- 2.9 拒绝采样检查（公式16） ----------------------
        print("\n2.9 拒绝采样检查（公式16: s mod 2^L == c_h）")
        s_low = s % (2 ** L)
        print(f"s的低{L}位: {s_low} (二进制 {bin(s_low)})")
        print(f"目标值c_h: {C} (二进制 {bin(C)})")
        
        if s_low == C:
            print("\n✅ 拒绝采样条件满足！签名成功")
            print(f"本次签名共尝试了 {attempt} 次")
            
            # 构造最终签名
            sig = Signature(R=R, s=s)
            
            # 验证签名有效性
            print("\n" + "-" * 60)
            print("验证最终签名有效性")
            print("-" * 60)
            v = verify(m, sr.pk, sig)
            print(f"签名验证结果: {v}")
            
            # 提取嵌入信息
            print("\n" + "-" * 60)
            print("提取嵌入信息")
            print("-" * 60)
            recovered_M = extract(m, sr.pk, sig, sr.Kext, L)
            print(f"原始嵌入信息 M: {M}")
            print(f"提取得到的信息: {recovered_M}")
            print(f"信息匹配: {recovered_M == M}")
            
            print("\n" + "=" * 80)
            print("SignEmb算法测试完成！")
            print("=" * 80)
            return
        
        else:
            print("❌ 拒绝采样条件不满足，进入下一轮重试")

    # 如果超过最大重试次数
    print(f"\n❌ 超过最大重试次数 {Nmax}，签名失败")
    print("可以尝试增大Nmax参数或减小L参数")

if __name__ == "__main__":
    test_sign_emb_step_by_step()