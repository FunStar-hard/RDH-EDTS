# src/test_sddh_tss.py
"""完整流程测试脚本
在src目录下运行: python test_sddh_tss.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheme.setup import setup
from scheme.sign_emb import sign_emb
from scheme.verify import verify
from scheme.extract import extract
from crypto.encoding import encode_point, encode_scalar

def main():

    # ====================== 1. 参数设置 ======================
    print("\n[1/4] 系统参数设置")
    n = 5       # 总节点数
    t = 3       # 门限值
    L = 4       # 嵌入比特长度
    Nmax = 256  # 最大重试次数
    print(f"  总节点数 n = {n}")
    print(f"  门限值 t = {t}")
    print(f"  嵌入比特长度 L = {L}")
    print(f"  最大重试次数 Nmax = {Nmax}")

    # ====================== 2. 系统初始化 Setup ======================
    print("\n[2/4] 运行系统初始化 Setup")
    sr = setup(t, n, L)
    print(f"  ✓ 系统公钥 pk 生成成功 (压缩格式: {encode_point(sr.pk).hex()[:32]}...)")
    print(f"  ✓ 生成 {n} 个私钥份额和对应的公钥份额")
    print(f"  ✓ 提取密钥 Kext 生成成功 (前16字节: {sr.Kext.hex()[:32]})")

    # ====================== 3. 生成测试数据 ======================
    print("\n[3/4] 生成测试消息和嵌入信息")
    import secrets
    m = secrets.token_bytes(32)
    print(f"  待签名消息 (前16字节): {m.hex()[:32]}...")
    
    # 生成随机嵌入信息 (L位)
    M = secrets.randbelow(2 ** L)
    print(f"  待嵌入信息 M: 十进制 {M}, 二进制 {bin(M)[2:].zfill(L)}")

    # 选择前t个节点参与签名
    participants = list(range(1, t + 1))
    print(f"  参与签名的节点: {participants}")

    # ====================== 4. 带嵌入的门限签名 SignEmb ======================
    print("\n[4/4] 运行带信息嵌入的门限签名 SignEmb")
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

    if not result.success:
        print(f"  ✗ 签名失败! 已达到最大重试次数 {Nmax}")
        return

    print(f"  ✓ 签名成功! 共重试 {result.retries} 次 (理论平均重试次数: {2**L})")
    sig = result.signature
    print(f"  签名 R (压缩格式): {encode_point(sig.R).hex()}")
    print(f"  签名 s: {encode_scalar(sig.s).hex()}")
    print(f"  s 的低{L}位: {bin(sig.s % (2**L))[2:].zfill(L)} (十进制 {sig.s % (2**L)})")

    # ====================== 5. 公共验证 Verify ======================
    print("\n[验证阶段] 运行标准Schnorr公共验证")
    verify_result = verify(m, sr.pk, sig)
    if verify_result:
        print("  ✓ 签名验证通过! (与标准Schnorr验证完全一致)")
    else:
        print("  ✗ 签名验证失败!")
        return

    # ====================== 6. 授权信息提取 Extract ======================
    print("\n[提取阶段] 运行授权信息提取")
    recovered_M = extract(m, sr.pk, sig, sr.Kext, L)
    if recovered_M is None:
        print("  ✗ 信息提取失败!")
        return

    print(f"  提取到的信息: 十进制 {recovered_M}, 二进制 {bin(recovered_M)[2:].zfill(L)}")
    print(f"  原始嵌入信息: 十进制 {M}, 二进制 {bin(M)[2:].zfill(L)}")
    
    if recovered_M == M:
        print("  ✓ 信息提取成功! 与原始信息完全匹配")
    else:
        print("  ✗ 信息提取失败! 与原始信息不匹配")
if __name__ == "__main__":
    main()