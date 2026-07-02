"""
SDDH-TSS 4.5节 方案对比专项测试脚本
运行路径：在 src 根目录下打开 cmd，执行 python test_comparison.py
对比维度：通信成本、签名时延、验证时延
对比方案：逐节点签名、标准门限Schnorr、门限+额外字段、本文方案
"""
import sys
import os
import time
import secrets
import numpy as np

# ====================== 修复导入路径（关键，解决ModuleNotFoundError）======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.baselines import (
    per_node_sign, per_node_verify, per_node_comm_cost,
    threshold_schnorr_sign, threshold_schnorr_comm_cost,
    threshold_schnorr_extra_sign, threshold_schnorr_extra_comm_cost,
    embedded_threshold_comm_cost,
)

def test_comparison():
    print("=" * 90)
    print("  SDDH-TSS 4.5节 方案对比专项测试")
    print("=" * 90)

    # ====================== 测试参数（与论文表8/图7完全一致）======================
    print("\n📌 测试参数配置")
    n = 10       # 总节点数（论文固定n=10）
    t = 5        # 门限值（论文固定t=5）
    Ls = [2, 4, 6]  # 嵌入比特长度（论文测试L=2,4,6）
    Nmax = 2048  # 最大重试次数
    run_times = 10  # 小样本运行次数（快速测试，可自行增加到100）
    print(f"  总节点数 n = {n}")
    print(f"  门限值 t = {t}")
    print(f"  嵌入比特长度 L = {Ls}")
    print(f"  每个参数组合运行次数 = {run_times} 次")

    # 初始化系统参数（所有方案共用同一套密钥）
    print("\n🔧 初始化系统参数...")
    sr = setup(t, n, L=max(Ls))
    participants = list(range(1, t + 1))
    node_keys = [(sr.shares[vi], sr.share_pks[vi]) for vi in participants]
    node_pks = [sr.share_pks[vi] for vi in participants]

    # 存储所有测试结果
    results = []

    # ====================== 遍历所有L值进行测试 ======================
    for L in Ls:
        print(f"\n{'='*70}")
        print(f"  正在测试 L = {L}")
        print(f"{'='*70}")

        # ---------------------- 1. 计算通信成本（对应论文表8）----------------------
        print("\n📡 通信成本对比（字节）")
        print("-" * 50)
        cc_per_node = per_node_comm_cost(t, L)
        cc_std = threshold_schnorr_comm_cost(t, L)
        cc_extra = threshold_schnorr_extra_comm_cost(t, L)
        cc_ours = embedded_threshold_comm_cost(t, L)
        
        print(f"  逐节点签名: {cc_per_node:4d} 字节")
        print(f"  标准门限:   {cc_std:4d} 字节")
        print(f"  门限+额外:   {cc_extra:4d} 字节")
        print(f"  本文方案:   {cc_ours:4d} 字节")
        #print(f"  ✅ 本文方案通信成本与标准门限完全相同，比逐节点节省 {((cc_per_node - cc_ours)/cc_per_node)*100:.1f}%")

        # ---------------------- 2. 签名时延测试 ----------------------
        print("\n⏱️  签名时延对比（毫秒）")
        print("-" * 50)
        
        # 2.1 逐节点签名
        sign_times_pn = []
        for _ in range(run_times):
            m = secrets.token_bytes(32)
            t0 = time.perf_counter()
            per_node_sign(m, node_keys)
            sign_times_pn.append(time.perf_counter() - t0)
        avg_pn_sign = np.mean(sign_times_pn) * 1000
        print(f"  逐节点签名: {avg_pn_sign:6.2f} ms")

        # 2.2 标准门限签名
        sign_times_std = []
        for _ in range(run_times):
            m = secrets.token_bytes(32)
            t0 = time.perf_counter()
            threshold_schnorr_sign(m, participants, sr.shares, sr.share_pks, sr.pk)
            sign_times_std.append(time.perf_counter() - t0)
        avg_std_sign = np.mean(sign_times_std) * 1000
        print(f"  标准门限:   {avg_std_sign:6.2f} ms")

        # 2.3 门限+额外字段
        sign_times_extra = []
        for _ in range(run_times):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(2 ** L)
            t0 = time.perf_counter()
            threshold_schnorr_extra_sign(m, participants, sr.shares, sr.share_pks, sr.pk, sr.Kext, M, L)
            sign_times_extra.append(time.perf_counter() - t0)
        avg_extra_sign = np.mean(sign_times_extra) * 1000
        print(f"  门限+额外:   {avg_extra_sign:6.2f} ms")

        # 2.4 本文方案
        sign_times_ours = []
        for _ in range(run_times):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(2 ** L)
            t0 = time.perf_counter()
            sign_emb(
                m=m, participants=participants,
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=L, Nmax=Nmax,
                verify_partial=False
            )
            sign_times_ours.append(time.perf_counter() - t0)
        avg_ours_sign = np.mean(sign_times_ours) * 1000
        print(f"  本文方案:   {avg_ours_sign:6.2f} ms")
        #print(f"  ℹ️  本文方案签名时延约为标准门限的 {avg_ours_sign/avg_std_sign:.1f} 倍（L={L}）")

        # ---------------------- 3. 验证时延测试 ----------------------
        print("\n✅ 验证时延对比（毫秒）")
        print("-" * 50)
        
        # 预生成签名用于验证测试
        m_test = secrets.token_bytes(32)
        sig_pn = per_node_sign(m_test, node_keys)
        sig_std = threshold_schnorr_sign(m_test, participants, sr.shares, sr.share_pks, sr.pk)
        sig_extra, _ = threshold_schnorr_extra_sign(m_test, participants, sr.shares, sr.share_pks, sr.pk, sr.Kext, 0, L)
        res_ours = sign_emb(m=m_test, participants=participants, shares=sr.shares, share_pks=sr.share_pks,
                           pk=sr.pk, Kext=sr.Kext, M=0, L=L, Nmax=Nmax, verify_partial=False)
        sig_ours = res_ours.signature

        # 3.1 逐节点验证
        ver_times_pn = []
        for _ in range(run_times):
            t0 = time.perf_counter()
            per_node_verify(m_test, node_pks, sig_pn)
            ver_times_pn.append(time.perf_counter() - t0)
        avg_pn_ver = np.mean(ver_times_pn) * 1000
        print(f"  逐节点签名: {avg_pn_ver:6.3f} ms")

        # 3.2 标准门限验证
        ver_times_std = []
        for _ in range(run_times):
            t0 = time.perf_counter()
            verify(m_test, sr.pk, sig_std)
            ver_times_std.append(time.perf_counter() - t0)
        avg_std_ver = np.mean(ver_times_std) * 1000
        print(f"  标准门限:   {avg_std_ver:6.3f} ms")

        # 3.3 门限+额外字段验证
        ver_times_extra = []
        for _ in range(run_times):
            t0 = time.perf_counter()
            verify(m_test, sr.pk, sig_extra)
            ver_times_extra.append(time.perf_counter() - t0)
        avg_extra_ver = np.mean(ver_times_extra) * 1000
        print(f"  门限+额外:   {avg_extra_ver:6.3f} ms")

        # 3.4 本文方案验证
        ver_times_ours = []
        for _ in range(run_times):
            t0 = time.perf_counter()
            verify(m_test, sr.pk, sig_ours)
            ver_times_ours.append(time.perf_counter() - t0)
        avg_ours_ver = np.mean(ver_times_ours) * 1000
        print(f"  本文方案:   {avg_ours_ver:6.3f} ms")
        #print(f"  ✅ 本文方案验证时延与标准门限完全一致")

        # 保存当前L的结果
        results.append({
            "L": L,
            "comm": {"pn": cc_per_node, "std": cc_std, "extra": cc_extra, "ours": cc_ours},
            "sign": {"pn": avg_pn_sign, "std": avg_std_sign, "extra": avg_extra_sign, "ours": avg_ours_sign},
            "verify": {"pn": avg_pn_ver, "std": avg_std_ver, "extra": avg_extra_ver, "ours": avg_ours_ver}
        })



if __name__ == "__main__":
    try:
        test_comparison()
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)