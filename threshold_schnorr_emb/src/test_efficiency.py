"""
SDDH-TSS 效率专项测试脚本（对应论文4.4节 Efficiency）
运行路径：在 src 根目录下打开 cmd，执行 python test_efficiency.py
"""
import sys
import os

# ====================== 修复导入路径（关键）======================
# 获取项目根目录（src的上级目录 threshold_schnorr_emb）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python模块搜索路径
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
from src.scheme.setup import setup
from src.experiments.common import timed_full_auth

def test_efficiency():
    print("=" * 80)
    print("  SDDH-TSS 效率测试（对应论文4.4节 Efficiency）")
    print("=" * 80)

    # ====================== 测试参数（与论文表6完全一致） ======================
    print("\n📌 测试参数配置")
    n = 20       # 总节点数（表6固定n=20）
    t = 5        # 门限值（表6固定t=5）
    L = 4        # 嵌入比特长度（表6固定l=4）
    Nmax = 256   # 最大重试次数
    run_times = 10  # 小样本运行次数（取平均值，可自行调整）
    print(f"  总节点数 n = {n}")
    print(f"  门限值 t = {t}")
    print(f"  嵌入比特长度 L = {L}")
    print(f"  最大重试次数 Nmax = {Nmax}")
    print(f"  测试运行次数 = {run_times} 次")

    # ====================== 1. Setup 阶段计时 ======================
    print("\n" + "-" * 70)
    print("⏱️  阶段1：系统初始化 Setup")
    print("-" * 70)
    t0 = time.perf_counter()
    sr = setup(t, n, L)
    setup_time = time.perf_counter() - t0
    print(f"  Setup 耗时: {setup_time * 1e6:.3f} μs")
    print(f"  理论时间复杂度: O(nt)")
    print(f"  理论空间复杂度: O(n)")

    # ====================== 2. 完整认证流程模块计时 ======================
    print("\n" + "-" * 70)
    print(f"⏱️  阶段2：完整认证流程模块分解（平均 {run_times} 次运行）")
    print("-" * 70)

    # 收集所有运行结果
    results = {
        "com_gen": [], "part_sign": [], "share_ver": [], "agg": [],
        "verify": [], "extract": [], "full_auth": [], "retries": [], "mem_peak": []
    }

    for i in range(run_times):
        print(f"  正在运行第 {i+1}/{run_times} 次...", end="\r")
        tb = timed_full_auth(sr, t, L, Nmax)
        results["com_gen"].append(tb.com_gen)
        results["part_sign"].append(tb.part_sign)
        results["share_ver"].append(tb.share_ver)
        results["agg"].append(tb.agg)
        results["verify"].append(tb.verify)
        results["extract"].append(tb.extract)
        results["full_auth"].append(tb.full_auth)
        results["retries"].append(tb.retries)
        results["mem_peak"].append(tb.mem_peak_bytes)

    # 计算平均值（转换为微秒，与论文表6单位完全一致）
    avg = {}
    for key in results:
        if key == "retries" or key == "mem_peak":
            avg[key] = np.mean(results[key])
        else:
            avg[key] = np.mean(results[key]) * 1e6  # 秒 → 微秒

    # 打印各模块详细信息
    print(f"\n  {'模块':<12} {'平均时延(μs)':<15} {'理论时间复杂度':<18} {'占总耗时比例':<10}")
    print("  " + "-" * 65)
    
    modules = [
        ("ComGen(承诺生成)", avg["com_gen"], "O(t)"),
        ("PartSign(部分签名)", avg["part_sign"], "O(t)"),
        ("ShareVer(份额验证)", avg["share_ver"], "O(t)"),
        ("Agg(签名聚合)", avg["agg"], "O(t)"),
        ("Verify(公共验证)", avg["verify"], "O(1)"),
        ("Extract(授权提取)", avg["extract"], "O(1)"),
        ("FullAuth(完整认证)", avg["full_auth"], "O(2^l·t²)")
    ]

    total_sign_time = avg["com_gen"] + avg["part_sign"] + avg["share_ver"] + avg["agg"]
    for name, latency, complexity in modules:
        ratio = latency / avg["full_auth"] * 100 if avg["full_auth"] > 0 else 0
        print(f"  {name:<12} {latency:<15.3f} {complexity:<18} {ratio:<10.1f}%")

    # ====================== 3. 额外统计信息 ======================
    print("\n" + "-" * 70)
    print("📊 额外统计信息")
    print("-" * 70)
    print(f"  平均重试次数: {avg['retries']:.2f} 次（理论值: 2^{L} = {2**L} 次）")
    print(f"  平均内存峰值: {avg['mem_peak'] / 1024:.2f} KB")
    print(f"  签名生成阶段总占比: {total_sign_time / avg['full_auth'] * 100:.1f}%")
    print(f"  验证+提取阶段总占比: {(avg['verify'] + avg['extract']) / avg['full_auth'] * 100:.1f}%")
   

   

if __name__ == "__main__":
    try:
        test_efficiency()
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)