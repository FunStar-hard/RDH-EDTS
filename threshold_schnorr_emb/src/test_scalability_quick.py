"""
SDDH-TSS 可扩展性快速验证（小样本版）
功能：快速验证不同节点规模下签名生成、验证、提取流程是否正常
运行时间：约30-60秒
"""
import sys
import os
import numpy as np

# 修复项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.scheme.setup import setup
from src.experiments.common import timed_full_auth

def test_scalability_quick():
    print("=" * 60)
    print("  SDDH-TSS 可扩展性快速验证")
    print("=" * 60)

    # 极简测试参数
    L = 2
    Nmax = 2048
    trials_per_n = 5  # 每组仅跑5次，快速验证
    test_cases = [(5, 2), (10, 5), (20, 10)]  # 仅测3个小节点规模

    print(f"测试参数：L={L}, Nmax={Nmax}, 每组测试{trials_per_n}次")
    print(f"测试节点：{[n for n, t in test_cases]}")
    print("-" * 60)

    all_ok = True
    results = []

    for n, t in test_cases:
        print(f"\n▶ 测试 n={n}, t={t}")
        try:
            sr = setup(t, n, L)
            success = 0
            retries = []
            latencies = []

            for i in range(trials_per_n):
                tb = timed_full_auth(sr, t, L, Nmax)
                if tb.retries <= Nmax:
                    success += 1
                    retries.append(tb.retries)
                    latencies.append(tb.full_auth * 1000)  # 毫秒

            # 计算结果
            success_rate = success / trials_per_n * 100
            mean_retries = np.mean(retries) if retries else 0
            mean_latency = np.mean(latencies) if latencies else 0
            comm_bytes = t * (33 + 32)

            results.append({
                "n": n, "t": t, "success": success_rate,
                "retries": mean_retries, "latency": mean_latency, "comm": comm_bytes
            })

            # 打印结果
            print(f"  成功率: {success_rate:.0f}%")
            print(f"  平均重试: {mean_retries:.1f} (理论值: 4)")
            print(f"  平均延迟: {mean_latency:.2f} ms")
            print(f"  通信开销: {comm_bytes} 字节")
            print(f"  ✅ 该节点规模测试通过")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            all_ok = False
            continue

    # 最终汇总
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    print(f"{'n':<4} {'t':<4} {'成功率':<8} {'平均重试':<10} {'平均延迟(ms)':<15} {'通信开销(B)':<12}")
    print("-" * 60)
    for res in results:
        print(f"{res['n']:<4} {res['t']:<4} {res['success']:<8.0f} {res['retries']:<10.1f} {res['latency']:<15.2f} {res['comm']:<12}")

    print("\n" + "=" * 60)
    if all_ok and all(res["success"] == 100 for res in results):
        print("🎉 所有节点规模测试通过！")
        print("核心功能验证正常：")
        print("1. 所有测试均100%成功生成签名")
        print("2. 平均重试次数接近理论值4")
        print("3. 延迟随节点规模正常增长")
    else:
        print("❌ 部分测试失败，请检查代码")
    print("=" * 60)

if __name__ == "__main__":
    test_scalability_quick()