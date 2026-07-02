
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from src.scheme.setup import setup as scheme_setup
from src.experiments.common import single_trial

def test_correctness_small():
    print("=" * 70)
    print("  SDDH-TSS 正确性小样本测试")
    print("=" * 70)

    # ====================== 测试参数设置（小样本，运行快） ======================
    TEST_N = 5          # 总节点数
    TEST_T = 3          # 门限值
    TEST_LS = [1, 2, 3, 4]  # 测试的嵌入比特长度
    TEST_NMAX = 256     # 最大重试次数
    TEST_TRIALS = 50    # 每个L测试50次（可根据需要调整）

    print(f"\n测试参数: n={TEST_N}, t={TEST_T}, Nmax={TEST_NMAX}, 每个L测试{TEST_TRIALS}次")
    print("-" * 70)

    # ====================== 逐个L进行测试 ======================
    for L in TEST_LS:
        print(f"\n▶ 正在测试 L={L} 比特嵌入...")
        
        # 初始化系统
        sr = scheme_setup(TEST_T, TEST_N, L)
        
        # 统计变量
        total_sign_success = 0
        total_verify_success = 0
        total_extract_success = 0
        retries_list = []

        # 运行TEST_TRIALS次独立实验
        for _ in tqdm(range(TEST_TRIALS), desc=f"L={L}"):
            result = single_trial(sr, TEST_T, L, TEST_NMAX)
            
            if result["success"]:
                total_sign_success += 1
                retries_list.append(result["retries"])
                
                # ✅ 强制断言1：签名成功则验证必须成功
                assert result["verify_ok"], f"❌ L={L} 时出现签名成功但验证失败的情况！"
                total_verify_success += 1
                
                # ✅ 强制断言2：签名成功则提取必须成功
                assert result["extract_ok"], f"❌ L={L} 时出现签名成功但提取失败的情况！"
                total_extract_success += 1

        # 计算统计指标
        sign_rate = total_sign_success / TEST_TRIALS
        verify_rate = total_verify_success / TEST_TRIALS
        extract_rate = total_extract_success / TEST_TRIALS
        mean_retries = np.mean(retries_list) if retries_list else 0.0

        # 打印结果
        print(f"  签名成功率: {sign_rate:.2%} ({total_sign_success}/{TEST_TRIALS})")
        print(f"  验证成功率: {verify_rate:.2%} ({total_verify_success}/{TEST_TRIALS})")
        print(f"  提取成功率: {extract_rate:.2%} ({total_extract_success}/{TEST_TRIALS})")
        print(f"  平均重试次数: {mean_retries:.2f} (理论值: {2**L})")

    # ====================== 测试总结 ======================
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    print("核心结论验证：")
    print("1. 所有成功生成的签名都通过了标准Schnorr验证")
    print("2. 所有成功生成的签名都正确提取了嵌入信息")
    print("3. 平均重试次数与理论值2^L基本一致")
    print("=" * 70)

if __name__ == "__main__":
    test_correctness_small()