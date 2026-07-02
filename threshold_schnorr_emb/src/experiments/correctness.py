"""Experiment A: Correctness and feasibility + Fig3 Enhanced."""#正确性与可行性实验
#1.签名能不能成功生成 2.生成的签名能不能被验证成功 3.能不能成功提取出嵌入信息 4.不同参数下的成功率变化
from __future__ import annotations

import logging#实验运行时记录详细日志，包括：实验开始时间、当前运行的参数组合、进度信息、错误提示、最终完成时间。所有日志会同时输出到控制台和run.log文件，方便你调试和复现实验。
from pathlib import Path#1.自动创建带时间戳的实验输出目录（outputs/[时间戳]_correctness/2.拼接 CSV 表格和图片文件的保存路径
from typing import Any, Dict, List

import numpy as np#数值计算库
from tqdm import tqdm#实验需要运行大量参数组合 × 大量试验次数（比如 8×2×2×1000=32000 次完整签名流程），tqdm会在命令行显示实时进度条，让你直观看到实验运行状态和剩余时间。

from src.config import get_scheme_params #配置文件加载工具，从configs/experiment_correctness.yaml中读取所有实验参数
from src.experiments.common import single_trial #单次完整实验流程函数，执行一次完整的 "Setup→SignEmb→Verify→Extract" 流程，返回本次实验的所有结果
from src.scheme.setup import setup as scheme_setup#方案的setup函数，生成公私钥和其他参数
from src.utils.io_utils import save_rows_csv#CSV文件保存工具，将实验结果以表格形式保存到CSV文件中，方便后续分析和绘图
from src.utils.plot_utils import setup_style, save_fig#绘图工具，设置统一的图表风格和保存图表文件的函数

import matplotlib#绘图库
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_correctness(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:#正确性实验主函数，接受配置参数、输出目录和日志记录器作为输入，执行完整的正确性实验流程
  
    exp = cfg.get("correctness", {})# 获取实验参数
    ns = exp.get("n_values", [5, 10])
    ts = exp.get("t_values", [2, 3])
    Ls = exp.get("L_values", [1, 2, 3, 4, 5, 6, 7, 8])
    Nmaxs = exp.get("Nmax_values", [256, 2048])
    num_trials = exp.get("num_trials", 1000)# 每组参数组合的试验次数

    raw_rows: List[Dict[str, Any]] = []#记录每次试验的详细结果，包括：输入参数、是否成功生成签名、验证结果、提取结果、重试次数等指标，最终会保存到CSV文件中，方便后续分析和绘图
    summary_rows: List[Dict[str, Any]] = []#记录每组参数组合的统计结果，包括：成功率、平均重试次数等指标，最终会保存到CSV文件中，方便后续分析和绘图
    #遍历参数组合，运行实验，并记录结果
    combos = []#初始化一个空列表，用来存储所有最终有效的参数组合。生成所有参数组合，过滤掉不合法的组合（比如t > n），最终得到一个包含所有合法参数组合的列表，供后续实验使用
    for n in ns:# 遍历不同的用户数量n
        for t in ts:# 遍历不同的阈值t
            if t > n:# 阈值t不能大于用户数量n，否则不合法，直接跳过这个组合
                continue
            for L in Ls:# 遍历不同的嵌入比特数L
                for Nmax in Nmaxs:# 遍历不同的最大重试次数Nmax
                    combos.append((n, t, L, Nmax))#将合法的参数组合(n, t, L, Nmax)添加到combos列表中，供后续实验使用

    logger.info(f"Correctness: {len(combos)} parameter combos x {num_trials} trials")#
    #每组合参数运行num_trials次实验，记录成功率和重试次数等指标
    for n, t, L, Nmax in tqdm(combos, desc="Correctness"):
        logger.info(f"  n={n}, t={t}, L={L}, Nmax={Nmax}")
        sr = scheme_setup(t, n, L)#调用方案的setup函数，生成公私钥和其他参数
        # 统计指标初始化
        retries_list = []
        sign_ok = 0
        verify_ok = 0
        extract_ok = 0
        #每组合参数运行num_trials次实验，调用single_trial函数执行一次完整的 "Setup→SignEmb→Verify→Extract" 流程，记录每次试验的结果，并统计成功率和重试次数等指标
        for trial_i in range(num_trials):
            row = single_trial(sr, t, L, Nmax)
            row.update({"n": n, "t": t, "L": L, "Nmax": Nmax, "trial": trial_i})
            raw_rows.append(row)
            # 统计成功率和重试次数
            if row["success"]:
                sign_ok += 1
                retries_list.append(row["retries"])
            if row["verify_ok"]:# 验证成功率
                verify_ok += 1
            if row["extract_ok"]:# 提取成功率
                extract_ok += 1
        # 计算统计指标
        retries_arr = np.array(retries_list) if retries_list else np.array([0])
        summary_rows.append({
            "n": n, "t": t, "L": L, "Nmax": Nmax,
            "num_trials": num_trials,
            "sign_success": sign_ok,
            "sign_rate": sign_ok / num_trials,
            "verify_rate": verify_ok / num_trials,
            "extract_rate": extract_ok / num_trials,
            "mean_retries": float(np.mean(retries_arr)) if retries_list else float("nan"),
            "median_retries": float(np.median(retries_arr)) if retries_list else float("nan"),
            "p95_retries": float(np.percentile(retries_arr, 95)) if retries_list else float("nan"),
        })
        logger.info(f"    sign_rate={sign_ok}/{num_trials}, "
                     f"mean_retries={summary_rows[-1]['mean_retries']:.2f}")
    #保存结果到CSV文件
    save_rows_csv(raw_rows, out / "raw" / "correctness_raw.csv")
    save_rows_csv(summary_rows, out / "tables" / "correctness_summary.csv")
    logger.info("Correctness CSV saved.")

    #绘制成功率随嵌入比特L变化的图表，分不同n和t展示
    setup_style()
    for Nmax in Nmaxs:
        fig, ax = plt.subplots()
        for n in ns:
            for t in ts:
                if t > n:
                    continue
                xs = []
                ys = []
                for row in summary_rows:
                    if row["n"] == n and row["t"] == t and row["Nmax"] == Nmax:
                        xs.append(row["L"])
                        ys.append(row["sign_rate"])
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"n={n},t={t}")
        ax.set_xlabel("Embedding bits L")
        ax.set_ylabel("Sign success rate")
        ax.set_title(f"Sign success rate vs L (Nmax={Nmax})")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / f"correctness_sign_rate_Nmax{Nmax}.png")

    logger.info("Correctness figures saved.")

    # ---- Run Fig3 Enhanced experiments ----
    from src.experiments.fig3_enhanced import run_fig3_enhanced
    run_fig3_enhanced(cfg, out, logger)