"""Experiment E: Efficiency – timing and memory."""
from __future__ import annotations
#1.各模块耗时 2.FullAuth 随 L 的变化 3.FullAuth 随 t 的变化
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.config import get_scheme_params
from src.experiments.common import timed_full_auth
from src.scheme.setup import setup as scheme_setup
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig
from src.utils.time_utils import timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 运行效率实验，分为三个部分：1.模块耗时分解，2.FullAuth 随 L 的变化，3.FullAuth 随 t 的变化。每个部分都统计平均耗时和标准差，并保存结果为 CSV 文件和图表。
def run_efficiency(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("efficiency", {})
    num_trials = exp.get("num_trials", 50)

    # ---- Part 1: 模块耗时分解 ----
    logger.info("Efficiency Part 1: Module breakdown")
    bd_n = exp.get("breakdown_n", 20)# 分解实验的节点数，默认20
    bd_t = exp.get("breakdown_t", 5)# 分解实验的门限，默认5
    bd_L = exp.get("breakdown_L", 4)
    bd_Nmax = exp.get("breakdown_Nmax", 256)

    sr = scheme_setup(bd_t, bd_n, bd_L)

    #单独测量 setup 时间，后续实验中不重复测量，并记录初始化耗时
    with timer() as ts:
        _ = scheme_setup(bd_t, bd_n, bd_L)
    setup_time = ts["elapsed"]
    # 统计每个模块的耗时，运行 num_trials 次完整认证流程，并记录每个模块的时间和内存峰值
    breakdown_rows: List[Dict[str, Any]] = []
    for i in tqdm(range(num_trials), desc="Breakdown"):#运行 num_trials 次完整认证流程，记录每个模块的时间和内存峰值，保存为 breakdown_rows 列表
        tb = timed_full_auth(sr, bd_t, bd_L, bd_Nmax)
        breakdown_rows.append({
            "trial": i,
            "setup": setup_time,
            "com_gen": tb.com_gen,
            "part_sign": tb.part_sign,
            "share_ver": tb.share_ver,
            "agg": tb.agg,
            "verify": tb.verify,
            "extract": tb.extract,
            "full_auth": tb.full_auth,
            "retries": tb.retries,
            "mem_peak_bytes": tb.mem_peak_bytes,
        })
    # 将分解结果保存为 CSV 文件，包含每个模块的平均时间和标准差，以及内存峰值等指标
    save_rows_csv(breakdown_rows, out / "raw" / "efficiency_breakdown_raw.csv")

    # 计算每个模块的平均时间和标准差，并保存为 summary CSV 文件，方便后续分析和绘图
    keys = ["setup", "com_gen", "part_sign", "share_ver", "agg", "verify", "extract", "full_auth"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in breakdown_rows]
        summary[k + "_mean"] = float(np.mean(vals))#平均时间，转换为us
        summary[k + "_std"] = float(np.std(vals))
    summary["mem_peak_mean"] = float(np.mean([r["mem_peak_bytes"] for r in breakdown_rows]))
    summary["n"] = bd_n
    summary["t"] = bd_t
    summary["L"] = bd_L
    summary["Nmax"] = bd_Nmax
    save_rows_csv([summary], out / "tables" / "efficiency_breakdown_summary.csv")
    logger.info(f"  Breakdown: full_auth_mean={summary['full_auth_mean']*1000:.1f}ms")

    # ---- Part 2: FullAuth 随 L 的变化 ----
    logger.info("Efficiency Part 2: FullAuth vs L")
    Ls = exp.get("L_values", [2, 4, 6])
    ts_list = exp.get("t_values_for_L", [2, 3, 5, 10])
    fixed_n = exp.get("fixed_n_for_L", 20)# 固定节点数，观察不同 L 和 t 下的 FullAuth 耗时变化
    Nmax_eff = exp.get("Nmax", 2048)
    trials_per = exp.get("trials_per_combo", 30)
    # 统计不同 L 和 t 组合下的 FullAuth 平均耗时和标准差，保存为 CSV 文件，并绘制 FullAuth 耗时随 L 变化的图表
    lat_L_rows: List[Dict[str, Any]] = [] 
    #数据采集，对每个 t 和 L 组合，运行 trials_per 次完整认证流程，记录 FullAuth 的耗时，并计算平均值和标准差，保存为 lat_L_rows 列表
    for t_val in tqdm(ts_list, desc="FullAuth vs L"):#
        if t_val > fixed_n:#如果门限 t 大于节点数 n，则跳过该组合，因为不合法
            continue
        for L in Ls:#对于每个 L，初始化方案，运行多次认证流程，记录 FullAuth 的耗时，并计算平均值和标准差，保存为 lat_L_rows 列表
            sr2 = scheme_setup(t_val, fixed_n, L)#初始化方案，传入门限 t、节点数 n 和嵌入比特数 L，得到 SetupResult 对象 sr2，其中包含了预先生成的密钥和共享信息
            times = []
            for _ in range(trials_per):#对每个L初始化方案，运行多次认证流程，记录 FullAuth 的耗时，并计算平均值和标准差
                tb = timed_full_auth(sr2, t_val, L, Nmax_eff)
                times.append(tb.full_auth)
            lat_L_rows.append({ 
                "n": fixed_n, "t": t_val, "L": L,
                "mean_latency": float(np.mean(times)),
                "std_latency": float(np.std(times)),
            })

    save_rows_csv(lat_L_rows, out / "tables" / "efficiency_fullauth_vs_L.csv")

    #绘图
    setup_style()
    fig, ax = plt.subplots()
    for t_val in ts_list:
        if t_val > fixed_n:
            continue
        xs = [r["L"] for r in lat_L_rows if r["t"] == t_val]
        ys = [r["mean_latency"] * 1000 for r in lat_L_rows if r["t"] == t_val]
        es = [r["std_latency"] * 1000 for r in lat_L_rows if r["t"] == t_val]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"t={t_val}")
    ax.set_xlabel("Embedding bits L")
    ax.set_ylabel("FullAuth latency (ms)")
    ax.set_title(f"FullAuth latency vs L (n={fixed_n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, out / "figures" / "efficiency_fullauth_vs_L.png")

    # ---- Part 3: FullAuth随t变化   ----
    logger.info("Efficiency Part 3: FullAuth vs t")
    ns_list = exp.get("n_values_for_t", [5, 10, 20])#不同节点数 n 的列表，观察不同 n 和 t 下的 FullAuth 耗时变化
    fixed_L = exp.get("fixed_L_for_t", 4)# 固定嵌入比特数，观察不同 n 和 t 下的 FullAuth 耗时变化

    lat_t_rows: List[Dict[str, Any]] = []
    #数据采集，对每个 n 和 t 组合，运行 trials_per 次完整认证流程，记录 FullAuth 的耗时，并计算平均值和标准差，保存为 lat_t_rows 列表
    for n_val in tqdm(ns_list, desc="FullAuth vs t"):
        t_range = list(range(2, n_val + 1, max(1, n_val // 5)))
        if t_range[-1] != n_val:
            t_range.append(n_val)
        for t_val in t_range:
            sr3 = scheme_setup(t_val, n_val, fixed_L)
            times = []
            for _ in range(trials_per):
                tb = timed_full_auth(sr3, t_val, fixed_L, Nmax_eff)
                times.append(tb.full_auth)
            lat_t_rows.append({
                "n": n_val, "t": t_val, "L": fixed_L,
                "mean_latency": float(np.mean(times)),
                "std_latency": float(np.std(times)),
            })

    save_rows_csv(lat_t_rows, out / "tables" / "efficiency_fullauth_vs_t.csv")
    #绘图
    fig, ax = plt.subplots()
    for n_val in ns_list:
        xs = [r["t"] for r in lat_t_rows if r["n"] == n_val]
        ys = [r["mean_latency"] * 1000 for r in lat_t_rows if r["n"] == n_val]
        es = [r["std_latency"] * 1000 for r in lat_t_rows if r["n"] == n_val]
        ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3, label=f"n={n_val}")
    ax.set_xlabel("Threshold t")
    ax.set_ylabel("FullAuth latency (ms)")
    ax.set_title(f"FullAuth latency vs t (L={fixed_L})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, out / "figures" / "efficiency_fullauth_vs_t.png")

    logger.info("Efficiency experiment complete.")