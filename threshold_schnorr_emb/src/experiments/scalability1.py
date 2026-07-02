"""
对应论文5.6节 可扩展性与算法级实时分析
生成：表9、图8a、图8b
"""
from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.experiments.common import timed_full_auth
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_scalability(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("scalability", {})
    Nmax = exp.get("Nmax", 4096)
    num_trials = exp.get("num_trials", 100)

    # ================================================================
    # 实验1：固定t/n=0.5和L=2，变化n → 对应论文表9、图8a
    # ================================================================
    scale_n_cfg = exp.get("scale_n", {})
    if scale_n_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp1: Vary n, fixed t/n=0.5, L=2")
        logger.info("=" * 60)

        # 论文参数：固定阈值比0.5，嵌入长度L=2
        ratio = 0.5
        L = 2
        n_values = [5, 10, 20, 30, 50, 70, 100]  

        rows_n: List[Dict[str, Any]] = []
        for n in tqdm(n_values, desc="Scale-n"):
            t_val = max(2, int(n * ratio))
            logger.info(f"  n={n}, t={t_val}, L={L}")

            sr = scheme_setup(t_val, n, L)
            sign_ok = 0
            retry_list = []
            auth_times = []

            # 运行num_trials次实验
            for _ in range(num_trials):
                tb = timed_full_auth(sr, t_val, L, Nmax)
                auth_times.append(tb.full_auth)
                if tb.verify > 0:
                    sign_ok += 1
                    retry_list.append(tb.retries)

            # 计算通信开销（与论文表9一致）
            comm = 65 * t_val + 64  # 论文通信模型：t个节点各65字节 + 最终签名64字节

            #表数据
            rows_n.append({
                "n": n,
                "t": t_val,
                "success_rate": sign_ok / num_trials * 100, 
                "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
                "mean_auth_us": float(np.mean(auth_times)) * 1000 * 1000,  # 单位为微秒
                "comm_bytes": comm,
            })
            logger.info(f"    success={sign_ok}/{num_trials}, "
                         f"auth={rows_n[-1]['mean_auth_us']:.2f}μs, "
                         f"comm={comm}B")

        # 保存论文表9数据
        save_rows_csv(rows_n, out / "tables" / "table9_scalability_vary_n.csv")

        # 生成论文图8a：FullAuth时延 + 通信开销 vs n
        setup_style()
        fig, ax1 = plt.subplots(figsize=(10, 7))
        xs = [r["n"] for r in rows_n]
        ys_latency = [r["mean_auth_us"] / 1000 for r in rows_n]  # 转换为毫秒便于绘图
        ys_comm = [r["comm_bytes"] for r in rows_n]

        # 左轴：时延
        ax1.errorbar(xs, ys_latency, marker="o", capsize=4, linewidth=2,
                     label="FullAuth Latency", color="#4e79a7")
        ax1.set_xlabel("Number of nodes n", fontsize=13)
        ax1.set_ylabel("Mean FullAuth Latency (ms)", fontsize=13)
        ax1.tick_params(axis='y', labelcolor="#4e79a7")

        # 右轴：通信开销
        ax2 = ax1.twinx()
        ax2.plot(xs, ys_comm, marker="^", color="#f28e2b", linewidth=2,
                 label="Communication Overhead")
        ax2.set_ylabel("Communication Overhead (bytes)", fontsize=13)
        ax2.tick_params(axis='y', labelcolor="#f28e2b")

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

        ax1.set_title("FullAuth Latency and Communication vs n (t/n=0.5, L=2)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "fig8a_scalability_latency_comm.png")

    # ================================================================
    # 实验3：n和L的交互影响 → 对应论文图8b
    # ================================================================
    scale_nL_cfg = exp.get("scale_n_L", {})
    if scale_nL_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp3: Vary n with multiple L values")
        logger.info("=" * 60)

        ratio = 0.5
        L_values = [2, 4, 6]  # 论文图8b展示的嵌入长度
        n_values = [5, 10, 20, 30, 50, 70, 100]

        rows_nL: List[Dict[str, Any]] = []
        for L in tqdm(L_values, desc="Scale-nL"):
            for n in n_values:
                t_val = max(2, int(n * ratio))
                logger.info(f"  n={n}, t={t_val}, L={L}")

                sr = scheme_setup(t_val, n, L)
                auth_times = []

                for _ in range(num_trials):
                    tb = timed_full_auth(sr, t_val, L, Nmax)
                    auth_times.append(tb.full_auth)

                rows_nL.append({
                    "n": n,
                    "t": t_val,
                    "L": L,
                    "mean_auth_ms": float(np.mean(auth_times)) * 1000,
                })

        # 保存实验3原始数据
        save_rows_csv(rows_nL, out / "tables" / "scalability_n_L_interaction.csv")

        # 生成论文图8b：不同L下FullAuth时延 vs n
        setup_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        colors_L = ["tab:blue", "tab:orange", "tab:green"]
        markers_L = ["o", "s", "^"]

        for idx, L in enumerate(L_values):
            sub = [r for r in rows_nL if r["L"] == L]
            xs = [r["n"] for r in sub]
            ys = [r["mean_auth_ms"] for r in sub]
            ax.plot(xs, ys, marker=markers_L[idx], color=colors_L[idx],
                    linewidth=2, label=f"L={L}")

        ax.set_xlabel("Number of nodes n", fontsize=13)
        ax.set_ylabel("Mean FullAuth Latency (ms)", fontsize=13)
        ax.set_title("FullAuth Latency vs n for Different L (t/n=0.5)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "fig8b_scalability_multiL.png")

    logger.info("Scalability experiment complete (only paper-published content retained).")