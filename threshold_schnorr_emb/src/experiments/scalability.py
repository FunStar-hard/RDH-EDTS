"""Experiment G: Scalability – improved version."""
from __future__ import annotations
#1.固定 t/n 比例和 L，变化 n，观察认证时延、通信量、内存占用等指标的增长趋势 2.固定 n，变化 t/n 比例，观察认证时延和成功率的变化 3.分析 n 和 L 的交互影响，例如在较大 n 下不同 L 的表现差异
import logging#日志输出
import math
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract
from src.scheme.baselines import (
    per_node_sign, per_node_verify, per_node_comm_cost,
    threshold_schnorr_sign, threshold_schnorr_comm_cost,
    embedded_threshold_comm_cost,
)
from src.experiments.common import timed_full_auth
from src.crypto.curve_utils import get_order
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#cfg：实验配置（字典，控制各子实验的参数）
def run_scalability(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("scalability", {})
    Nmax = exp.get("Nmax", 4096)
    num_trials = exp.get("num_trials", 100)

    # ================================================================
    # 实验1：固定 t/n 比例和 L，变化 n
    # ================================================================
    scale_n_cfg = exp.get("scale_n", {})
    if scale_n_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp1: Vary n, fixed t/n ratio")
        logger.info("=" * 60)
        #固定0.5比例
        ratio = scale_n_cfg.get("ratio", 0.5)
        L = scale_n_cfg.get("L", 2)
        n_values = scale_n_cfg.get("n_values", [5, 10, 20, 30, 50, 70, 100])

        rows_n: List[Dict[str, Any]] = []
        #根据固定阈值比计算门限值t
        for n in tqdm(n_values, desc="Scale-n"):
            t_val = max(2, int(n * ratio))
            logger.info(f"  n={n}, t={t_val}, L={L}")
            #初始化系统参数
            sr = scheme_setup(t_val, n, L)

            # 分别计时 Setup
            t0 = time.perf_counter()
            _ = scheme_setup(t_val, n, L)
            setup_time = time.perf_counter() - t0
            #初始化统计变量
            sign_ok = 0
            retry_list = []
            auth_times = []
            com_gen_times = []
            part_sign_times = []
            share_ver_times = []
            agg_times = []
            verify_times = []
            extract_times = []
            mem_list = []
            #循环运行num_trials次实验
            for trial_i in range(num_trials):
                tb = timed_full_auth(sr, t_val, L, Nmax)
                auth_times.append(tb.full_auth)
                com_gen_times.append(tb.com_gen)
                part_sign_times.append(tb.part_sign)
                share_ver_times.append(tb.share_ver)
                agg_times.append(tb.agg)
                verify_times.append(tb.verify)
                extract_times.append(tb.extract)
                mem_list.append(tb.mem_peak_bytes)

                if tb.verify > 0:  # 签名成功了才会有verify
                    sign_ok += 1
                    retry_list.append(tb.retries)
           
            comm = embedded_threshold_comm_cost(t_val, L)#计算通信成本，基于门限 Schnorr + 额外字段的通信成本函数，传入当前的门限值 t_val 和嵌入信息长度 L，得到总的通信成本（以字节为单位）
            #计算平均值
            rows_n.append({
                "n": n,
                "t": t_val,
                "ratio": ratio,
                "L": L,
                "Nmax": Nmax,
                "num_trials": num_trials,
                "success_rate": sign_ok / num_trials,
                "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
                "setup_ms": setup_time * 1000,
                "mean_auth_ms": float(np.mean(auth_times)) * 1000,
                "std_auth_ms": float(np.std(auth_times)) * 1000,
                "mean_com_gen_ms": float(np.mean(com_gen_times)) * 1000,
                "mean_part_sign_ms": float(np.mean(part_sign_times)) * 1000,
                "mean_share_ver_ms": float(np.mean(share_ver_times)) * 1000,
                "mean_agg_ms": float(np.mean(agg_times)) * 1000,
                "mean_verify_ms": float(np.mean(verify_times)) * 1000,
                "mean_extract_ms": float(np.mean(extract_times)) * 1000,
                "mean_mem_KB": float(np.mean(mem_list)) / 1024,
                "comm_bytes": comm,
                "per_node_auth_ms": float(np.mean(auth_times)) * 1000 / t_val,
            })
            logger.info(f"    success={sign_ok}/{num_trials}, "
                         f"auth={rows_n[-1]['mean_auth_ms']:.1f}ms, "
                         f"comm={comm}B")

        save_rows_csv(rows_n, out / "tables" / "scalability_vary_n.csv")

        # ---- 图1a：总认证时延 vs n ----
        setup_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        xs = [r["n"] for r in rows_n]
        ys = [r["mean_auth_ms"] for r in rows_n]
        es = [r["std_auth_ms"] for r in rows_n]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=2,
                     label="FullAuth (total)")
        ax.set_xlabel("Number of nodes n", fontsize=13)
        ax.set_ylabel("Mean latency (ms)", fontsize=13)
        ax.set_title(f"FullAuth latency vs n (t/n={ratio}, L={L})", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "scale_auth_vs_n.png")

        # ---- 图1b：各模块时延堆叠 vs n ----
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        modules = ["com_gen", "part_sign", "share_ver", "agg", "verify", "extract"]
        labels_m = ["ComGen", "PartSign", "ShareVer", "Aggregate", "Verify", "Extract"]
        colors_m = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

        bottom = np.zeros(len(xs))
        for mod, lab, clr in zip(modules, labels_m, colors_m):
            vals = [r[f"mean_{mod}_ms"] for r in rows_n]
            ax2.bar(range(len(xs)), vals, bottom=bottom, label=lab, color=clr)
            bottom += np.array(vals)

        ax2.set_xticks(range(len(xs)))
        ax2.set_xticklabels([str(x) for x in xs])
        ax2.set_xlabel("Number of nodes n", fontsize=13)
        ax2.set_ylabel("Time (ms)", fontsize=13)
        ax2.set_title(f"Module breakdown vs n (t/n={ratio}, L={L})", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
        save_fig(fig2, out / "figures" / "scale_module_breakdown_vs_n.png")

        # ---- 图1c：每节点平均认证时延 vs n ----
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        ys_per = [r["per_node_auth_ms"] for r in rows_n]
        ax3.plot(xs, ys_per, "s-", linewidth=2, markersize=8)
        ax3.set_xlabel("Number of nodes n", fontsize=13)
        ax3.set_ylabel("Per-node auth latency (ms)", fontsize=13)
        ax3.set_title(f"Per-node latency vs n (t/n={ratio}, L={L})", fontsize=14)
        ax3.grid(True, alpha=0.3)
        save_fig(fig3, out / "figures" / "scale_per_node_vs_n.png")

        # ---- 图1d：Setup时间 vs n ----
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        ys_setup = [r["setup_ms"] for r in rows_n]
        ax4.plot(xs, ys_setup, "D-", color="tab:purple", linewidth=2, markersize=8)
        ax4.set_xlabel("Number of nodes n", fontsize=13)
        ax4.set_ylabel("Setup time (ms)", fontsize=13)
        ax4.set_title(f"Setup time vs n (t/n={ratio}, L={L})", fontsize=14)
        ax4.grid(True, alpha=0.3)
        save_fig(fig4, out / "figures" / "scale_setup_vs_n.png")

        # ---- 图1e：通信量 vs n ----
        fig5, ax5 = plt.subplots(figsize=(10, 7))
        ys_comm = [r["comm_bytes"] for r in rows_n]
        ax5.plot(xs, ys_comm, "^-", color="tab:brown", linewidth=2, markersize=8)
        ax5.set_xlabel("Number of nodes n", fontsize=13)
        ax5.set_ylabel("Communication cost (bytes)", fontsize=13)
        ax5.set_title(f"Communication vs n (t/n={ratio}, L={L})", fontsize=14)
        ax5.grid(True, alpha=0.3)
        save_fig(fig5, out / "figures" / "scale_comm_vs_n.png")

        # ---- 图1f：内存 vs n ----
        fig6, ax6 = plt.subplots(figsize=(10, 7))
        ys_mem = [r["mean_mem_KB"] for r in rows_n]
        ax6.plot(xs, ys_mem, "v-", color="tab:olive", linewidth=2, markersize=8)
        ax6.set_xlabel("Number of nodes n", fontsize=13)
        ax6.set_ylabel("Peak memory (KB)", fontsize=13)
        ax6.set_title(f"Memory usage vs n (t/n={ratio}, L={L})", fontsize=14)
        ax6.grid(True, alpha=0.3)
        save_fig(fig6, out / "figures" / "scale_memory_vs_n.png")

    # ================================================================
    # 实验2：固定 n，变化 t/n 比例
    # ================================================================
    scale_ratio_cfg = exp.get("scale_ratio", {})
    if scale_ratio_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp2: Vary t/n ratio, fixed n")
        logger.info("=" * 60)

        n = scale_ratio_cfg.get("n", 50)
        L = scale_ratio_cfg.get("L", 2)
        ratios = scale_ratio_cfg.get("ratios", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        rows_ratio: List[Dict[str, Any]] = []

        for ratio in tqdm(ratios, desc="Scale-ratio"):
            t_val = max(2, int(n * ratio))
            logger.info(f"  n={n}, t={t_val}, ratio={ratio}, L={L}")

            sr = scheme_setup(t_val, n, L)
            auth_times = []
            share_ver_times = []
            sign_ok = 0
            retry_list = []

            for _ in range(num_trials):
                tb = timed_full_auth(sr, t_val, L, Nmax)
                auth_times.append(tb.full_auth)
                share_ver_times.append(tb.share_ver)
                if tb.verify > 0:
                    sign_ok += 1
                    retry_list.append(tb.retries)

            comm = embedded_threshold_comm_cost(t_val, L)

            rows_ratio.append({
                "n": n,
                "t": t_val,
                "ratio": ratio,
                "L": L,
                "success_rate": sign_ok / num_trials,
                "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
                "mean_auth_ms": float(np.mean(auth_times)) * 1000,
                "std_auth_ms": float(np.std(auth_times)) * 1000,
                "mean_share_ver_ms": float(np.mean(share_ver_times)) * 1000,
                "comm_bytes": comm,
            })

        save_rows_csv(rows_ratio, out / "tables" / "scalability_vary_ratio.csv")

        # ---- 图2：认证时延 vs t/n 比例 ----
        setup_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        xs = [r["ratio"] for r in rows_ratio]
        ys = [r["mean_auth_ms"] for r in rows_ratio]
        es = [r["std_auth_ms"] for r in rows_ratio]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=2)
        ax.set_xlabel("Threshold ratio t/n", fontsize=13)
        ax.set_ylabel("Mean FullAuth latency (ms)", fontsize=13)
        ax.set_title(f"FullAuth latency vs t/n ratio (n={n}, L={L})", fontsize=14)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "scale_auth_vs_ratio.png")

    # ================================================================
    # 实验3：n和L的交互影响
    # ================================================================
    scale_nL_cfg = exp.get("scale_n_L", {})
    if scale_nL_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp3: Vary n with multiple L values")
        logger.info("=" * 60)

        ratio = scale_nL_cfg.get("ratio", 0.5)
        L_values = scale_nL_cfg.get("L_values", [1, 2, 4, 6])
        n_values = scale_nL_cfg.get("n_values", [5, 10, 20, 30, 50, 70, 100])

        rows_nL: List[Dict[str, Any]] = []
        #遍历所有L和n的组合
        for L in tqdm(L_values, desc="Scale-nL"):
            for n in n_values:
                t_val = max(2, int(n * ratio))
                logger.info(f"  n={n}, t={t_val}, L={L}")

                sr = scheme_setup(t_val, n, L)
                auth_times = []
                sign_ok = 0
                retry_list = []

                for _ in range(num_trials):
                    tb = timed_full_auth(sr, t_val, L, Nmax)
                    auth_times.append(tb.full_auth)
                    if tb.verify > 0:
                        sign_ok += 1
                        retry_list.append(tb.retries)

                rows_nL.append({
                    "n": n,
                    "t": t_val,
                    "L": L,
                    "ratio": ratio,
                    "success_rate": sign_ok / num_trials,
                    "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
                    "mean_auth_ms": float(np.mean(auth_times)) * 1000,
                    "std_auth_ms": float(np.std(auth_times)) * 1000,
                })

        save_rows_csv(rows_nL, out / "tables" / "scalability_n_L_interaction.csv")

        # ---- 图3：多条L曲线 ----
        setup_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        colors_L = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        markers_L = ["o", "s", "^", "D"]

        for idx, L in enumerate(L_values):
            sub = [r for r in rows_nL if r["L"] == L]
            xs = [r["n"] for r in sub]
            ys = [r["mean_auth_ms"] for r in sub]
            es = [r["std_auth_ms"] for r in sub]
            c = colors_L[idx % len(colors_L)]
            m = markers_L[idx % len(markers_L)]
            ax.errorbar(xs, ys, yerr=es, marker=m, color=c, capsize=4,
                         linewidth=2, label=f"L={L}")

        ax.set_xlabel("Number of nodes n", fontsize=13)
        ax.set_ylabel("Mean FullAuth latency (ms)", fontsize=13)
        ax.set_title(f"FullAuth latency vs n for different L (t/n={ratio})", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "scale_auth_vs_n_multiL.png")

        # ---- 图3b：成功率 vs n for different L ----
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        for idx, L in enumerate(L_values):
            sub = [r for r in rows_nL if r["L"] == L]
            xs = [r["n"] for r in sub]
            ys = [r["success_rate"] for r in sub]
            c = colors_L[idx % len(colors_L)]
            m = markers_L[idx % len(markers_L)]
            ax2.plot(xs, ys, marker=m, color=c, linewidth=2, label=f"L={L}")

        ax2.set_xlabel("Number of nodes n", fontsize=13)
        ax2.set_ylabel("Success rate", fontsize=13)
        ax2.set_title(f"Success rate vs n for different L (t/n={ratio}, Nmax={Nmax})", fontsize=14)
        ax2.legend(fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        save_fig(fig2, out / "figures" / "scale_success_vs_n_multiL.png")

    # ================================================================
    # 实验4：与基线方案的规模对比
    # ================================================================
    scale_cmp_cfg = exp.get("scale_compare", {})
    if scale_cmp_cfg:
        logger.info("=" * 60)
        logger.info("Scalability Exp4: Comparison at scale")
        logger.info("=" * 60)

        ratio = scale_cmp_cfg.get("ratio", 0.5)
        L = scale_cmp_cfg.get("L", 2)
        n_values = scale_cmp_cfg.get("n_values", [5, 10, 20, 50, 100])
        trials_cmp = min(num_trials, 50)  # 对比实验少跑点

        rows_cmp: List[Dict[str, Any]] = []

        for n in tqdm(n_values, desc="Scale-compare"):
            t_val = max(2, int(n * ratio))
            logger.info(f"  Compare: n={n}, t={t_val}")

            sr = scheme_setup(t_val, n, L)
            participants = list(range(1, t_val + 1))
            node_keys = [(sr.shares[vi], sr.share_pks[vi]) for vi in participants]
            node_pks = [sr.share_pks[vi] for vi in participants]

            # ---- Per-node ----
            pn_times = []
            for _ in range(trials_cmp):
                m = secrets.token_bytes(32)
                t0 = time.perf_counter()
                sigs = per_node_sign(m, node_keys)
                per_node_verify(m, node_pks, sigs)
                pn_times.append(time.perf_counter() - t0)

            rows_cmp.append({
                "n": n, "t": t_val, "L": L,
                "scheme": "Per-Node",
                "mean_auth_ms": float(np.mean(pn_times)) * 1000,
                "comm_bytes": per_node_comm_cost(t_val, L),
            })

            # ---- Standard Threshold ----
            std_times = []
            for _ in range(trials_cmp):
                m = secrets.token_bytes(32)
                t0 = time.perf_counter()
                sig = threshold_schnorr_sign(m, participants, sr.shares, sr.share_pks, sr.pk)
                verify(m, sr.pk, sig)
                std_times.append(time.perf_counter() - t0)

            rows_cmp.append({
                "n": n, "t": t_val, "L": L,
                "scheme": "Std-Threshold",
                "mean_auth_ms": float(np.mean(std_times)) * 1000,
                "comm_bytes": threshold_schnorr_comm_cost(t_val, L),
            })

            # ---- Embedded (Ours) ----
            emb_times = []
            for _ in range(trials_cmp):
                m = secrets.token_bytes(32)
                M_val = secrets.randbelow(2 ** L)
                t0 = time.perf_counter()
                result = sign_emb(
                    m=m, participants=participants,
                    shares=sr.shares, share_pks=sr.share_pks,
                    pk=sr.pk, Kext=sr.Kext, M=M_val, L=L, Nmax=Nmax,
                    verify_partial=False,
                )
                if result.success:
                    verify(m, sr.pk, result.signature)
                emb_times.append(time.perf_counter() - t0)

            rows_cmp.append({
                "n": n, "t": t_val, "L": L,
                "scheme": "Embedded(Ours)",
                "mean_auth_ms": float(np.mean(emb_times)) * 1000,
                "comm_bytes": embedded_threshold_comm_cost(t_val, L),
            })

        save_rows_csv(rows_cmp, out / "tables" / "scalability_comparison.csv")

        # ---- 图4a：各方案认证时延 vs n ----
        setup_style()
        schemes = ["Per-Node", "Std-Threshold", "Embedded(Ours)"]
        sch_colors = ["tab:blue", "tab:orange", "tab:red"]
        sch_markers = ["o", "s", "D"]

        fig, ax = plt.subplots(figsize=(10, 7))
        for sch, clr, mkr in zip(schemes, sch_colors, sch_markers):
            sub = [r for r in rows_cmp if r["scheme"] == sch]
            xs = [r["n"] for r in sub]
            ys = [r["mean_auth_ms"] for r in sub]
            ax.plot(xs, ys, marker=mkr, color=clr, linewidth=2, markersize=8, label=sch)

        ax.set_xlabel("Number of nodes n", fontsize=13)
        ax.set_ylabel("Mean auth latency (ms)", fontsize=13)
        ax.set_title(f"Scheme comparison: latency vs n (t/n={ratio}, L={L})", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "scale_compare_latency.png")

        # ---- 图4b：各方案通信量 vs n ----
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        for sch, clr, mkr in zip(schemes, sch_colors, sch_markers):
            sub = [r for r in rows_cmp if r["scheme"] == sch]
            xs = [r["n"] for r in sub]
            ys = [r["comm_bytes"] for r in sub]
            ax2.plot(xs, ys, marker=mkr, color=clr, linewidth=2, markersize=8, label=sch)

        ax2.set_xlabel("Number of nodes n", fontsize=13)
        ax2.set_ylabel("Communication (bytes)", fontsize=13)
        ax2.set_title(f"Scheme comparison: communication vs n (t/n={ratio}, L={L})", fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        save_fig(fig2, out / "figures" / "scale_compare_comm.png")

    logger.info("Scalability experiment complete.")