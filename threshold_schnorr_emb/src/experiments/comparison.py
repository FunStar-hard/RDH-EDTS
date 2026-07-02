"""Experiment F: Scheme comparison."""
from __future__ import annotations
#对比四种方案：逐节点签名、标准门限 Schnorr、门限 Schnorr + 额外字段、我的方案。比较它们在通信成本、签名和验证延迟等方面的性能。
import logging
import math
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify as scheme_verify
from src.scheme.extract import extract as scheme_extract
from src.scheme.baselines import (#导入四种方案的签名、验证和通信成本计算函数
    per_node_sign, per_node_verify, per_node_comm_cost,
    threshold_schnorr_sign, threshold_schnorr_comm_cost,
    threshold_schnorr_extra_sign, threshold_schnorr_extra_comm_cost,
    embedded_threshold_comm_cost,
)
from src.crypto.curve_utils import get_order
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_comparison(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("comparison", {})
    n = exp.get("n", 10)
    t = exp.get("t", 5)
    Ls = exp.get("L_values", [2, 4, 6])
    Nmax = exp.get("Nmax", 2048)
    num_trials = exp.get("num_trials", 50)

    comp_rows: List[Dict[str, Any]] = []

    for L in tqdm(Ls, desc="Comparison"):#遍历不同的嵌入位数 L，比较四种方案的性能
        sr = scheme_setup(t, n, L)
        participants = list(range(1, t + 1))

        # 计算每种方案的通信成本（以字节为单位）
        per_node_cc = per_node_comm_cost(t, L)
        std_thresh_cc = threshold_schnorr_comm_cost(t, L)
        extra_thresh_cc = threshold_schnorr_extra_comm_cost(t, L)
        emb_thresh_cc = embedded_threshold_comm_cost(t, L)

        #逐节点签名计时：运行 num_trials 次签名和验证过程，记录平均签名时间和验证时间。对于逐节点签名，生成 n 个独立的 Schnorr 签名，并验证它们的正确性。
        pn_sign_times = []
        pn_ver_times = []
        node_keys = [(sr.shares[vi], sr.share_pks[vi]) for vi in participants]
        node_pks = [sr.share_pks[vi] for vi in participants]

        for _ in range(num_trials):
            m = secrets.token_bytes(32)
            t0 = time.perf_counter()#t0 记录签名开始时间，调用 per_node_sign 函数生成每个节点的 Schnorr 签名，传入消息 m 和节点密钥列表 node_keys，返回所有节点的签名列表 sigs，并记录签名耗时；然后调用 per_node_verify 函数验证每个节点的签名，传入消息 m、节点公钥列表 node_pks 和签名列表 sigs，并记录验证耗时
            sigs = per_node_sign(m, node_keys)#对于每个节点，生成一个随机的 Schnorr 签名，包含一个承诺 R 和一个响应 s，返回所有节点的签名列表
            pn_sign_times.append(time.perf_counter() - t0)#对于每个节点的公钥和签名，计算挑战 c，并验证 Schnorr 签名的正确性，如果有任何一个签名验证失败，则返回 False；如果所有签名都验证成功，则返回 True
            t0 = time.perf_counter()
            per_node_verify(m, node_pks, sigs)
            pn_ver_times.append(time.perf_counter() - t0)

        # ---- Standard threshold Schnorr ----
        std_sign_times = []
        std_ver_times = []
        for _ in range(num_trials):
            m = secrets.token_bytes(32)
            t0 = time.perf_counter()
            sig = threshold_schnorr_sign(m, participants, sr.shares, sr.share_pks, sr.pk)
            std_sign_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            scheme_verify(m, sr.pk, sig)
            std_ver_times.append(time.perf_counter() - t0)

        # ---- Threshold + Extra ----
        ext_sign_times = []
        ext_ver_times = []
        for _ in range(num_trials):
            m = secrets.token_bytes(32)
            M_val = secrets.randbelow(2 ** L)
            t0 = time.perf_counter()
            sig, ef = threshold_schnorr_extra_sign(
                m, participants, sr.shares, sr.share_pks, sr.pk, sr.Kext, M_val, L,
            )
            ext_sign_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            scheme_verify(m, sr.pk, sig)
            ext_ver_times.append(time.perf_counter() - t0)

        # ---- Embedded (ours) ----
        emb_sign_times = []
        emb_ver_times = []
        for _ in range(num_trials):
            m = secrets.token_bytes(32)
            M_val = secrets.randbelow(2 ** L)
            t0 = time.perf_counter()
            result = sign_emb(
                m=m, participants=participants,
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M_val, L=L, Nmax=Nmax,
                verify_partial=False,
            )
            emb_sign_times.append(time.perf_counter() - t0)
            if result.success:
                t0 = time.perf_counter()
                scheme_verify(m, sr.pk, result.signature)
                emb_ver_times.append(time.perf_counter() - t0)
            else:
                emb_ver_times.append(0.0)
        #汇总结果，记录每种方案的通信成本、平均签名时间和平均验证时间等指标，并保存到 CSV 文件中
        comp_rows.append({
            "L": L, "n": n, "t": t,
            "scheme": "Per-Node",
            "comm_bytes": per_node_cc,
            "sign_ms": float(np.mean(pn_sign_times)) * 1000,
            "verify_ms": float(np.mean(pn_ver_times)) * 1000,
        })
        comp_rows.append({
            "L": L, "n": n, "t": t,
            "scheme": "Std-Threshold",
            "comm_bytes": std_thresh_cc,
            "sign_ms": float(np.mean(std_sign_times)) * 1000,
            "verify_ms": float(np.mean(std_ver_times)) * 1000,
        })
        comp_rows.append({
            "L": L, "n": n, "t": t,
            "scheme": "Threshold+Extra",
            "comm_bytes": extra_thresh_cc,
            "sign_ms": float(np.mean(ext_sign_times)) * 1000,
            "verify_ms": float(np.mean(ext_ver_times)) * 1000,
        })
        comp_rows.append({
            "L": L, "n": n, "t": t,
            "scheme": "Embedded(Ours)",
            "comm_bytes": emb_thresh_cc,
            "sign_ms": float(np.mean(emb_sign_times)) * 1000,
            "verify_ms": float(np.mean(emb_ver_times)) * 1000,
        })

    save_rows_csv(comp_rows, out / "tables" / "comparison_summary.csv")

    # ---- Plots ----
    setup_style()
    schemes = ["Per-Node", "Std-Threshold", "Threshold+Extra", "Embedded(Ours)"]
    markers = ["o", "s", "^", "D"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for metric, ylabel, fname in [
        ("comm_bytes", "Communication (bytes)", "comparison_comm"),
        ("sign_ms", "Signing latency (ms)", "comparison_sign"),
        ("verify_ms", "Verify latency (ms)", "comparison_verify"),
    ]:
        fig, ax = plt.subplots()
        for sch, mkr, clr in zip(schemes, markers, colors):
            xs = [r["L"] for r in comp_rows if r["scheme"] == sch]
            ys = [r[metric] for r in comp_rows if r["scheme"] == sch]
            if xs:
                ax.plot(xs, ys, marker=mkr, color=clr, label=sch)
        ax.set_xlabel("Embedding bits L")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs L (n={n}, t={t})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / f"{fname}.png")

    logger.info("Comparison experiment complete.")