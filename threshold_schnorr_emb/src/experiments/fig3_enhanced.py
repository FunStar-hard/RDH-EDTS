"""Figure 3 enhanced: retry statistics, distribution, tail, success rate vs Nmax."""
from __future__ import annotations

import logging
import math
import secrets
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.experiments.common import geometric_conditional_expectation
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_fig3_enhanced(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    """Run all Figure 3 enhanced experiments."""
    exp = cfg.get("correctness", {})
    fig3 = exp.get("fig3", {})

    n = fig3.get("n", 5)
    t = fig3.get("t", 3)

    # ================================================================
    # Fig 3A: Retry statistics with different Nmax
    # ================================================================
    logger.info("Fig3A: Retry statistics")
    Ls_A = fig3.get("Ls_A", [1, 2, 3, 4, 5, 6, 7, 8])
    Nmaxs_A = fig3.get("Nmaxs_A", [256, 2048])
    trials_A = fig3.get("trials_A", 500)

    rows_3a: List[Dict[str, Any]] = []

    for Nmax in Nmaxs_A:
        for L in tqdm(Ls_A, desc=f"Fig3A Nmax={Nmax}"):
            sr = scheme_setup(t, n, L)
            participants = list(range(1, t + 1))
            retries_list = []

            for _ in range(trials_A):
                m = secrets.token_bytes(32)
                M = secrets.randbelow(2 ** L)
                result = sign_emb(
                    m=m, participants=participants,
                    shares=sr.shares, share_pks=sr.share_pks,
                    pk=sr.pk, Kext=sr.Kext, M=M, L=L, Nmax=Nmax,
                )
                if result.success:
                    retries_list.append(result.retries)

            arr = np.array(retries_list) if retries_list else np.array([])
            p = 2 ** (-L)
            theory_uncond = 2 ** L
            theory_cond = geometric_conditional_expectation(p, Nmax)

            mean_val = float(np.mean(arr)) if len(arr) > 0 else float("nan")
            std_val = float(np.std(arr)) if len(arr) > 0 else float("nan")
            ci95 = 1.96 * std_val / math.sqrt(len(arr)) if len(arr) > 1 else 0.0

            rows_3a.append({
                "L": L, "Nmax": Nmax, "n": n, "t": t,
                "num_success": len(retries_list),
                "num_trials": trials_A,
                "empirical_mean": mean_val,
                "empirical_std": std_val,
                "ci95": ci95,
                "theory_uncond": theory_uncond,
                "theory_cond": theory_cond,
            })

    save_rows_csv(rows_3a, out / "tables" / "fig3a_retry_stats.csv")

    # Plot Fig3A
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for Nmax in Nmaxs_A:
        sub = [r for r in rows_3a if r["Nmax"] == Nmax]
        xs = [r["L"] for r in sub]
        ys = [r["empirical_mean"] for r in sub]
        errs = [r["ci95"] for r in sub]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4,
                     label=f"Empirical mean (Nmax={Nmax})")

        ys_cond = [r["theory_cond"] for r in sub]
        ax.plot(xs, ys_cond, "--", label=f"E[X|X≤{Nmax}] (theory)")

    # Unconditional
    xs_all = Ls_A
    ys_uncond = [2 ** L for L in xs_all]
    ax.plot(xs_all, ys_uncond, "k-.", linewidth=2, label="E[X]=2^L (unconditional)")

    ax.set_xlabel("Embedding bits L")
    ax.set_ylabel("Mean retries")
    ax.set_title(f"Fig 3A: Mean retries vs L (n={n}, t={t})")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out / "figures" / "fig3a_retry_vs_L.png")

    # ================================================================
    # Fig 3B: Retry distribution (boxplot)
    # ================================================================
    logger.info("Fig3B: Retry distribution boxplot")
    Ls_B = fig3.get("Ls_B", [2, 4, 6, 8])
    Nmax_B = fig3.get("Nmax_B", 2048)
    trials_B = fig3.get("trials_B", 500)

    box_data = {}
    for L in tqdm(Ls_B, desc="Fig3B"):
        sr = scheme_setup(t, n, L)
        participants = list(range(1, t + 1))
        retries = []
        for _ in range(trials_B):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(2 ** L)
            result = sign_emb(
                m=m, participants=participants,
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=L, Nmax=Nmax_B,
            )
            if result.success:
                retries.append(result.retries)
        box_data[L] = retries

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    data_list = [box_data.get(L, []) for L in Ls_B]
    bp = ax2.boxplot(data_list, labels=[f"L={L}" for L in Ls_B],
                      patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_xlabel("Embedding bits L")
    ax2.set_ylabel("Retries")
    ax2.set_title(f"Fig 3B: Retry distribution (n={n}, t={t}, Nmax={Nmax_B})")
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, out / "figures" / "fig3b_retry_boxplot.png")

    # ================================================================
    # Fig 3C: Tail statistics
    # ================================================================
    logger.info("Fig3C: Tail statistics")
    Ls_C = fig3.get("Ls_C", [1, 2, 3, 4, 5, 6, 7, 8])
    Nmax_C = fig3.get("Nmax_C", 2048)
    trials_C = fig3.get("trials_C", 500)

    rows_3c: List[Dict[str, Any]] = []
    for L in tqdm(Ls_C, desc="Fig3C"):
        sr = scheme_setup(t, n, L)
        participants = list(range(1, t + 1))
        retries = []
        for _ in range(trials_C):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(2 ** L)
            result = sign_emb(
                m=m, participants=participants,
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=L, Nmax=Nmax_C,
            )
            if result.success:
                retries.append(result.retries)

        arr = np.array(retries) if retries else np.array([0])
        rows_3c.append({
            "L": L,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        })

    save_rows_csv(rows_3c, out / "tables" / "fig3c_tail_stats.csv")

    fig3c, ax3c = plt.subplots(figsize=(10, 7))
    xs = [r["L"] for r in rows_3c]
    for metric, style in [
        ("mean", "o-"),
        ("median", "s--"),
        ("p90", "^:"),
        ("p95", "v-."),
        ("p99", "D-"),
    ]:
        ys = [r[metric] for r in rows_3c]
        ax3c.plot(xs, ys, style, label=metric)
    ax3c.set_xlabel("Embedding bits L")
    ax3c.set_ylabel("Retries")
    ax3c.set_title(f"Fig 3C: Tail statistics (n={n}, t={t}, Nmax={Nmax_C})")
    ax3c.legend()
    ax3c.set_yscale("log")
    ax3c.grid(True, alpha=0.3)
    save_fig(fig3c, out / "figures" / "fig3c_tail_stats.png")

    # ================================================================
    # Fig: Success rate vs Nmax
    # ================================================================
    logger.info("Success rate vs Nmax")
    sr_cfg = fig3.get("success_rate", {})
    sr_Ls = sr_cfg.get("L_values", [2, 4, 6, 8])
    sr_Nmaxs = sr_cfg.get("Nmax_values", [16, 32, 64, 128, 256, 512, 1024, 2048])
    sr_trials = sr_cfg.get("num_trials", 300)

    rows_sr: List[Dict[str, Any]] = []

    for L in tqdm(sr_Ls, desc="SuccRate vs Nmax"):
        sr_setup = scheme_setup(t, n, L)
        participants = list(range(1, t + 1))
        p = 2 ** (-L)

        for Nmax in sr_Nmaxs:
            successes = 0
            for _ in range(sr_trials):
                m = secrets.token_bytes(32)
                M = secrets.randbelow(2 ** L)
                result = sign_emb(
                    m=m, participants=participants,
                    shares=sr_setup.shares, share_pks=sr_setup.share_pks,
                    pk=sr_setup.pk, Kext=sr_setup.Kext, M=M, L=L, Nmax=Nmax,
                )
                if result.success:
                    successes += 1

            theory_ps = 1 - (1 - p) ** Nmax
            rows_sr.append({
                "L": L, "Nmax": Nmax,
                "empirical_rate": successes / sr_trials,
                "theory_rate": theory_ps,
                "num_trials": sr_trials,
            })

    save_rows_csv(rows_sr, out / "tables" / "success_rate_vs_Nmax.csv")

    fig_sr, ax_sr = plt.subplots(figsize=(10, 7))
    colors_sr = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, L in enumerate(sr_Ls):
        sub = [r for r in rows_sr if r["L"] == L]
        xs = [r["Nmax"] for r in sub]
        ys_emp = [r["empirical_rate"] for r in sub]
        ys_th = [r["theory_rate"] for r in sub]
        c = colors_sr[idx % len(colors_sr)]
        ax_sr.plot(xs, ys_emp, "o-", color=c, label=f"L={L} (empirical)")
        ax_sr.plot(xs, ys_th, "--", color=c, label=f"L={L} (theory)")

    ax_sr.set_xlabel("Nmax")
    ax_sr.set_ylabel("Success rate")
    ax_sr.set_title(f"Success rate vs Nmax (n={n}, t={t})")
    ax_sr.legend(fontsize=9)
    ax_sr.set_xscale("log", base=2)
    ax_sr.grid(True, alpha=0.3)
    ax_sr.set_ylim(-0.05, 1.05)
    save_fig(fig_sr, out / "figures" / "success_rate_vs_Nmax.png")

    logger.info("Fig3 enhanced experiments complete.")