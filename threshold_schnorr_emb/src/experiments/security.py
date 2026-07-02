from __future__ import annotations
# 1.低位均匀性实验；2.掉线实验；3.低于阈值的合谋伪造实验

import logging
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.common import single_trial
from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.types import Signature
from src.scheme.lagrange import all_lagrange_coefficients
from src.crypto.curve_utils import (
    get_order,
    get_generator,
    random_scalar,
    scalar_mult,
    point_sum,
)
from src.crypto.hash_prf import compute_challenge
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig


def run_security(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("security", {})

    # =========================================================
    # 低位均匀性实验
    # =========================================================
    logger.info("Security Part 1: Low-bit uniformity")

    # 参数设置
    unif_cfg = exp.get("uniformity", {})
    n = unif_cfg.get("n", 5)
    t = unif_cfg.get("t", 3)
    Nmax = unif_cfg.get("Nmax", 4096)
    Ls = unif_cfg.get("L_values", [1, 2, 4, 6])
    base_samples = unif_cfg.get("num_samples", 5000)

    # 汇总统计结果：每个 L 一行
    unif_rows: List[Dict[str, Any]] = []

    # 原始 pattern 统计结果：每个 L 下的每个 pattern 一行
    pattern_rows: List[Dict[str, Any]] = []

    # 缓存同一轮实验数据，用于后续生成整体 grid 图
    # 这样不会为了画图重新跑第二轮实验
    uniformity_plot_data: Dict[int, Dict[str, Any]] = {}

    setup_style()

    # =========================================================
      # 这一轮同时完成：
    # 1. 采样
    # 2. 统计 chi2 / p-value / entropy
    # 3. 保存每个 pattern 的 count / empirical_probability
    # 4. 缓存数据用于整体 Figure 5
    # =========================================================
    for L in tqdm(Ls, desc="Uniformity"):
        target_samples = int(base_samples * (2 ** max(L - 2, 0)))
        #调用scheme_setup函数生成预先的密钥和共享信息，传入参数t、n和L，返回一个SetupResult对象sr
        sr = scheme_setup(t, n, L)
        counts = Counter()
        collected = 0
        attempts = 0
        max_attempts = target_samples * 20
        #若成功生成签名的次数小于目标样本数且尝试次数小于最大尝试次数，则继续进行单次试验
        while collected < target_samples and attempts < max_attempts:
            attempts += 1
            row = single_trial(sr, t, L, Nmax)

            if row["success"] and "s_low_bits" in row:
                counts[row["s_low_bits"]] += 1
                collected += 1
        #统计结果分析：计算 chi2 统计量和 p-value，计算经验概率分布，计算经验熵和熵比率，并保存每个 pattern 的统计数据
        num_bins = 2 ** L
        observed = [counts.get(i, 0) for i in range(num_bins)]# observed 是一个长度为 num_bins 的列表，表示每个低位模式的出现次数，使用 counts.get(i, 0) 来获取每个模式的计数，如果某个模式没有出现则默认为 0
        
        if collected > 0:
            expected = [collected / num_bins] * num_bins# expected 是一个长度为 num_bins 的列表，表示每个低位模式的期望出现次数，假设分布是均匀的，则每个模式的期望计数应该是 collected / num_bins
            chi2_stat, chi2_p = stats.chisquare(observed, expected)
        else:
            chi2_stat, chi2_p = 0.0, 1.0

        probs = np.array(observed, dtype=float)
        if collected > 0:
            probs /= collected

        probs_nonzero = probs[probs > 0]

        emp_entropy = (
            -np.sum(probs_nonzero * np.log2(probs_nonzero))
            if len(probs_nonzero) > 0
            else 0.0
        )

        entropy_ratio = emp_entropy / L if L > 0 else 0.0

        # 保存每个 L 的汇总统计结果
        unif_rows.append({
            "L": L,
            "samples": collected,
            "attempts": attempts,
            "num_bins": num_bins,
            "chi2": chi2_stat,
            "p_value": chi2_p,
            "entropy": emp_entropy,
            "entropy_ratio": entropy_ratio,
        })

        # =====================================================
        # 保存每个 pattern 的原始统计数据
        # 这就是 Origin 画图需要的 CSV
        # =====================================================
        uniform_probability = 1.0 / num_bins
        expected_count = collected / num_bins if collected > 0 else 0.0

        for pattern, count in enumerate(observed):
            empirical_probability = count / collected if collected > 0 else 0.0

            pattern_rows.append({
                "L": L,
                "samples": collected,
                "pattern": pattern,
                "count": count,
                "empirical_probability": empirical_probability,
                "uniform_probability": uniform_probability,
                "expected_count": expected_count,
                "difference": empirical_probability - uniform_probability,
                "ratio_to_uniform": (
                    empirical_probability / uniform_probability
                    if uniform_probability > 0
                    else 0.0
                ),
            })

        # 缓存当前 L 的数据，用于后面生成整体 grid 图
        uniformity_plot_data[L] = {
            "collected": collected,
            "num_bins": num_bins,
            "observed": observed,
        }

        logger.info(
            "L=%s, samples=%s, attempts=%s, chi2=%.6f, p=%.6f, entropy_ratio=%.8f",
            L,
            collected,
            attempts,
            chi2_stat,
            chi2_p,
            entropy_ratio,
        )

    # 保存汇总统计 CSV
    save_rows_csv(
        unif_rows,
        out / "tables" / "security_uniformity.csv"
    )

    # 保存每个 pattern 的 count / probability CSV
    save_rows_csv(
        pattern_rows,
        out / "tables" / "security_uniformity_patterns.csv"
    )

    # =========================================================
    # 论文级整体 Grid 图
    # =========================================================
    logger.info("Plotting grid figure from one-round uniformity data")

    num_L = len(Ls)
    cols = 2
    rows = (num_L + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))

    if rows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    last_idx = -1

    for idx, L in enumerate(Ls):
        last_idx = idx

        data = uniformity_plot_data[L]
        collected = data["collected"]
        num_bins = data["num_bins"]
        observed = data["observed"]

        ax = axes[idx]

        if collected > 0:
            x = np.arange(num_bins)
            empirical_probs = [o / collected for o in observed]
            uniform_probs = [1 / num_bins] * num_bins

            ax.bar(
                x,
                empirical_probs,
                width=0.8,
                alpha=0.8,
                label="Empirical"
            )

            ax.plot(
                x,
                uniform_probs,
                linestyle="--",
                marker="o",
                label="Uniform"
            )

        ax.set_title(f"L = {L}")
        ax.set_xlabel("Low-bit pattern")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend()

    for j in range(last_idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Low-bit Pattern Uniformity Across Different L", fontsize=14)
    plt.tight_layout()

    save_fig(
        fig,
        out / "figures" / "security_uniformity_grid.png"
    )

    # =========================================================
    # Part 2: 掉线容忍实验
    # =========================================================
    logger.info("Security Part 2: Dropout")

    drop_trials = 500
    drop_rows = []

    sr = scheme_setup(3, 5, 2)  # n=5，t=3，L=2

    for online in range(3, 6):  # 在线参与者数量从 t=3 到 n=5 变化，测试掉线容忍能力
        participants = list(range(1, online + 1))  # 参与者 ID 从 1 到 online
        ok = 0  # 成功计数器

        for _ in range(drop_trials):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(4)

            res = sign_emb(
                m,
                participants,
                sr.shares,
                sr.share_pks,
                sr.pk,
                sr.Kext,
                M,
                2,
                256
            )

            if res.success and verify(m, sr.pk, res.signature):
                ok += 1

        drop_rows.append({
            "online": online,
            "trials": drop_trials,
            "success": ok,
            "rate": ok / drop_trials if drop_trials > 0 else 0.0,
        })

    save_rows_csv(
        drop_rows,
        out / "tables" / "security_dropout.csv"
    )

    # =========================================================
    # Part 3: 低于阈值的合谋伪造实验
    # =========================================================
    logger.info("Security Part 3: Below-threshold collusion forgery")

    forge_cfg = exp.get("forgery", {})
    forge_combos = forge_cfg.get("combos", [
        {"n": 5, "t": 3},
        {"n": 5, "t": 5},
        {"n": 10, "t": 3},
        {"n": 10, "t": 5},
    ])
    forge_trials = forge_cfg.get("num_trials", 200)
    forge_L = forge_cfg.get("L", 2)

    forge_rows: List[Dict[str, Any]] = []
    q = get_order()
    G = get_generator()

    for combo in tqdm(forge_combos, desc="Forgery"):
        fn = combo["n"]
        ft = combo["t"]
        if ft > fn:
            continue

        # k_prime 表示合谋者数量。这里只测试 k_prime < t 的低于阈值场景。
        for k_prime in range(1, ft):
            sr = scheme_setup(ft, fn, forge_L)
            forge_attempts = 0
            forge_verify_pass = 0

            colluding = list(range(1, k_prime + 1))
            missing = list(range(k_prime + 1, ft + 1))
            all_parts = list(range(1, ft + 1))

            # 使用 t 个参与者对应的 Lagrange 系数，模拟攻击者尝试补齐缺失份额。
            lambdas = all_lagrange_coefficients(all_parts, q)

            for _ in range(forge_trials):
                m = secrets.token_bytes(32)

                # 合谋节点生成真实 nonce 和承诺。
                nonces = {}
                comms = {}
                for vi in colluding:
                    ki = random_scalar()
                    nonces[vi] = ki
                    comms[vi] = scalar_mult(ki, G)

                # 缺失节点没有真实私钥份额，攻击者只能伪造随机承诺。
                for vi in missing:
                    ki = random_scalar()
                    nonces[vi] = ki
                    comms[vi] = scalar_mult(ki, G)

                R = point_sum([comms[vi] for vi in all_parts])
                c = compute_challenge(R, m, sr.pk)

                # 合谋节点可以计算真实 partial signature。
                s_value = 0
                for vi in colluding:
                    si = (nonces[vi] + c * lambdas[vi] * sr.shares[vi]) % q
                    s_value = (s_value + si) % q

                # 对于缺失节点，攻击者只能随机猜测 partial signature。
                for _vi in missing:
                    fake_si = random_scalar()
                    s_value = (s_value + fake_si) % q
                    forge_attempts += 1

                fake_sig = Signature(R=R, s=s_value)
                if verify(m, sr.pk, fake_sig):
                    forge_verify_pass += 1

            forge_rows.append({
                "n": fn,
                "t": ft,
                "L": forge_L,
                "k_prime": k_prime,
                "missing_shares": ft - k_prime,
                "num_trials": forge_trials,
                "forge_attempts": forge_attempts,
                "forge_verify_pass": forge_verify_pass,
                "forge_rate": forge_verify_pass / forge_trials if forge_trials > 0 else 0.0,
            })

            logger.info(
                "n=%s,t=%s,k'=%s: forge_pass=%s/%s",
                fn,
                ft,
                k_prime,
                forge_verify_pass,
                forge_trials,
            )

    save_rows_csv(
        forge_rows,
        out / "tables" / "security_forgery.csv"
    )

    logger.info("Security experiment complete.")