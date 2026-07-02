from __future__ import annotations

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
from src.crypto.curve_utils import get_order, get_generator, random_scalar, scalar_mult
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig


def run_security(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("security", {})

    # =========================================================
    # Part 1: Low-bit uniformity
    # =========================================================
    logger.info("Security Part 1: Low-bit uniformity")

    unif_cfg = exp.get("uniformity", {})
    n = unif_cfg.get("n", 5)
    t = unif_cfg.get("t", 3)
    Nmax = unif_cfg.get("Nmax", 4096)
    Ls = unif_cfg.get("L_values", [1, 2, 4, 6])
    base_samples = unif_cfg.get("num_samples", 5000)

    unif_rows: List[Dict[str, Any]] = []
    # Detailed per-pattern rows for Origin:
    # each row corresponds to one low-bit pattern under one L.
    pattern_rows: List[Dict[str, Any]] = []

    # Cache the first uniformity experiment results so the grid figure is drawn
    # from exactly the same samples that are exported to CSV.
    uniformity_plot_data: Dict[int, Dict[str, Any]] = {}

    setup_style()

    # ===== 单图 + 数据统计 =====
    for L in tqdm(Ls, desc="Uniformity"):

        target_samples = int(base_samples * (2 ** max(L - 2, 0)))

        sr = scheme_setup(t, n, L)
        counts = Counter()
        collected = 0
        attempts = 0
        max_attempts = target_samples * 20

        while collected < target_samples and attempts < max_attempts:
            attempts += 1
            row = single_trial(sr, t, L, Nmax)
            if row["success"] and "s_low_bits" in row:
                counts[row["s_low_bits"]] += 1
                collected += 1

        num_bins = 2 ** L
        observed = [counts.get(i, 0) for i in range(num_bins)]

        if collected > 0:
            expected = [collected / num_bins] * num_bins
            chi2_stat, chi2_p = stats.chisquare(observed, expected)
        else:
            chi2_stat, chi2_p = 0.0, 1.0

        probs = np.array(observed, dtype=float)
        if collected > 0:
            probs /= collected
        probs = probs[probs > 0]

        emp_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

        unif_rows.append({
            "L": L,
            "samples": collected,
            "chi2": chi2_stat,
            "p_value": chi2_p,
            "entropy": emp_entropy,
            "entropy_ratio": emp_entropy / L if L > 0 else 0.0,
        })

        # Save detailed per-pattern counts/probabilities for Origin.
        # Output columns:
        #   L: observed low-bit length
        #   pattern: integer pattern index, ranging from 0 to 2^L - 1
        #   count: number of times this pattern appears
        #   empirical_probability: count / collected
        #   uniform_probability: theoretical probability 1 / 2^L
        #   expected_count: collected / 2^L
        #   difference: empirical_probability - uniform_probability
        #   ratio_to_uniform: empirical_probability / uniform_probability
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

        # Cache data for the grid figure so it matches the exported CSV exactly.
        uniformity_plot_data[L] = {
            "collected": collected,
            "num_bins": num_bins,
            "observed": observed,
        }

        logger.info(f"L={L}, samples={collected}, p={chi2_p:.4f}")

        # ===== 单独图 =====
        if collected > 0:
            fig, ax = plt.subplots()
            x = np.arange(num_bins)
            w = 0.35

            ax.bar(x - w/2, [o/collected for o in observed], w, label="Empirical")
            ax.bar(x + w/2, [1/num_bins]*num_bins, w, label="Uniform")

            ax.set_title(f"L={L}")
            ax.set_xlabel("pattern")
            ax.set_ylabel("probability")
            ax.legend()
            ax.grid(True)

            save_fig(fig, out / "figures" / f"security_uniformity_L{L}.png")

    save_rows_csv(unif_rows, out / "tables" / "security_uniformity.csv")
    save_rows_csv(
        pattern_rows,
        out / "tables" / "security_uniformity_patterns.csv"
    )

    # =========================================================
    # 📊 论文级 Grid 图
    # =========================================================
    logger.info("Plotting grid figure")

    num_L = len(Ls)
    cols = 2
    rows = (num_L + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))

    if rows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for idx, L in enumerate(Ls):

        data = uniformity_plot_data[L]
        collected = data["collected"]
        num_bins = data["num_bins"]
        observed = data["observed"]

        ax = axes[idx]

        if collected > 0:
            x = np.arange(num_bins)

            ax.bar(
                x,
                [o / collected for o in observed],
                width=0.8,
                alpha=0.8,
                label="Empirical"
            )

            ax.plot(
                x,
                [1 / num_bins] * num_bins,
                linestyle="--",
                marker="o",
                label="Uniform"
            )

        ax.set_title(f"L = {L}")
        ax.set_xlabel("Pattern")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend()

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Low-bit Uniformity Across Different L", fontsize=14)

    plt.tight_layout()

    save_fig(fig, out / "figures" / "security_uniformity_grid.png")

    # =========================================================
    # Part 2: Dropout
    # =========================================================
    logger.info("Security Part 2: Dropout")

    drop_trials = 500
    drop_rows = []

    sr = scheme_setup(3, 5, 2)

    for online in range(3, 6):
        participants = list(range(1, online + 1))
        ok = 0

        for _ in range(drop_trials):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(4)

            res = sign_emb(
                m, participants,
                sr.shares, sr.share_pks,
                sr.pk, sr.Kext,
                M, 2, 256
            )

            if res.success and verify(m, sr.pk, res.signature):
                ok += 1

        drop_rows.append({
            "online": online,
            "rate": ok / drop_trials
        })

    save_rows_csv(drop_rows, out / "tables" / "security_dropout.csv")

    # =========================================================
    # Part 3: Forgery
    # =========================================================
    logger.info("Security Part 3: Forgery")

    forge_trials = 1000
    forge_pass = 0

    sr = scheme_setup(3, 5, 2)
    q = get_order()
    G = get_generator()

    for _ in range(forge_trials):
        m = secrets.token_bytes(32)

        fake_sig = Signature(
            R=scalar_mult(random_scalar(), G),
            s=random_scalar()
        )

        if verify(m, sr.pk, fake_sig):
            forge_pass += 1

    save_rows_csv([{
        "forge_rate": forge_pass / forge_trials
    }], out / "tables" / "security_forgery.csv")

    logger.info("Security experiment complete.")