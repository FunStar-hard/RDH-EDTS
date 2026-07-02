"""Experiment D: Security experiments."""
from __future__ import annotations

import logging
import math
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from tqdm import tqdm

from src.config import get_scheme_params
from src.experiments.common import single_trial, geometric_conditional_expectation
from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.types import Signature
from src.crypto.curve_utils import get_order, random_scalar, scalar_mult, get_generator
from src.scheme.lagrange import all_lagrange_coefficients
from src.utils.io_utils import save_rows_csv
from src.utils.plot_utils import setup_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    Ls = unif_cfg.get("L_values", [2, 3, 4, 5, 6])
    num_samples = unif_cfg.get("num_samples", 2000)

    unif_rows: List[Dict[str, Any]] = []

    for L in tqdm(Ls, desc="Uniformity"):
        sr = scheme_setup(t, n, L)
        counts = Counter()
        collected = 0
        attempts = 0
        max_attempts = num_samples * (2 ** L) * 3  # generous upper bound

        while collected < num_samples and attempts < max_attempts:
            attempts += 1
            row = single_trial(sr, t, L, Nmax)
            if row["success"] and "s_low_bits" in row:
                counts[row["s_low_bits"]] += 1
                collected += 1

        num_bins = 2 ** L
        expected = collected / num_bins
        observed = [counts.get(i, 0) for i in range(num_bins)]

        chi2_stat, chi2_p = stats.chisquare(observed, [expected] * num_bins)

        # empirical entropy
        probs = np.array(observed, dtype=float) / collected if collected > 0 else np.zeros(num_bins)
        probs = probs[probs > 0]
        emp_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

        unif_rows.append({
            "L": L, "n": n, "t": t, "Nmax": Nmax,
            "num_samples": collected,
            "num_bins": num_bins,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "empirical_entropy": emp_entropy,
            "max_entropy": float(L),
            "entropy_ratio": emp_entropy / L if L > 0 else 0.0,
        })
        logger.info(f"  L={L}: chi2={chi2_stat:.2f}, p={chi2_p:.4f}, "
                     f"entropy={emp_entropy:.4f}/{L}")

    save_rows_csv(unif_rows, out / "tables" / "security_uniformity.csv")

    # Plot L=4 distribution
    setup_style()
    if 4 in Ls:
        sr4 = scheme_setup(t, n, 4)
        counts4 = Counter()
        collected4 = 0
        att4 = 0
        while collected4 < num_samples and att4 < num_samples * 50:
            att4 += 1
            row = single_trial(sr4, t, 4, Nmax)
            if row["success"] and "s_low_bits" in row:
                counts4[row["s_low_bits"]] += 1
                collected4 += 1

        num_bins4 = 16
        obs4 = [counts4.get(i, 0) for i in range(num_bins4)]
        ideal4 = [collected4 / num_bins4] * num_bins4

        fig, ax = plt.subplots()
        x = np.arange(num_bins4)
        w = 0.35
        ax.bar(x - w / 2, [o / collected4 for o in obs4], w, label="Empirical", alpha=0.8)
        ax.bar(x + w / 2, [1 / num_bins4] * num_bins4, w, label="Uniform", alpha=0.5)
        ax.set_xlabel("Low 4-bit pattern")
        ax.set_ylabel("Probability")
        ax.set_title("Low-bit distribution (L=4)")
        ax.legend()
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / "figures" / "security_uniformity_L4.png")

    # =========================================================
    # Part 2: Drop-out tolerance
    # =========================================================
    logger.info("Security Part 2: Drop-out tolerance")
    drop_cfg = exp.get("dropout", {})
    drop_L = drop_cfg.get("L", 2)
    drop_Nmax = drop_cfg.get("Nmax", 256)
    drop_combos = drop_cfg.get("combos", [
        {"n": 5, "t": 2}, {"n": 5, "t": 3},
        {"n": 10, "t": 3}, {"n": 10, "t": 5},
    ])
    drop_trials = drop_cfg.get("num_trials", 200)

    drop_rows: List[Dict[str, Any]] = []

    for combo in tqdm(drop_combos, desc="Dropout"):
        dn = combo["n"]
        dt = combo["t"]
        if dt > dn:
            continue
        sr = scheme_setup(dt, dn, drop_L)

        for online in range(dt, dn + 1):
            participants = list(range(1, online + 1))
            sign_ok = 0
            ver_ok = 0
            retry_list = []

            for _ in range(drop_trials):
                m = secrets.token_bytes(32)
                M = secrets.randbelow(2 ** drop_L)
                result = sign_emb(
                    m=m, participants=participants,
                    shares=sr.shares, share_pks=sr.share_pks,
                    pk=sr.pk, Kext=sr.Kext, M=M, L=drop_L, Nmax=drop_Nmax,
                )
                if result.success:
                    sign_ok += 1
                    retry_list.append(result.retries)
                    v = verify(m, sr.pk, result.signature)
                    if v:
                        ver_ok += 1

            drop_rows.append({
                "n": dn, "t": dt, "online": online,
                "L": drop_L, "Nmax": drop_Nmax,
                "num_trials": drop_trials,
                "sign_rate": sign_ok / drop_trials,
                "verify_rate": ver_ok / drop_trials,
                "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
            })
            logger.info(f"  n={dn},t={dt},online={online}: "
                         f"sign_rate={sign_ok}/{drop_trials}")

    save_rows_csv(drop_rows, out / "tables" / "security_dropout.csv")

    # =========================================================
    # Part 3: Below-threshold forgery
    # =========================================================
    logger.info("Security Part 3: Below-threshold forgery")
    forge_cfg = exp.get("forgery", {})
    forge_combos = forge_cfg.get("combos", [
        {"n": 5, "t": 3}, {"n": 5, "t": 5},
        {"n": 10, "t": 3}, {"n": 10, "t": 5},
    ])
    forge_trials = forge_cfg.get("num_trials", 200)

    forge_rows: List[Dict[str, Any]] = []
    q = get_order()
    G = get_generator()

    for combo in tqdm(forge_combos, desc="Forgery"):
        fn = combo["n"]
        ft = combo["t"]
        if ft > fn:
            continue

        for k_prime in range(1, ft):
            sr = scheme_setup(ft, fn, 2)
            forge_attempts = 0
            forge_verify_pass = 0

            colluding = list(range(1, k_prime + 1))
            lambdas = all_lagrange_coefficients(list(range(1, ft + 1)), q)

            for _ in range(forge_trials):
                m = secrets.token_bytes(32)

                # Honest partial: generate commitments for colluding nodes
                nonces = {}
                comms = {}
                for vi in colluding:
                    ki = random_scalar()
                    nonces[vi] = ki
                    comms[vi] = scalar_mult(ki, G)

                # For missing nodes, attacker picks random R_j
                missing = list(range(k_prime + 1, ft + 1))
                for vi in missing:
                    ki = random_scalar()
                    nonces[vi] = ki
                    comms[vi] = scalar_mult(ki, G)

                from src.crypto.curve_utils import point_sum
                all_parts = list(range(1, ft + 1))
                R = point_sum([comms[vi] for vi in all_parts])
                from src.crypto.hash_prf import compute_challenge
                c = compute_challenge(R, m, sr.pk)

                # Colluding nodes compute honest partials
                s = 0
                for vi in colluding:
                    si = (nonces[vi] + c * lambdas[vi] * sr.shares[vi]) % q
                    s = (s + si) % q

                # For missing, attacker guesses random partial
                for vi in missing:
                    fake_si = random_scalar()
                    s = (s + fake_si) % q
                    forge_attempts += 1

                sig = Signature(R=R, s=s)
                if verify(m, sr.pk, sig):
                    forge_verify_pass += 1

            forge_rows.append({
                "n": fn, "t": ft, "k_prime": k_prime,
                "num_trials": forge_trials,
                "forge_attempts": forge_attempts,
                "forge_verify_pass": forge_verify_pass,
                "forge_rate": forge_verify_pass / forge_trials,
            })
            logger.info(f"  n={fn},t={ft},k'={k_prime}: "
                         f"forge_pass={forge_verify_pass}/{forge_trials}")

    save_rows_csv(forge_rows, out / "tables" / "security_forgery.csv")
    logger.info("Security experiment complete.")