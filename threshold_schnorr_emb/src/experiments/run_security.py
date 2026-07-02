def run_security(cfg: Dict[str, Any], out: Path, logger: logging.Logger) -> None:
    exp = cfg.get("security", {})

    # =========================================================
    # Part 1: Low-bit uniformity（多 L + 自适应采样）
    # =========================================================
    logger.info("Security Part 1: Low-bit uniformity")

    unif_cfg = exp.get("uniformity", {})
    n = unif_cfg.get("n", 5)
    t = unif_cfg.get("t", 3)
    Nmax = unif_cfg.get("Nmax", 4096)

    # 👉 你想要的 L 组合
    Ls = unif_cfg.get("L_values", [1, 2, 4, 6])

    # 👉 基础样本（会自动放大）
    base_samples = unif_cfg.get("num_samples", 5000)

    unif_rows: List[Dict[str, Any]] = []

    setup_style()

    for L in tqdm(Ls, desc="Uniformity"):

        # ✅ 自适应样本数（关键）
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
        expected = collected / num_bins
        observed = [counts.get(i, 0) for i in range(num_bins)]

        # χ² test
        chi2_stat, chi2_p = stats.chisquare(
            observed,
            [expected] * num_bins if collected > 0 else None
        )

        # entropy
        probs = np.array(observed, dtype=float)
        if collected > 0:
            probs /= collected
        probs = probs[probs > 0]

        emp_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

        unif_rows.append({
            "L": L,
            "n": n,
            "t": t,
            "num_samples": collected,
            "num_bins": num_bins,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "empirical_entropy": emp_entropy,
            "max_entropy": float(L),
            "entropy_ratio": emp_entropy / L if L > 0 else 0.0,
        })

        logger.info(
            f"L={L}: samples={collected}, chi2={chi2_stat:.2f}, "
            f"p={chi2_p:.4f}, entropy={emp_entropy:.4f}/{L}"
        )

        # =========================
        # 👉 每个 L 单独画图
        # =========================
        if collected > 0:
            fig, ax = plt.subplots()

            x = np.arange(num_bins)
            w = 0.35

            ax.bar(
                x - w / 2,
                [o / collected for o in observed],
                w,
                label="Empirical",
                alpha=0.8
            )

            ax.bar(
                x + w / 2,
                [1 / num_bins] * num_bins,
                w,
                label="Uniform",
                alpha=0.5
            )

            ax.set_xlabel(f"{L}-bit pattern")
            ax.set_ylabel("Probability")
            ax.set_title(f"Low-bit distribution (L={L})")

            ax.legend()
            ax.set_xticks(x)
            ax.grid(True, alpha=0.3)

            save_fig(fig, out / "figures" / f"security_uniformity_L{L}.png")

    save_rows_csv(unif_rows, out / "tables" / "security_uniformity.csv")

    # =========================================================
    # Part 2: Drop-out tolerance（保持原逻辑）
    # =========================================================
    logger.info("Security Part 2: Drop-out tolerance")

    drop_cfg = exp.get("dropout", {})
    drop_L = drop_cfg.get("L", 2)
    drop_Nmax = drop_cfg.get("Nmax", 256)

    drop_combos = drop_cfg.get("combos", [
        {"n": 5, "t": 2}, {"n": 5, "t": 3},
        {"n": 10, "t": 3}, {"n": 10, "t": 5},
    ])

    drop_trials = drop_cfg.get("num_trials", 500)  # 👉 增加

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
                    m=m,
                    participants=participants,
                    shares=sr.shares,
                    share_pks=sr.share_pks,
                    pk=sr.pk,
                    Kext=sr.Kext,
                    M=M,
                    L=drop_L,
                    Nmax=drop_Nmax,
                )

                if result.success:
                    sign_ok += 1
                    retry_list.append(result.retries)

                    if verify(m, sr.pk, result.signature):
                        ver_ok += 1

            drop_rows.append({
                "n": dn,
                "t": dt,
                "online": online,
                "sign_rate": sign_ok / drop_trials,
                "verify_rate": ver_ok / drop_trials,
                "mean_retries": float(np.mean(retry_list)) if retry_list else float("nan"),
            })

    save_rows_csv(drop_rows, out / "tables" / "security_dropout.csv")

    # =========================================================
    # Part 3: Forgery（提高次数）
    # =========================================================
    logger.info("Security Part 3: Forgery")

    forge_cfg = exp.get("forgery", {})

    forge_combos = forge_cfg.get("combos", [
        {"n": 5, "t": 3},
        {"n": 10, "t": 5},
    ])

    forge_trials = forge_cfg.get("num_trials", 1000)  # 👉 提升

    forge_rows: List[Dict[str, Any]] = []

    q = get_order()
    G = get_generator()

    for combo in tqdm(forge_combos, desc="Forgery"):
        fn = combo["n"]
        ft = combo["t"]

        for k_prime in range(1, ft):

            sr = scheme_setup(ft, fn, 2)

            forge_verify_pass = 0

            for _ in range(forge_trials):
                m = secrets.token_bytes(32)

                # 直接随机签名攻击
                fake_sig = Signature(
                    R=scalar_mult(random_scalar(), G),
                    s=random_scalar()
                )

                if verify(m, sr.pk, fake_sig):
                    forge_verify_pass += 1

            forge_rows.append({
                "n": fn,
                "t": ft,
                "k_prime": k_prime,
                "forge_rate": forge_verify_pass / forge_trials,
            })

    save_rows_csv(forge_rows, out / "tables" / "security_forgery.csv")

    logger.info("Security experiment complete.")