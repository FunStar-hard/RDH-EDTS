"""Command-line interface for running experiments."""
from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

from src.config import load_config, save_config_snapshot, get_scheme_params
from src.utils.io_utils import make_output_dir, save_json
from src.utils.log_utils import setup_logger
from src.utils.seed_utils import set_seed


def _build_metadata(cfg: dict, exp_name: str, start_time: float) -> dict:
    import platform
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "curve": "P-256",
        "hash": "SHA-256",
        "prf": "HMAC-SHA-256",
        "kext_bits": 256,
        "experiment": exp_name,
        "seed": cfg.get("seed", None),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    }
    try:
        import ecdsa
        meta["ecdsa_version"] = ecdsa.__version__
    except Exception:
        pass
    try:
        import numpy
        meta["numpy_version"] = numpy.__version__
    except Exception:
        pass
    return meta


def cmd_demo_sign(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    sp = get_scheme_params(cfg)

    from src.scheme.setup import setup
    from src.scheme.sign_emb import sign_emb
    from src.scheme.verify import verify
    from src.scheme.extract import extract

    print("=" * 60)
    print("  Demo: Threshold Schnorr with Embedded Information")
    print("=" * 60)
    print(f"  n={sp['n']}, t={sp['t']}, L={sp['L']}, Nmax={sp['Nmax']}")

    sr = setup(sp["t"], sp["n"], sp["L"])
    m = secrets.token_bytes(32)
    M = secrets.randbelow(2 ** sp["L"])

    participants = list(range(1, sp["t"] + 1))
    print(f"  Participants: {participants}")
    print(f"  Message (hex): {m.hex()[:32]}...")
    print(f"  Embedded M: {bin(M)} (decimal {M})")

    result = sign_emb(
        m=m,
        participants=participants,
        shares=sr.shares,
        share_pks=sr.share_pks,
        pk=sr.pk,
        Kext=sr.Kext,
        M=M,
        L=sp["L"],
        Nmax=sp["Nmax"],
    )

    if result.success:
        print(f"\n  Signing SUCCESS after {result.retries} retries")
        sig = result.signature
        v = verify(m, sr.pk, sig)
        print(f"  Verify: {v}")
        recovered = extract(m, sr.pk, sig, sr.Kext, sp["L"])
        print(f"  Extracted M: {recovered}, Original M: {M}, Match: {recovered == M}")
    else:
        print(f"\n  Signing FAILED after {result.retries} retries (Nmax reached)")

    print("=" * 60)


def cmd_run_correctness(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    start = time.time()
    out = make_output_dir("outputs", "correctness")
    save_config_snapshot(cfg, out / "config_snapshot.yaml")
    logger = setup_logger("correctness", out / "run.log")
    logger.info("Starting correctness experiment")

    from src.experiments.correctness import run_correctness
    run_correctness(cfg, out, logger)

    meta = _build_metadata(cfg, "correctness", start)
    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = time.time() - start
    save_json(meta, out / "metadata.json")
    logger.info(f"Done in {meta['total_seconds']:.1f}s. Output: {out}")


def cmd_run_security(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    start = time.time()
    out = make_output_dir("outputs", "security")
    save_config_snapshot(cfg, out / "config_snapshot.yaml")
    logger = setup_logger("security", out / "run.log")
    logger.info("Starting security experiment")

    from src.experiments.security import run_security
    run_security(cfg, out, logger)

    meta = _build_metadata(cfg, "security", start)
    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = time.time() - start
    save_json(meta, out / "metadata.json")
    logger.info(f"Done in {meta['total_seconds']:.1f}s. Output: {out}")


def cmd_run_efficiency(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    start = time.time()
    out = make_output_dir("outputs", "efficiency")
    save_config_snapshot(cfg, out / "config_snapshot.yaml")
    logger = setup_logger("efficiency", out / "run.log")
    logger.info("Starting efficiency experiment")

    from src.experiments.efficiency import run_efficiency
    run_efficiency(cfg, out, logger)

    meta = _build_metadata(cfg, "efficiency", start)
    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = time.time() - start
    save_json(meta, out / "metadata.json")
    logger.info(f"Done in {meta['total_seconds']:.1f}s. Output: {out}")


def cmd_run_comparison(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    start = time.time()
    out = make_output_dir("outputs", "comparison")
    save_config_snapshot(cfg, out / "config_snapshot.yaml")
    logger = setup_logger("comparison", out / "run.log")
    logger.info("Starting comparison experiment")

    from src.experiments.comparison import run_comparison
    run_comparison(cfg, out, logger)

    meta = _build_metadata(cfg, "comparison", start)
    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = time.time() - start
    save_json(meta, out / "metadata.json")
    logger.info(f"Done in {meta['total_seconds']:.1f}s. Output: {out}")


def cmd_run_scalability(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))
    start = time.time()
    out = make_output_dir("outputs", "scalability")
    save_config_snapshot(cfg, out / "config_snapshot.yaml")
    logger = setup_logger("scalability", out / "run.log")
    logger.info("Starting scalability experiment")

    from src.experiments.scalability import run_scalability
    run_scalability(cfg, out, logger)

    meta = _build_metadata(cfg, "scalability", start)
    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = time.time() - start
    save_json(meta, out / "metadata.json")
    logger.info(f"Done in {meta['total_seconds']:.1f}s. Output: {out}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run every experiment sequentially."""
    for fn in [
        cmd_run_correctness,
        cmd_run_security,
        cmd_run_efficiency,
        cmd_run_comparison,
        cmd_run_scalability,
    ]:
        try:
            fn(args)
        except Exception as exc:
            print(f"[ERROR] {fn.__name__}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="threshold_schnorr_emb",
        description="Threshold Schnorr with Embedded Information – Research Prototype",
    )
    sub = parser.add_subparsers(dest="command")

    for name, func in [
        ("demo-sign", cmd_demo_sign),
        ("run-correctness", cmd_run_correctness),
        ("run-security", cmd_run_security),
        ("run-efficiency", cmd_run_efficiency),
        ("run-comparison", cmd_run_comparison),
        ("run-scalability", cmd_run_scalability),
        ("run-all", cmd_run_all),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=str, default="configs/default.yaml")
        p.set_defaults(func=func)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()