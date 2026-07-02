from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from src.experiments.common import timed_full_auth
from src.scheme.setup import setup as scheme_setup
from src.utils.io_utils import save_rows_csv
from src.utils.time_utils import timer

def run_efficiency_small_fast(cfg, out: Path, logger: logging.Logger) -> None:
    """快速效率 demo：只生成 CSV，不绘图"""
    exp = cfg.get("efficiency", {})
    num_trials = exp.get("num_trials", 5)

    # ---- Part 1: Module breakdown ----
    logger.info("Efficiency Part 1: Module breakdown (fast)")

    bd_n = exp.get("breakdown_n", 5)
    bd_t = exp.get("breakdown_t", 3)
    bd_L = exp.get("breakdown_L", 2)
    bd_Nmax = exp.get("breakdown_Nmax", 128)

    sr = scheme_setup(bd_t, bd_n, bd_L)

    # Also time setup separately
    with timer() as ts:
        _ = scheme_setup(bd_t, bd_n, bd_L)
    setup_time = ts["elapsed"]

    breakdown_rows = []
    for i in tqdm(range(num_trials), desc="Breakdown"):
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

    save_rows_csv(breakdown_rows, out / "raw" / "efficiency_breakdown_raw.csv")

    # Summary
    keys = ["setup", "com_gen", "part_sign", "share_ver", "agg", "verify", "extract", "full_auth"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in breakdown_rows]
        summary[k + "_mean"] = float(np.mean(vals))
        summary[k + "_std"] = float(np.std(vals))
    summary["mem_peak_mean"] = float(np.mean([r["mem_peak_bytes"] for r in breakdown_rows]))
    summary["n"] = bd_n
    summary["t"] = bd_t
    summary["L"] = bd_L
    summary["Nmax"] = bd_Nmax
    save_rows_csv([summary], out / "tables" / "efficiency_breakdown_summary.csv")
    logger.info(f"Breakdown summary saved. full_auth_mean={summary['full_auth_mean']*1000:.1f}ms")

    # ---- Part 2: FullAuth vs L ----
    logger.info("Efficiency Part 2: FullAuth vs L (fast)")
    Ls = exp.get("L_values", [1, 2, 3])
    ts_list = exp.get("t_values_for_L", [2, 3])
    fixed_n = exp.get("fixed_n_for_L", 5)
    Nmax_eff = exp.get("Nmax", 128)
    trials_per = exp.get("trials_per_combo", 5)

    lat_L_rows = []
    for t_val in tqdm(ts_list, desc="FullAuth vs L"):
        if t_val > fixed_n:
            continue
        for L in Ls:
            sr2 = scheme_setup(t_val, fixed_n, L)
            times = [timed_full_auth(sr2, t_val, L, Nmax_eff).full_auth for _ in range(trials_per)]
            lat_L_rows.append({
                "n": fixed_n, "t": t_val, "L": L,
                "mean_latency": float(np.mean(times)),
                "std_latency": float(np.std(times)),
            })
    save_rows_csv(lat_L_rows, out / "tables" / "efficiency_fullauth_vs_L.csv")

    # ---- Part 3: FullAuth vs t ----
    logger.info("Efficiency Part 3: FullAuth vs t (fast)")
    ns_list = exp.get("n_values_for_t", [5])
    fixed_L = exp.get("fixed_L_for_t", 2)

    lat_t_rows = []
    for n_val in tqdm(ns_list, desc="FullAuth vs t"):
        t_range = list(range(2, n_val + 1, max(1, n_val // 5)))
        if t_range[-1] != n_val:
            t_range.append(n_val)
        for t_val in t_range:
            sr3 = scheme_setup(t_val, n_val, fixed_L)
            times = [timed_full_auth(sr3, t_val, fixed_L, Nmax_eff).full_auth for _ in range(trials_per)]
            lat_t_rows.append({
                "n": n_val, "t": t_val, "L": fixed_L,
                "mean_latency": float(np.mean(times)),
                "std_latency": float(np.std(times)),
            })
    save_rows_csv(lat_t_rows, out / "tables" / "efficiency_fullauth_vs_t.csv")
    logger.info("Efficiency CSVs saved (no figures).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo-efficiency-fast")

    out = Path("outputs/demo_efficiency_fast")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    cfg = {
        "efficiency": {
            "num_trials": 5,
            "breakdown_n": 5,
            "breakdown_t": 3,
            "breakdown_L": 2,
            "breakdown_Nmax": 128,
            "L_values": [1, 2, 3],
            "t_values_for_L": [2, 3],
            "fixed_n_for_L": 5,
            "Nmax": 128,
            "trials_per_combo": 5,
            "n_values_for_t": [5],
            "fixed_L_for_t": 2
        }
    }

    run_efficiency_small_fast(cfg, out, logger)

    print("Done. CSV files saved in:")
    print("  outputs/demo_efficiency_fast/raw/")
    print("  outputs/demo_efficiency_fast/tables/")