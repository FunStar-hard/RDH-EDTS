from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from src.experiments.common import single_trial
from src.scheme.setup import setup as scheme_setup
from src.utils.io_utils import save_rows_csv

def run_correctness_small_fast(cfg, out: Path, logger: logging.Logger) -> None:
    """快速 correctness demo：只生成 CSV，不绘图，不调用 Fig3 Enhanced"""
    exp = cfg.get("correctness", {})
    ns = exp.get("n_values", [5])
    ts = exp.get("t_values", [3])
    Ls = exp.get("L_values", [1, 2, 3])
    Nmaxs = exp.get("Nmax_values", [128])
    num_trials = exp.get("num_trials", 10)

    raw_rows = []
    summary_rows = []

    combos = [(n, t, L, Nmax) for n in ns for t in ts for L in Ls for Nmax in Nmaxs if t <= n]

    logger.info(f"Correctness small fast: {len(combos)} parameter combos x {num_trials} trials")

    for n, t, L, Nmax in tqdm(combos, desc="Correctness"):
        logger.info(f"  n={n}, t={t}, L={L}, Nmax={Nmax}")
        sr = scheme_setup(t, n, L)

        retries_list = []
        sign_ok = 0
        verify_ok = 0
        extract_ok = 0

        for trial_i in range(num_trials):
            row = single_trial(sr, t, L, Nmax)
            row.update({"n": n, "t": t, "L": L, "Nmax": Nmax, "trial": trial_i})
            raw_rows.append(row)

            if row["success"]:
                sign_ok += 1
                retries_list.append(row["retries"])
            if row["verify_ok"]:
                verify_ok += 1
            if row["extract_ok"]:
                extract_ok += 1

        retries_arr = np.array(retries_list) if retries_list else np.array([0])
        summary_rows.append({
            "n": n, "t": t, "L": L, "Nmax": Nmax,
            "num_trials": num_trials,
            "sign_success": sign_ok,
            "sign_rate": sign_ok / num_trials,
            "verify_rate": verify_ok / num_trials,
            "extract_rate": extract_ok / num_trials,
            "mean_retries": float(np.mean(retries_arr)) if retries_list else float("nan"),
            "median_retries": float(np.median(retries_arr)) if retries_list else float("nan"),
            "p95_retries": float(np.percentile(retries_arr, 95)) if retries_list else float("nan"),
        })
        logger.info(f"    sign_rate={sign_ok}/{num_trials}, mean_retries={summary_rows[-1]['mean_retries']:.2f}")

    # 保存 CSV
    save_rows_csv(raw_rows, out / "raw" / "correctness_raw.csv")
    save_rows_csv(summary_rows, out / "tables" / "correctness_summary.csv")
    logger.info("Correctness CSV saved (no figures).")

# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo-correctness-fast")

    out = Path("outputs/demo_correctness_fast")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    cfg = {
        "correctness": {
            "n_values": [5],
            "t_values": [3],
            "L_values": [1, 2, 3],
            "Nmax_values": [128],
            "num_trials": 10
        }
    }

    run_correctness_small_fast(cfg, out, logger)

    print("Done. CSV files saved in:")
    print("  outputs/demo_correctness_fast/raw/correctness_raw.csv")
    print("  outputs/demo_correctness_fast/tables/correctness_summary.csv")