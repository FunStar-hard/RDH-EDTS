"""
UAV Swarm Authentication Simulation under Stochastic Link Conditions
=====================================================================
Section 5.X — RDHS-CTS UAV Swarm Authentication Simulation
 
Simulates three representative UAV network scenarios with:
  - Random node dropout per round
  - Per-hop virtual link delay (simulated, no real sleep)
  - Rejection-sampling embedded signing (RDHS-CTS)
  - End-to-end task-label extraction verification
 
Scenarios
---------
  Stable  : dropout=10%, link delay 5–15 ms per hop
  Mobile  : dropout=20%, link delay 15–40 ms per hop
  Harsh   : dropout=35%, link delay 40–100 ms per hop
 
Fixed parameters: n=10, t=5, l=4, n_max=512, trials=500
 
Outputs (all written under <out_dir>/)
---------------------------------------
  raw/uav_swarm_raw.csv           — per-trial raw records
  tables/uav_swarm_summary.csv    — per-scenario aggregated metrics
  tables/uav_swarm_latex_table.txt— ready-to-paste LaTeX table rows
  figures/uav_swarm_latency_bar.png
  figures/uav_swarm_latency_bar_p95.png
  figures/uav_swarm_latency_box.png
  figures/uav_swarm_failure_pie_<scenario>.png
  run_uav_swarm.log               — full run log
"""
from __future__ import annotations
 
import csv
import logging
import math
import random
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
 
# ── matplotlib (non-interactive) ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
# ── project imports ────────────────────────────────────────────────────────────
# Adjust sys.path so this script can be run from the project root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR  # override below when running from cmd
# Users run:  python uav_swarm_sim.py [project_root] [out_dir]
if len(sys.argv) >= 2:
    _PROJECT_ROOT = Path(sys.argv[1]).resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from src.scheme.setup import setup as scheme_setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract
 
# ── output directory ───────────────────────────────────────────────────────────
if len(sys.argv) >= 3:
    OUT = Path(sys.argv[2]).resolve()
else:
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    OUT = _PROJECT_ROOT / "outputs" / f"{ts_str}_uav_swarm"
 
(OUT / "raw").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
 
# ── logging ────────────────────────────────────────────────────────────────────
_LOG_FILE = OUT / "run_uav_swarm.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("uav_swarm")
 
# ══════════════════════════════════════════════════════════════════════════════
# 1.  Experiment parameters
# ══════════════════════════════════════════════════════════════════════════════
N_NODES   = 10      # total UAV nodes
THRESHOLD = 5       # minimum required participants
L_BITS    = 4       # embedding bit length
N_MAX     = 512     # max rejection-sampling retries
TRIALS    = 500     # independent rounds per scenario
 
# Virtual delay model (milliseconds per hop / transmission segment)
# Each signing round has 3 logical "hops":
#   Hop 1: UAV → Aggregator (commitment upload)
#   Hop 2: Aggregator → UAV  (challenge broadcast)
#   Hop 3: UAV → Aggregator (partial response upload)
# Delay is drawn uniformly from [lo, hi] for each hop, summed over all
# participating nodes (parallel → use MAX across nodes per hop).
SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "Stable",
        "label": "Stable Swarm",
        "dropout_prob": 0.10,
        "delay_lo_ms": 5.0,
        "delay_hi_ms": 15.0,
        "color": "#2196F3",
    },
    {
        "name": "Mobile",
        "label": "Mobile Swarm",
        "dropout_prob": 0.20,
        "delay_lo_ms": 15.0,
        "delay_hi_ms": 40.0,
        "color": "#FF9800",
    },
    {
        "name": "Harsh",
        "label": "Harsh Swarm",
        "dropout_prob": 0.35,
        "delay_lo_ms": 40.0,
        "delay_hi_ms": 100.0,
        "color": "#F44336",
    },
]
 
# ══════════════════════════════════════════════════════════════════════════════
# 2.  Virtual delay helper
# ══════════════════════════════════════════════════════════════════════════════
 
def sample_hop_delay_ms(lo: float, hi: float) -> float:
    """Sample a uniform random link delay in [lo, hi] ms for one hop."""
    return random.uniform(lo, hi)
 
 
def virtual_round_delay_ms(
    n_participants: int,
    lo: float,
    hi: float,
    n_hops: int = 3,
) -> float:
    """
    Virtual end-to-end delay for one signing round.
 
    Model:
      - Each of the 3 hops is a parallel broadcast/collection over
        n_participants links.  Bottleneck = MAX of n_participants samples.
      - Total delay = sum of hop bottlenecks.
    """
    total = 0.0
    for _ in range(n_hops):
        hop_delays = [sample_hop_delay_ms(lo, hi) for _ in range(n_participants)]
        total += max(hop_delays)
    return total
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  Core per-trial function
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class TrialResult:
    scenario: str
    trial_idx: int
    # --- availability ---
    n_online: int
    below_threshold: bool        # online < t  → immediate failure
    # --- signing outcome ---
    emb_failed: bool             # online >= t but nmax exceeded
    sign_success: bool
    # --- correctness ---
    verify_ok: bool
    extract_ok: bool
    # --- timing (seconds, real CPU) ---
    sign_cpu_s: float
    verify_cpu_s: float
    extract_cpu_s: float
    total_cpu_s: float
    # --- virtual link delay (ms) ---
    virtual_delay_ms: float       # sum over all attempts
    # --- retry count ---
    retries: int
    # --- end-to-end latency (ms): cpu + virtual ---
    e2e_latency_ms: float
 
 
def run_trial(
    scenario: Dict[str, Any],
    trial_idx: int,
    sr: Any,           # SetupResult
) -> TrialResult:
    """Execute one authentication round under the given scenario."""
    name        = scenario["name"]
    drop_p      = scenario["dropout_prob"]
    delay_lo    = scenario["delay_lo_ms"]
    delay_hi    = scenario["delay_hi_ms"]
 
    # ── Determine online nodes ──────────────────────────────────────────────
    all_nodes = list(range(1, N_NODES + 1))
    online    = [v for v in all_nodes if random.random() >= drop_p]
    n_online  = len(online)
 
    if n_online < THRESHOLD:
        return TrialResult(
            scenario=name, trial_idx=trial_idx,
            n_online=n_online, below_threshold=True,
            emb_failed=False, sign_success=False,
            verify_ok=False, extract_ok=False,
            sign_cpu_s=0.0, verify_cpu_s=0.0, extract_cpu_s=0.0,
            total_cpu_s=0.0,
            virtual_delay_ms=0.0,
            retries=0,
            e2e_latency_ms=0.0,
        )
 
    # ── Select exactly t participants ───────────────────────────────────────
    participants = random.sample(online, THRESHOLD)
 
    # ── Random message and task label ────────────────────────────────────────
    m  = secrets.token_bytes(32)
    mh = secrets.randbelow(2 ** L_BITS)
 
    # ── Signing (CPU timed) ──────────────────────────────────────────────────
    t_sign_start = time.perf_counter()
    result = sign_emb(
        m=m,
        participants=participants,
        shares={v: sr.shares[v] for v in participants},
        share_pks={v: sr.share_pks[v] for v in participants},
        pk=sr.pk,
        Kext=sr.Kext,
        M=mh,
        L=L_BITS,
        Nmax=N_MAX,
    )
    sign_cpu_s = time.perf_counter() - t_sign_start
 
    # ── Virtual link delay (per attempt × 3 hops, bottleneck model) ─────────
    # Each retry invokes 3 broadcast/collect phases.
    # We accumulate delay for each attempt actually performed.
    n_attempts  = result.retries  # == N_MAX if failed, else attempt that succeeded
    virtual_delay = 0.0
    for _ in range(n_attempts):
        virtual_delay += virtual_round_delay_ms(
            THRESHOLD, delay_lo, delay_hi, n_hops=3
        )
 
    if not result.success or result.signature is None:
        return TrialResult(
            scenario=name, trial_idx=trial_idx,
            n_online=n_online, below_threshold=False,
            emb_failed=True, sign_success=False,
            verify_ok=False, extract_ok=False,
            sign_cpu_s=sign_cpu_s, verify_cpu_s=0.0, extract_cpu_s=0.0,
            total_cpu_s=sign_cpu_s,
            virtual_delay_ms=virtual_delay,
            retries=result.retries,
            e2e_latency_ms=sign_cpu_s * 1000 + virtual_delay,
        )
 
    # ── Verification (CPU timed) ─────────────────────────────────────────────
    t_ver_start = time.perf_counter()
    v_ok = verify(m, sr.pk, result.signature)
    verify_cpu_s = time.perf_counter() - t_ver_start
 
    # ── Extraction (CPU timed) ────────────────────────────────────────────────
    t_ext_start = time.perf_counter()
    recovered = extract(m, sr.pk, result.signature, sr.Kext, L_BITS)
    extract_cpu_s = time.perf_counter() - t_ext_start
 
    ext_ok = (recovered == mh)
 
    total_cpu_s  = sign_cpu_s + verify_cpu_s + extract_cpu_s
    e2e_ms       = total_cpu_s * 1000 + virtual_delay
 
    return TrialResult(
        scenario=name, trial_idx=trial_idx,
        n_online=n_online, below_threshold=False,
        emb_failed=False, sign_success=True,
        verify_ok=v_ok, extract_ok=ext_ok,
        sign_cpu_s=sign_cpu_s, verify_cpu_s=verify_cpu_s,
        extract_cpu_s=extract_cpu_s,
        total_cpu_s=total_cpu_s,
        virtual_delay_ms=virtual_delay,
        retries=result.retries,
        e2e_latency_ms=e2e_ms,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  Run all scenarios
# ══════════════════════════════════════════════════════════════════════════════
 
def run_all(sr: Any) -> List[TrialResult]:
    all_results: List[TrialResult] = []
    for scen in SCENARIOS:
        logger.info(
            f"  Scenario '{scen['name']}': dropout={scen['dropout_prob']*100:.0f}%, "
            f"delay=[{scen['delay_lo_ms']:.0f},{scen['delay_hi_ms']:.0f}]ms"
        )
        for trial_idx in range(TRIALS):
            r = run_trial(scen, trial_idx, sr)
            all_results.append(r)
            if (trial_idx + 1) % 100 == 0:
                logger.info(
                    f"    [{scen['name']}] trial {trial_idx+1}/{TRIALS} — "
                    f"online={r.n_online}, bt={r.below_threshold}, "
                    f"success={r.sign_success}"
                )
    return all_results
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5.  Aggregate metrics per scenario
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class ScenarioStats:
    name: str
    label: str
    color: str
    n_trials: int
    # rates (0–1)
    e2e_success_rate: float      # sign_success AND verify AND extract
    below_threshold_rate: float
    emb_failure_rate: float
    extraction_accuracy: float   # among successes
    # latency (ms) — over successful rounds only
    mean_e2e_ms: float
    p95_e2e_ms: float
    mean_sign_ms: float
    p95_sign_ms: float
    mean_retries: float
    p95_retries: float
    # extra: mean n_online
    mean_n_online: float
    # latency over ALL trials (including aborts → 0)
    mean_e2e_all_ms: float
 
 
def compute_stats(results: List[TrialResult], scen: Dict[str, Any]) -> ScenarioStats:
    name  = scen["name"]
    rows  = [r for r in results if r.scenario == name]
    N     = len(rows)
 
    n_bt        = sum(1 for r in rows if r.below_threshold)
    n_emb_fail  = sum(1 for r in rows if r.emb_failed)
    n_success   = sum(1 for r in rows if r.sign_success)
    n_e2e_ok    = sum(1 for r in rows if r.sign_success and r.verify_ok and r.extract_ok)
 
    # extraction accuracy: only among fully successful
    n_ext_ok = sum(1 for r in rows if r.sign_success and r.verify_ok and r.extract_ok)
    ext_acc  = n_ext_ok / n_success if n_success > 0 else float("nan")
 
    # latency statistics over successful trials
    succ_rows    = [r for r in rows if r.sign_success and r.verify_ok]
    e2e_vals     = [r.e2e_latency_ms for r in succ_rows]
    sign_vals    = [r.sign_cpu_s * 1000 for r in succ_rows]
    retry_vals   = [r.retries for r in succ_rows]
    online_vals  = [r.n_online for r in rows]
 
    def safe_mean(arr): return float(np.mean(arr)) if arr else float("nan")
    def safe_p95 (arr): return float(np.percentile(arr, 95)) if arr else float("nan")
 
    # mean e2e over ALL trials (treat failures as 0 delay for the system metric)
    all_e2e = [r.e2e_latency_ms if r.sign_success else 0.0 for r in rows]
 
    return ScenarioStats(
        name=name,
        label=scen["label"],
        color=scen["color"],
        n_trials=N,
        e2e_success_rate=n_e2e_ok / N,
        below_threshold_rate=n_bt / N,
        emb_failure_rate=n_emb_fail / N,
        extraction_accuracy=ext_acc,
        mean_e2e_ms=safe_mean(e2e_vals),
        p95_e2e_ms=safe_p95(e2e_vals),
        mean_sign_ms=safe_mean(sign_vals),
        p95_sign_ms=safe_p95(sign_vals),
        mean_retries=safe_mean(retry_vals),
        p95_retries=safe_p95(retry_vals),
        mean_n_online=safe_mean(online_vals),
        mean_e2e_all_ms=safe_mean(all_e2e),
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6.  Save raw CSV
# ══════════════════════════════════════════════════════════════════════════════
 
def save_raw_csv(results: List[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario", "trial_idx", "n_online", "below_threshold",
        "emb_failed", "sign_success", "verify_ok", "extract_ok",
        "sign_cpu_s", "verify_cpu_s", "extract_cpu_s", "total_cpu_s",
        "virtual_delay_ms", "retries", "e2e_latency_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "scenario":         r.scenario,
                "trial_idx":        r.trial_idx,
                "n_online":         r.n_online,
                "below_threshold":  int(r.below_threshold),
                "emb_failed":       int(r.emb_failed),
                "sign_success":     int(r.sign_success),
                "verify_ok":        int(r.verify_ok),
                "extract_ok":       int(r.extract_ok),
                "sign_cpu_s":       f"{r.sign_cpu_s:.6f}",
                "verify_cpu_s":     f"{r.verify_cpu_s:.6f}",
                "extract_cpu_s":    f"{r.extract_cpu_s:.6f}",
                "total_cpu_s":      f"{r.total_cpu_s:.6f}",
                "virtual_delay_ms": f"{r.virtual_delay_ms:.3f}",
                "retries":          r.retries,
                "e2e_latency_ms":   f"{r.e2e_latency_ms:.3f}",
            })
    logger.info(f"  Raw CSV saved → {path}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7.  Save summary CSV + LaTeX table snippet
# ══════════════════════════════════════════════════════════════════════════════
 
def save_summary(stats_list: List[ScenarioStats], out_dir: Path) -> None:
    # ---- CSV ----
    csv_path = out_dir / "tables" / "uav_swarm_summary.csv"
    fields = [
        "scenario", "n_trials",
        "e2e_success_rate", "below_threshold_rate", "emb_failure_rate",
        "mean_retries", "p95_retries",
        "mean_e2e_ms", "p95_e2e_ms",
        "mean_sign_ms", "p95_sign_ms",
        "extraction_accuracy", "mean_n_online",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats_list:
            w.writerow({
                "scenario":              s.name,
                "n_trials":              s.n_trials,
                "e2e_success_rate":      f"{s.e2e_success_rate:.4f}",
                "below_threshold_rate":  f"{s.below_threshold_rate:.4f}",
                "emb_failure_rate":      f"{s.emb_failure_rate:.4f}",
                "mean_retries":          f"{s.mean_retries:.2f}",
                "p95_retries":           f"{s.p95_retries:.1f}",
                "mean_e2e_ms":           f"{s.mean_e2e_ms:.1f}",
                "p95_e2e_ms":            f"{s.p95_e2e_ms:.1f}",
                "mean_sign_ms":          f"{s.mean_sign_ms:.1f}",
                "p95_sign_ms":           f"{s.p95_sign_ms:.1f}",
                "extraction_accuracy":   f"{s.extraction_accuracy:.4f}",
                "mean_n_online":         f"{s.mean_n_online:.2f}",
            })
    logger.info(f"  Summary CSV saved → {csv_path}")
 
    # ---- LaTeX table snippet ----
    tex_path = out_dir / "tables" / "uav_swarm_latex_table.txt"
    lines = []
    lines.append("% === Table: UAV Swarm Authentication under Stochastic Link Conditions ===")
    lines.append("% n=10, t=5, l=4, n_max=512, trials=500 per scenario")
    lines.append("%")
    lines.append(r"% \begin{table}[!htbp]")
    lines.append(r"% \caption{UAV Swarm Authentication Performance under Stochastic Link Conditions ($n=10$, $t=5$, $l=4$, $n_{\max}=512$).}")
    lines.append(r"% \label{tab:uav_swarm}")
    lines.append(r"% \centering")
    lines.append(r"% \footnotesize")
    lines.append(r"% \begin{tabular}{lccccccc}")
    lines.append(r"% \toprule")
    lines.append(
        r"% Scenario & \makecell{E2E\\Success\\Rate} & \makecell{Below-\\Threshold\\Failure} & "
        r"\makecell{Emb.\\Failure\\Rate} & \makecell{Mean\\Retries} & "
        r"\makecell{Mean\\Delay\\(ms)} & \makecell{P95\\Delay\\(ms)} & "
        r"\makecell{Extraction\\Accuracy} \\"
    )
    lines.append(r"% \midrule")
    for s in stats_list:
        ext_str = f"{s.extraction_accuracy:.4f}" if not math.isnan(s.extraction_accuracy) else "--"
        mean_e2e = f"{s.mean_e2e_ms:.1f}" if not math.isnan(s.mean_e2e_ms) else "--"
        p95_e2e  = f"{s.p95_e2e_ms:.1f}"  if not math.isnan(s.p95_e2e_ms)  else "--"
        lines.append(
            f"% {s.label} & {s.e2e_success_rate:.4f} & {s.below_threshold_rate:.4f} & "
            f"{s.emb_failure_rate:.4f} & {s.mean_retries:.2f} & "
            f"{mean_e2e} & {p95_e2e} & {ext_str} \\\\"
        )
    lines.append(r"% \bottomrule")
    lines.append(r"% \end{tabular}")
    lines.append(r"% \end{table}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"  LaTeX snippet saved → {tex_path}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8.  Figures
# ══════════════════════════════════════════════════════════════════════════════
 
FONT_SIZE   = 11
TITLE_SIZE  = 12
BAR_WIDTH   = 0.45
FIG_DPI     = 200
 
def _setup_style():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": FIG_DPI,
    })
 
 
def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure saved → {path}")
 
 
# ── Fig A: average e2e latency bar chart (grouped: mean + P95) ───────────────
def fig_latency_bar(stats_list: List[ScenarioStats], out_dir: Path):
    _setup_style()
    names  = [s.label for s in stats_list]
    means  = [s.mean_e2e_ms for s in stats_list]
    p95s   = [s.p95_e2e_ms  for s in stats_list]
    colors = [s.color        for s in stats_list]
 
    x = np.arange(len(names))
    w = 0.35
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w / 2, means, width=w, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.7, label="Mean E2E Latency")
    bars2 = ax.bar(x + w / 2, p95s,  width=w, color=colors, alpha=0.45,
                   edgecolor="black", linewidth=0.7, hatch="///", label="P95 E2E Latency")
 
    for bar, v in zip(bars1, means):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, p95s):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9)
 
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title(
        f"UAV Swarm E2E Authentication Latency\n"
        f"(n={N_NODES}, t={THRESHOLD}, l={L_BITS}, "
        f"$n_{{\\max}}$={N_MAX}, trials={TRIALS})"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(p95s) * 1.2 if not all(math.isnan(v) for v in p95s) else 1)
    _savefig(fig, out_dir / "figures" / "uav_swarm_latency_bar.png")
 
 
# ── Fig B: P95 latency only (cleaner single-bar version for paper) ────────────
def fig_latency_p95(stats_list: List[ScenarioStats], out_dir: Path):
    _setup_style()
    names  = [s.label for s in stats_list]
    p95s   = [s.p95_e2e_ms  for s in stats_list]
    means  = [s.mean_e2e_ms for s in stats_list]
    colors = [s.color        for s in stats_list]
 
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, p95s, color=colors, alpha=0.82,
                  edgecolor="black", linewidth=0.8)
    # overlay mean as diamond markers
    ax.scatter(names, means, marker="D", s=50, color="white",
               edgecolors="black", linewidths=1.2, zorder=5, label="Mean")
 
    for bar, v in zip(bars, p95s):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f} ms", ha="center", va="bottom", fontsize=9)
 
    ax.set_ylabel("P95 End-to-End Latency (ms)")
    ax.set_title(
        "P95 E2E Latency under Different UAV Link Conditions\n"
        f"(n={N_NODES}, t={THRESHOLD}, l={L_BITS})"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    max_val = max((v for v in p95s if not math.isnan(v)), default=1)
    ax.set_ylim(0, max_val * 1.25)
    _savefig(fig, out_dir / "figures" / "uav_swarm_latency_bar_p95.png")
 
 
# ── Fig C: box plot of e2e latency distributions ─────────────────────────────
def fig_latency_box(results: List[TrialResult],
                    stats_list: List[ScenarioStats],
                    out_dir: Path):
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
 
    data_by_scen = []
    labels       = []
    colors       = []
    for scen, st in zip(SCENARIOS, stats_list):
        vals = [r.e2e_latency_ms
                for r in results
                if r.scenario == scen["name"] and r.sign_success]
        data_by_scen.append(vals)
        labels.append(st.label)
        colors.append(scen["color"])
 
    bp = ax.boxplot(
        data_by_scen,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
 
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title(
        "Distribution of E2E Authentication Latency\n"
        f"(n={N_NODES}, t={THRESHOLD}, l={L_BITS}, successful rounds only)"
    )
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir / "figures" / "uav_swarm_latency_box.png")
 
 
# ── Fig D: failure composition pie charts ────────────────────────────────────
def fig_failure_pies(stats_list: List[ScenarioStats],
                     results: List[TrialResult],
                     out_dir: Path):
    _setup_style()
    for scen, st in zip(SCENARIOS, stats_list):
        rows = [r for r in results if r.scenario == scen["name"]]
        N = len(rows)
        n_bt   = sum(1 for r in rows if r.below_threshold)
        n_emb  = sum(1 for r in rows if r.emb_failed)
        n_ok   = sum(1 for r in rows if r.sign_success and r.verify_ok and r.extract_ok)
 
        sizes  = [n_ok, n_bt, n_emb]
        labels = [
            f"E2E Success\n({n_ok}/{N})",
            f"Below-Threshold\n({n_bt}/{N})",
            f"Emb. Failure\n({n_emb}/{N})",
        ]
        pie_colors = ["#4CAF50", "#F44336", "#FF9800"]
        explode    = [0.03, 0.05, 0.05]
 
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0.5 else "",
            colors=pie_colors,
            explode=explode,
            startangle=140,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8.5)
        ax.set_title(
            f"Failure Composition — {st.label}\n"
            f"(dropout={scen['dropout_prob']*100:.0f}%, "
            f"delay=[{scen['delay_lo_ms']:.0f},{scen['delay_hi_ms']:.0f}]ms)"
        )
        _savefig(
            fig,
            out_dir / "figures" / f"uav_swarm_failure_pie_{scen['name'].lower()}.png"
        )
 
 
# ── Fig E: success rate & below-threshold rate stacked bar ───────────────────
def fig_outcome_stacked(stats_list: List[ScenarioStats], out_dir: Path):
    _setup_style()
    names = [s.label for s in stats_list]
    ok    = [s.e2e_success_rate      for s in stats_list]
    bt    = [s.below_threshold_rate  for s in stats_list]
    emb   = [s.emb_failure_rate      for s in stats_list]
 
    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4))
 
    p1 = ax.bar(x, ok,  color="#4CAF50", alpha=0.85, edgecolor="black",
                linewidth=0.7, label="E2E Success")
    p2 = ax.bar(x, bt,  bottom=ok, color="#F44336", alpha=0.85,
                edgecolor="black", linewidth=0.7, label="Below-Threshold Failure")
    p3 = ax.bar(x, emb, bottom=[a + b for a, b in zip(ok, bt)],
                color="#FF9800", alpha=0.85, edgecolor="black",
                linewidth=0.7, label="Embedding Failure")
 
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Proportion of Trials")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Authentication Outcome Breakdown per Scenario\n"
        f"(n={N_NODES}, t={THRESHOLD}, l={L_BITS}, trials={TRIALS})"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
 
    for xi, (o, b, e) in enumerate(zip(ok, bt, emb)):
        ax.text(xi, o / 2, f"{o:.2f}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        if b > 0.01:
            ax.text(xi, o + b / 2, f"{b:.2f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if e > 0.005:
            ax.text(xi, o + b + e / 2, f"{e:.3f}", ha="center", va="center",
                    fontsize=8, color="white")
 
    _savefig(fig, out_dir / "figures" / "uav_swarm_outcome_stacked.png")
 
 
# ── Fig F: mean retries per scenario ─────────────────────────────────────────
def fig_retries_bar(stats_list: List[ScenarioStats], out_dir: Path):
    _setup_style()
    names    = [s.label        for s in stats_list]
    means    = [s.mean_retries for s in stats_list]
    p95s     = [s.p95_retries  for s in stats_list]
    colors   = [s.color        for s in stats_list]
 
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - w / 2, means, width=w, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.7, label="Mean Retries")
    bars2 = ax.bar(x + w / 2, p95s,  width=w, color=colors, alpha=0.45,
                   edgecolor="black", linewidth=0.7, hatch="///", label="P95 Retries")
 
    # theoretical mean = 2^l
    ax.axhline(2 ** L_BITS, color="gray", linestyle="--", linewidth=1.2,
               label=f"Theoretical mean = $2^{L_BITS}$ = {2**L_BITS}")
 
    for bar, v in zip(bars1, means):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(bars2, p95s):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
 
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of Signing Retries")
    ax.set_title(
        f"Rejection-Sampling Retries per Scenario (l={L_BITS})\n"
        "(successful rounds only)"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir / "figures" / "uav_swarm_retries_bar.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 9.  Main entry point
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    logger.info("=" * 70)
    logger.info("UAV Swarm Authentication Simulation — RDHS-CTS")
    logger.info(f"  n={N_NODES}, t={THRESHOLD}, l={L_BITS}, "
                f"n_max={N_MAX}, trials={TRIALS}")
    logger.info(f"  Output directory: {OUT}")
    logger.info("=" * 70)
 
    # ── System setup (shared across all scenarios) ───────────────────────────
    logger.info("Running scheme Setup …")
    t_setup = time.perf_counter()
    sr = scheme_setup(THRESHOLD, N_NODES, L_BITS)
    logger.info(f"  Setup done in {(time.perf_counter()-t_setup)*1000:.2f} ms")
 
    # ── Run all trials ────────────────────────────────────────────────────────
    logger.info(f"\nRunning {len(SCENARIOS)} scenarios × {TRIALS} trials …")
    t_run = time.perf_counter()
    all_results = run_all(sr)
    logger.info(f"All trials done in {(time.perf_counter()-t_run):.2f} s")
 
    # ── Save raw data ─────────────────────────────────────────────────────────
    save_raw_csv(all_results, OUT / "raw" / "uav_swarm_raw.csv")
 
    # ── Compute per-scenario statistics ──────────────────────────────────────
    logger.info("\nAggregating statistics …")
    stats_list = [compute_stats(all_results, scen) for scen in SCENARIOS]
 
    # ── Print summary to log ──────────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info(
        f"{'Scenario':<18} {'E2E-OK':>7} {'BT-fail':>8} {'Emb-fail':>9} "
        f"{'MeanRetry':>10} {'MeanE2E(ms)':>12} {'P95E2E(ms)':>11} {'ExtAcc':>8}"
    )
    logger.info("─" * 70)
    for s in stats_list:
        ext_str  = f"{s.extraction_accuracy:.4f}" if not math.isnan(s.extraction_accuracy) else "   N/A"
        mean_str = f"{s.mean_e2e_ms:.1f}"         if not math.isnan(s.mean_e2e_ms)         else "  N/A"
        p95_str  = f"{s.p95_e2e_ms:.1f}"          if not math.isnan(s.p95_e2e_ms)          else "  N/A"
        logger.info(
            f"{s.label:<18} {s.e2e_success_rate:>7.4f} {s.below_threshold_rate:>8.4f} "
            f"{s.emb_failure_rate:>9.4f} {s.mean_retries:>10.2f} "
            f"{mean_str:>12} {p95_str:>11} {ext_str:>8}"
        )
    logger.info("─" * 70)
 
    # ── Save summary CSV and LaTeX ────────────────────────────────────────────
    save_summary(stats_list, OUT)
 
    # ── Generate all figures ──────────────────────────────────────────────────
    logger.info("\nGenerating figures …")
    fig_latency_bar    (stats_list,          OUT)
    fig_latency_p95    (stats_list,          OUT)
    fig_latency_box    (all_results, stats_list, OUT)
    fig_failure_pies   (stats_list, all_results, OUT)
    fig_outcome_stacked(stats_list,          OUT)
    fig_retries_bar    (stats_list,          OUT)
 
    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Experiment complete.")
    logger.info(f"  Raw data   : {OUT / 'raw' / 'uav_swarm_raw.csv'}")
    logger.info(f"  Summary CSV: {OUT / 'tables' / 'uav_swarm_summary.csv'}")
    logger.info(f"  LaTeX table: {OUT / 'tables' / 'uav_swarm_latex_table.txt'}")
    logger.info(f"  Figures    : {OUT / 'figures'}")
    logger.info(f"  Log file   : {_LOG_FILE}")
    logger.info("=" * 70)
 
 
if __name__ == "__main__":
    main()