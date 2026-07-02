"""Smoke test: ensure experiments can run with minimal parameters."""
import tempfile
import logging
import unittest
from pathlib import Path

from src.experiments.correctness import run_correctness
from src.experiments.security import run_security
from src.experiments.efficiency import run_efficiency
from src.experiments.comparison import run_comparison
from src.experiments.scalability import run_scalability


def _make_logger():
    logger = logging.getLogger("smoke_test")
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TestExperimentsSmoke(unittest.TestCase):
    """Each experiment runs with tiny parameters to verify no crashes."""

    def _tmpdir(self):
        d = Path(tempfile.mkdtemp())
        for sub in ("raw", "tables", "figures", "logs"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def test_correctness_smoke(self):
        cfg = {
            "correctness": {
                "n_values": [5],
                "t_values": [2],
                "L_values": [1, 2],
                "Nmax_values": [64],
                "num_trials": 5,
                "fig3": {
                    "n": 5, "t": 2,
                    "Ls_A": [1, 2], "Nmaxs_A": [64], "trials_A": 5,
                    "Ls_B": [2], "Nmax_B": 64, "trials_B": 5,
                    "Ls_C": [1, 2], "Nmax_C": 64, "trials_C": 5,
                    "success_rate": {
                        "L_values": [2],
                        "Nmax_values": [16, 64],
                        "num_trials": 5,
                    }
                }
            }
        }
        run_correctness(cfg, self._tmpdir(), _make_logger())

    def test_security_smoke(self):
        cfg = {
            "security": {
                "uniformity": {
                    "n": 5, "t": 2, "Nmax": 64,
                    "L_values": [2], "num_samples": 20,
                },
                "dropout": {
                    "L": 2, "Nmax": 64,
                    "combos": [{"n": 5, "t": 2}],
                    "num_trials": 5,
                },
                "forgery": {
                    "combos": [{"n": 5, "t": 3}],
                    "num_trials": 5,
                },
            }
        }
        run_security(cfg, self._tmpdir(), _make_logger())

    def test_efficiency_smoke(self):
        cfg = {
            "efficiency": {
                "num_trials": 3,
                "breakdown_n": 5, "breakdown_t": 2,
                "breakdown_L": 2, "breakdown_Nmax": 64,
                "L_values": [2], "t_values_for_L": [2],
                "fixed_n_for_L": 5,
                "n_values_for_t": [5],
                "fixed_L_for_t": 2,
                "Nmax": 64,
                "trials_per_combo": 3,
            }
        }
        run_efficiency(cfg, self._tmpdir(), _make_logger())

    def test_comparison_smoke(self):
        cfg = {
            "comparison": {
                "n": 5, "t": 2,
                "L_values": [2],
                "Nmax": 64,
                "num_trials": 3,
            }
        }
        run_comparison(cfg, self._tmpdir(), _make_logger())

    def test_scalability_smoke(self):
        cfg = {
            "scalability": {
                "Nmax": 64,
                "num_trials": 3,
                "combos": [{"n": 5, "t": 2, "L": 2}],
            }
        }
        run_scalability(cfg, self._tmpdir(), _make_logger())


if __name__ == "__main__":
    unittest.main()