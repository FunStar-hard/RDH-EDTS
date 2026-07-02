"""Test rejection sampling behavior."""
import secrets
import unittest

from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb


class TestRejectionSampling(unittest.TestCase):
    def test_small_L_succeeds(self):
        """L=1 should almost always succeed within Nmax=256."""
        sr = setup(2, 5, 1)
        successes = 0
        for _ in range(50):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(2)
            result = sign_emb(
                m=m, participants=[1, 2],
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=1, Nmax=256,
            )
            if result.success:
                successes += 1
        self.assertGreater(successes, 40)  # should be close to 50

    def test_large_L_small_Nmax_can_fail(self):
        """L=8 with Nmax=4 should fail frequently."""
        sr = setup(2, 5, 8)
        failures = 0
        for _ in range(20):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(256)
            result = sign_emb(
                m=m, participants=[1, 2],
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=8, Nmax=4,
            )
            if not result.success:
                failures += 1
        self.assertGreater(failures, 10)

    def test_retries_in_expected_range(self):
        """For L=2, mean retries should be around 4."""
        sr = setup(2, 5, 2)
        retries_list = []
        for _ in range(100):
            m = secrets.token_bytes(32)
            M = secrets.randbelow(4)
            result = sign_emb(
                m=m, participants=[1, 2],
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=2, Nmax=256,
            )
            if result.success:
                retries_list.append(result.retries)
        self.assertGreater(len(retries_list), 80)
        import numpy as np
        mean_r = np.mean(retries_list)
        # Expected ~4, allow range [2, 8]
        self.assertGreater(mean_r, 2.0)
        self.assertLess(mean_r, 8.0)


if __name__ == "__main__":
    unittest.main()