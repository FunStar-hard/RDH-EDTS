"""Test that Setup produces shares that reconstruct the system secret."""
import unittest
from src.scheme.setup import setup
from src.crypto.shamir import reconstruct_secret
from src.crypto.curve_utils import get_order, scalar_mult, get_generator


class TestSetup(unittest.TestCase):
    def test_share_reconstruction(self):
        """t shares can reconstruct the system secret."""
        for t, n in [(2, 5), (3, 5), (5, 10)]:
            sr = setup(t, n, 4)
            q = get_order()
            # Take first t shares
            subset = {vi: sr.shares[vi] for vi in list(sr.shares.keys())[:t]}
            recovered = reconstruct_secret(subset, q)
            self.assertEqual(recovered, sr.system_secret,
                             f"Failed for t={t}, n={n}")

    def test_pk_matches(self):
        """pk = x * G."""
        sr = setup(3, 5, 4)
        expected_pk = scalar_mult(sr.system_secret, get_generator())
        self.assertEqual(sr.pk, expected_pk)

    def test_share_pk_matches(self):
        """pk_i = x_i * G for each share."""
        sr = setup(3, 5, 4)
        G = get_generator()
        for vi, xi in sr.shares.items():
            expected = scalar_mult(xi, G)
            self.assertEqual(sr.share_pks[vi], expected)

    def test_kext_length(self):
        sr = setup(3, 5, 4)
        self.assertEqual(len(sr.Kext), 32)


if __name__ == "__main__":
    unittest.main()