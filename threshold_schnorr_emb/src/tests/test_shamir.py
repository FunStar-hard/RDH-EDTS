"""Test Shamir secret sharing."""
import unittest
from src.crypto.shamir import share_secret, reconstruct_secret
from src.crypto.curve_utils import get_order


class TestShamir(unittest.TestCase):
    def test_reconstruct_exact_threshold(self):
        q = get_order()
        secret = 123456789
        _, shares = share_secret(secret, 3, 5, q)
        subset = {1: shares[1], 2: shares[2], 3: shares[3]}
        recovered = reconstruct_secret(subset, q)
        self.assertEqual(recovered, secret)

    def test_reconstruct_more_than_threshold(self):
        q = get_order()
        secret = 987654321
        _, shares = share_secret(secret, 3, 5, q)
        recovered = reconstruct_secret(shares, q)
        self.assertEqual(recovered, secret)

    def test_different_subsets(self):
        q = get_order()
        secret = 42
        _, shares = share_secret(secret, 2, 5, q)
        for a in range(1, 6):
            for b in range(a + 1, 6):
                subset = {a: shares[a], b: shares[b]}
                recovered = reconstruct_secret(subset, q)
                self.assertEqual(recovered, secret)

    def test_below_threshold_fails(self):
        """With < t shares, reconstruction gives wrong value (almost certainly)."""
        q = get_order()
        secret = 999
        _, shares = share_secret(secret, 3, 5, q)
        subset = {1: shares[1]}
        recovered = reconstruct_secret(subset, q)
        # Almost certainly not equal
        self.assertNotEqual(recovered, secret)


if __name__ == "__main__":
    unittest.main()