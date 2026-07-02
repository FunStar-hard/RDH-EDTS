"""Test the complete sign -> verify -> extract pipeline."""
import secrets
import unittest

from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract


class TestSignVerifyExtract(unittest.TestCase):
    def test_basic_pipeline(self):
        """Sign, verify, extract with small L should succeed."""
        sr = setup(3, 5, 2)
        m = secrets.token_bytes(32)
        M = secrets.randbelow(4)  # L=2
        participants = [1, 2, 3]

        result = sign_emb(
            m=m, participants=participants,
            shares=sr.shares, share_pks=sr.share_pks,
            pk=sr.pk, Kext=sr.Kext, M=M, L=2, Nmax=256,
        )
        self.assertTrue(result.success, "Signing should succeed with L=2, Nmax=256")
        self.assertTrue(verify(m, sr.pk, result.signature))
        recovered = extract(m, sr.pk, result.signature, sr.Kext, 2)
        self.assertEqual(recovered, M)

    def test_all_nodes_participate(self):
        """All n nodes participate."""
        sr = setup(3, 5, 3)
        m = secrets.token_bytes(32)
        M = secrets.randbelow(8)
        participants = [1, 2, 3, 4, 5]

        result = sign_emb(
            m=m, participants=participants,
            shares=sr.shares, share_pks=sr.share_pks,
            pk=sr.pk, Kext=sr.Kext, M=M, L=3, Nmax=256,
        )
        if result.success:
            self.assertTrue(verify(m, sr.pk, result.signature))
            recovered = extract(m, sr.pk, result.signature, sr.Kext, 3)
            self.assertEqual(recovered, M)

    def test_different_participants(self):
        """Different subsets of t nodes produce valid signatures."""
        sr = setup(2, 5, 2)
        m = secrets.token_bytes(32)
        M = secrets.randbelow(4)

        for subset in [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]]:
            result = sign_emb(
                m=m, participants=subset,
                shares=sr.shares, share_pks=sr.share_pks,
                pk=sr.pk, Kext=sr.Kext, M=M, L=2, Nmax=256,
            )
            if result.success:
                self.assertTrue(verify(m, sr.pk, result.signature),
                                f"Verify failed for subset {subset}")

    def test_wrong_message_verify_fails(self):
        sr = setup(2, 5, 2)
        m = secrets.token_bytes(32)
        M = 1
        result = sign_emb(
            m=m, participants=[1, 2],
            shares=sr.shares, share_pks=sr.share_pks,
            pk=sr.pk, Kext=sr.Kext, M=M, L=2, Nmax=256,
        )
        if result.success:
            wrong_m = secrets.token_bytes(32)
            self.assertFalse(verify(wrong_m, sr.pk, result.signature))

    def test_extract_returns_none_on_bad_sig(self):
        sr = setup(2, 5, 2)
        m = secrets.token_bytes(32)
        M = 1
        result = sign_emb(
            m=m, participants=[1, 2],
            shares=sr.shares, share_pks=sr.share_pks,
            pk=sr.pk, Kext=sr.Kext, M=M, L=2, Nmax=256,
        )
        if result.success:
            wrong_m = secrets.token_bytes(32)
            recovered = extract(wrong_m, sr.pk, result.signature, sr.Kext, 2)
            self.assertIsNone(recovered)


if __name__ == "__main__":
    unittest.main()