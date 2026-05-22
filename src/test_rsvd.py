"""Unit tests for randomized_svd."""

import os
import sys
import unittest

import numpy as np
from scipy import sparse

# Allow running this file directly: python src/test_rsvd.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsvd import randomized_svd


class TestRandomizedSVD(unittest.TestCase):

    def test_singular_values_dense(self):
        """Top-k singular values match numpy.linalg.svd within 1e-3 relative."""
        rng = np.random.default_rng(0)
        m, n, k = 100, 80, 10

        # Build a matrix with quickly decaying spectrum so RSVD is accurate.
        U_true, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V_true, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s_true = np.exp(-0.5 * np.arange(min(m, n)))
        A = (U_true[:, :n] * s_true) @ V_true.T

        U, s, Vt = randomized_svd(A, k=k, p=10, q=2, random_state=42)

        self.assertEqual(U.shape, (m, k))
        self.assertEqual(s.shape, (k,))
        self.assertEqual(Vt.shape, (k, n))

        s_ref = np.linalg.svd(A, compute_uv=False)[:k]
        rel_err = np.abs(s - s_ref) / s_ref
        self.assertTrue(
            np.all(rel_err < 1e-3),
            f"Relative errors too large: {rel_err}",
        )

    def test_sparse_frobenius_error(self):
        """Frobenius error on sparse input is close to optimal truncation."""
        rng = np.random.default_rng(1)
        m, n, k = 200, 150, 20
        density = 0.05

        A_sparse = sparse.random(
            m, n, density=density, format="csr", random_state=rng
        )
        A_dense = A_sparse.toarray()

        U, s, Vt = randomized_svd(A_sparse, k=k, p=10, q=4, random_state=7)

        approx = U @ np.diag(s) @ Vt
        err = np.linalg.norm(A_dense - approx, "fro")
        norm_A = np.linalg.norm(A_dense, "fro")

        # Compare against the optimal rank-k Frobenius error (Eckart-Young).
        s_full = np.linalg.svd(A_dense, compute_uv=False)
        optimal_err = np.sqrt(np.sum(s_full[k:] ** 2))

        # RSVD error should not exceed ~1.5x the optimal error for this setting.
        self.assertLess(err, 1.5 * optimal_err + 1e-8)
        # Relative reconstruction error should be modest.
        self.assertLess(err / norm_A, 1.0)

    def test_edge_case_k_full(self):
        """k = min(m, n) should reconstruct A almost exactly."""
        rng = np.random.default_rng(2)
        m, n = 30, 40
        A = rng.standard_normal((m, n))
        k = min(m, n)

        U, s, Vt = randomized_svd(A, k=k, p=5, q=2, random_state=3)
        self.assertEqual(U.shape, (m, k))
        self.assertEqual(s.shape, (k,))
        self.assertEqual(Vt.shape, (k, n))

        recon = U @ np.diag(s) @ Vt
        self.assertTrue(np.allclose(A, recon, atol=1e-8))

    def test_edge_case_k_one(self):
        """k = 1 should approximate the leading singular triplet."""
        rng = np.random.default_rng(3)
        m, n = 50, 60
        # Use a matrix with a clear spectral gap so a rank-1 sketch is accurate.
        U_true, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V_true, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s_true = np.array([10.0] + [0.1] * (min(m, n) - 1))
        A = (U_true[:, : min(m, n)] * s_true) @ V_true[:, : min(m, n)].T

        U, s, Vt = randomized_svd(A, k=1, p=10, q=3, random_state=5)
        self.assertEqual(U.shape, (m, 1))
        self.assertEqual(s.shape, (1,))
        self.assertEqual(Vt.shape, (1, n))

        s_ref = np.linalg.svd(A, compute_uv=False)[0]
        self.assertLess(abs(s[0] - s_ref) / s_ref, 1e-3)

    def test_orthonormality(self):
        """U columns and Vt rows should be (approximately) orthonormal."""
        rng = np.random.default_rng(4)
        A = rng.standard_normal((60, 45))
        U, s, Vt = randomized_svd(A, k=10, p=10, q=2, random_state=11)

        self.assertTrue(
            np.allclose(U.T @ U, np.eye(10), atol=1e-8),
            "U columns not orthonormal",
        )
        self.assertTrue(
            np.allclose(Vt @ Vt.T, np.eye(10), atol=1e-8),
            "Vt rows not orthonormal",
        )
        self.assertTrue(np.all(s[:-1] >= s[1:]), "Singular values not sorted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
