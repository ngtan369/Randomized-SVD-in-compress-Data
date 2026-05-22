"""
Randomized SVD (Halko-Martinsson-Tropp, 2011).

Implements a rank-k approximation A ~= U @ diag(s) @ Vt without invoking a
full SVD on A itself. Supports dense numpy arrays and scipy sparse matrices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _matmat(A, X):
    """Multiply A @ X for dense or sparse A, returning a dense ndarray."""
    if sparse.issparse(A):
        # sparse @ dense -> dense ndarray
        return np.asarray(A @ X)
    return A @ X


def _rmatmat(A, Y):
    """Multiply A.T @ Y for dense or sparse A, returning a dense ndarray."""
    if sparse.issparse(A):
        return np.asarray(A.T @ Y)
    return A.T @ Y


def randomized_svd(A, k, p=10, q=2, random_state=None):
    """
    Truncated SVD via random projection (Halko-Martinsson-Tropp).

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix, shape (m, n)
        Input matrix to factorize.
    k : int
        Target rank (1 <= k <= min(m, n)).
    p : int, default 10
        Oversampling parameter; sketch dimension is k + p.
    q : int, default 2
        Number of power iterations to amplify dominant singular directions.
    random_state : int or None, default None
        Seed for reproducibility.

    Returns
    -------
    U  : ndarray, shape (m, k)
    s  : ndarray, shape (k,)
    Vt : ndarray, shape (k, n)
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    m, n = A.shape
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if k > min(m, n):
        raise ValueError("k must be <= min(m, n).")

    rng = np.random.default_rng(random_state)

    # Sketch size: oversample but never exceed n.
    l = min(k + p, n)

    # Step 1: random Gaussian test matrix.
    Omega = rng.standard_normal(size=(n, l))

    # Step 2: form sketch Y = A @ Omega.
    Y = _matmat(A, Omega)

    # Step 3: subspace (power) iteration with QR re-orthonormalization at every
    # half-step. Re-orthonormalizing prevents catastrophic loss of precision
    # in the trailing singular directions when singular values decay slowly.
    for _ in range(q):
        Q, _ = np.linalg.qr(Y)
        Z = _rmatmat(A, Q)
        Q2, _ = np.linalg.qr(Z)
        Y = _matmat(A, Q2)

    # Step 4: orthonormal basis Q for range(Y).
    Q, _ = np.linalg.qr(Y)

    # Step 5: project A onto the low-dim subspace.
    B = _rmatmat(A, Q).T  # B = Q.T @ A, shape (l, n)

    # Step 6: SVD of the small matrix B.
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 7: lift left singular vectors back to original space.
    U = Q @ U_tilde

    # Step 8: truncate to k leading components.
    return U[:, :k], s[:k], Vt[:k, :]
