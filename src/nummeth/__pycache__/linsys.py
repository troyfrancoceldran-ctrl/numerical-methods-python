"""Dense linear systems by Gaussian elimination.

The elimination loops are written out explicitly. NumPy arrays are used only
as containers and for norms.
"""

from __future__ import annotations

import numpy as np


class SingularMatrixError(ValueError):
    """Raised when elimination meets a (numerically) zero pivot."""


def _as_square(A) -> np.ndarray:
    A = np.array(A, dtype=float, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array")
    return A


def gauss_solve(A, b, pivoting: bool = True, pivot_tol: float = 1e-13) -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination.

    With ``pivoting=True`` (default) the largest entry in each column is used
    as the pivot (partial pivoting). ``pivoting=False`` is provided only to
    demonstrate why pivoting matters; it fails only on an exactly zero pivot.

    ``b`` may be a vector of length n or an n-by-k matrix of right-hand sides.
    Raises :class:`SingularMatrixError` when a pivot is smaller than
    ``pivot_tol`` times the largest entry of its own original row. Scaling by
    the row (not the whole matrix) keeps badly scaled but non-singular
    systems, such as those with exponential terms, from being rejected.
    """
    A = _as_square(A)
    b = np.array(b, dtype=float, copy=True)
    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError("b must have the same number of rows as A")
    row_scale = np.max(np.abs(A), axis=1) if n else np.array([])
    if np.any(row_scale == 0.0):
        raise SingularMatrixError("matrix has a zero row")

    for k in range(n):
        if pivoting:
            p = k + int(np.argmax(np.abs(A[k:, k])))
            if p != k:
                A[[k, p]] = A[[p, k]]
                b[[k, p]] = b[[p, k]]
                row_scale[[k, p]] = row_scale[[p, k]]
            threshold = pivot_tol * row_scale[k]
        else:
            threshold = 0.0
        if abs(A[k, k]) <= threshold:
            raise SingularMatrixError(f"zero pivot in column {k}")
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def determinant(A, pivot_tol: float = 1e-13) -> float:
    """Determinant from the elimination pivots (with the row-swap sign).

    Returns 0.0 when a pivot falls below ``pivot_tol`` times the largest entry
    of its own original row (the same test as :func:`gauss_solve`).
    """
    A = _as_square(A)
    n = A.shape[0]
    row_scale = np.max(np.abs(A), axis=1) if n else np.array([])
    if np.any(row_scale == 0.0):
        return 0.0
    sign = 1.0
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            row_scale[[k, p]] = row_scale[[p, k]]
            sign = -sign
        if abs(A[k, k]) <= pivot_tol * row_scale[k]:
            return 0.0
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
    return float(sign * np.prod(np.diag(A)))


def inverse(A) -> np.ndarray:
    """Inverse by solving ``A X = I``."""
    A = _as_square(A)
    return gauss_solve(A, np.eye(A.shape[0]))


def condition_number(A) -> float:
    """1-norm condition number ``||A||_1 * ||A^-1||_1``; ``inf`` if singular."""
    A = _as_square(A)
    try:
        A_inv = inverse(A)
    except SingularMatrixError:
        return float("inf")

    def norm1(M: np.ndarray) -> float:
        return float(np.max(np.sum(np.abs(M), axis=0)))

    return norm1(A) * norm1(A_inv)
