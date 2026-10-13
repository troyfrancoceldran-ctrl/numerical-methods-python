"""Newton's method for systems of nonlinear equations F(x) = 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .linsys import SingularMatrixError, gauss_solve

VectorFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class SystemResult:
    """Outcome of a Newton run on a system."""

    x: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float  # infinity norm of F(x)
    history: List[np.ndarray] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)


def fd_jacobian(F: VectorFunction, x, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``F`` at ``x``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((np.asarray(F(x)).size, n))
    for j in range(n):
        step = h * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        J[:, j] = (np.asarray(F(xp)) - np.asarray(F(xm))) / (2.0 * step)
    return J


def _norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def newton_system(F: VectorFunction, x0, jac: Optional[VectorFunction] = None,
                  tol: float = 1e-10, max_iter: int = 50,
                  damping: bool = True) -> SystemResult:
    """Solve ``F(x) = 0`` by Newton's method.

    ``jac`` is an optional analytic Jacobian; otherwise a central-difference
    Jacobian is used. With ``damping=True`` a backtracking line search halves
    the step until the residual decreases, which keeps exponential
    nonlinearities from blowing up on a poor starting point.
    """
    x = np.array(x0, dtype=float, copy=True)
    history = [x.copy()]
    Fx = np.asarray(F(x), dtype=float)
    r = _norm(Fx)
    residuals = [r]

    for k in range(max_iter):
        if r < tol:
            return SystemResult(x, k, True, r, history, residuals)
        J = jac(x) if jac is not None else fd_jacobian(F, x)
        try:
            dx = gauss_solve(J, -Fx)
        except SingularMatrixError:
            return SystemResult(x, k, False, r, history, residuals)

        step = 1.0
        x_new = x + dx
        F_new = np.asarray(F(x_new), dtype=float)
        r_new = _norm(F_new)
        if damping:
            while (not np.isfinite(r_new) or r_new >= (1 - 1e-4 * step) * r) and step > 1e-10:
                step *= 0.5
                x_new = x + step * dx
                F_new = np.asarray(F(x_new), dtype=float)
                r_new = _norm(F_new)
            if not np.isfinite(r_new):
                return SystemResult(x, k, False, r, history, residuals)
        elif not np.isfinite(r_new):
            history.append(x_new.copy())
            return SystemResult(x_new, k + 1, False, r_new, history, residuals + [r_new])

        x, Fx, r = x_new, F_new, r_new
        history.append(x.copy())
        residuals.append(r)

    return SystemResult(x, max_iter, r < tol, r, history, residuals)
