"""Scalar root-finding: bisection, secant and Newton-Raphson.

Every solver returns a :class:`RootResult` that records the sequence of
iterates, so convergence can be plotted and compared between methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List

Function = Callable[[float], float]


@dataclass
class RootResult:
    """Outcome of a root-finding run."""

    root: float
    iterations: int
    converged: bool
    residual: float  # |f(root)|
    history: List[float] = field(default_factory=list)  # successive iterates
    method: str = ""


def _validate(tol: float, max_iter: int) -> None:
    if not tol > 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")


def bisection(f: Function, a: float, b: float, tol: float = 1e-10,
              max_iter: int = 200) -> RootResult:
    """Find a root of ``f`` in ``[a, b]`` by interval halving.

    Requires ``f(a)`` and ``f(b)`` to have opposite signs. Stops when the
    half-width of the bracket is below ``tol``, so the returned root is within
    ``tol`` of a true root. The number of iterations is at most about
    ``ceil(log2((b - a) / tol))``.
    """
    _validate(tol, max_iter)
    if a > b:
        a, b = b, a
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return RootResult(a, 0, True, 0.0, [a], "bisection")
    if fb == 0.0:
        return RootResult(b, 0, True, 0.0, [b], "bisection")
    if (fa > 0) == (fb > 0):
        raise ValueError("f(a) and f(b) must have opposite signs")

    history: List[float] = []
    c, fc = a, fa
    for k in range(1, max_iter + 1):
        c = a + (b - a) / 2.0
        fc = f(c)
        history.append(c)
        if fc == 0.0 or (b - a) / 2.0 < tol:
            return RootResult(c, k, True, abs(fc), history, "bisection")
        if (fa > 0) == (fc > 0):
            a, fa = c, fc
        else:
            b = c
    return RootResult(c, max_iter, False, abs(fc), history, "bisection")


def secant(f: Function, x0: float, x1: float, tol: float = 1e-10,
           max_iter: int = 100) -> RootResult:
    """Secant method: Newton's method with the derivative replaced by a
    finite difference through the last two iterates. No bracket is needed,
    but convergence is not guaranteed.

    Raises ``ZeroDivisionError`` if two consecutive function values are equal
    (a flat secant line).
    """
    _validate(tol, max_iter)
    f0, f1 = f(x0), f(x1)
    history = [x0, x1]
    if f1 == 0.0:
        return RootResult(x1, 0, True, 0.0, history, "secant")
    for k in range(1, max_iter + 1):
        if f1 == f0:
            raise ZeroDivisionError("flat secant: f(x_{k-1}) == f(x_k)")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if not math.isfinite(x2):
            return RootResult(x2, k, False, math.inf, history, "secant")
        f2 = f(x2)
        if f2 == 0.0 or abs(x2 - x1) <= tol * max(1.0, abs(x2)):
            return RootResult(x2, k, True, abs(f2), history, "secant")
        x0, f0, x1, f1 = x1, f1, x2, f2
    return RootResult(x1, max_iter, False, abs(f1), history, "secant")


def newton_raphson(f: Function, df: Function, x0: float, tol: float = 1e-10,
                   max_iter: int = 50) -> RootResult:
    """Newton-Raphson iteration ``x <- x - f(x) / f'(x)``.

    Converges quadratically near a simple root but can diverge from a poor
    starting point. Raises ``ZeroDivisionError`` if the derivative is zero at
    an iterate.
    """
    _validate(tol, max_iter)
    x = x0
    fx = f(x)
    history = [x]
    for k in range(1, max_iter + 1):
        if fx == 0.0:
            return RootResult(x, k - 1, True, 0.0, history, "newton-raphson")
        d = df(x)
        if d == 0.0:
            raise ZeroDivisionError(f"zero derivative at x = {x!r}")
        step = fx / d
        x = x - step
        history.append(x)
        if not math.isfinite(x):
            return RootResult(x, k, False, math.inf, history, "newton-raphson")
        fx = f(x)
        if abs(step) <= tol * max(1.0, abs(x)):
            return RootResult(x, k, True, abs(fx), history, "newton-raphson")
    return RootResult(x, max_iter, False, abs(fx), history, "newton-raphson")
