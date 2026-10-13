import math

import pytest
from scipy.optimize import brentq

from nummeth import bisection, newton_raphson, secant


def sq2(x):
    return x * x - 2.0


def test_bisection_finds_sqrt2():
    r = bisection(sq2, 0.0, 2.0, tol=1e-12)
    assert r.converged
    assert r.root == pytest.approx(math.sqrt(2.0), abs=1e-11)


def test_bisection_iteration_bound():
    a, b, tol = 0.0, 2.0, 1e-8
    r = bisection(sq2, a, b, tol=tol)
    assert r.iterations <= math.ceil(math.log2((b - a) / tol)) + 1


def test_bisection_requires_a_sign_change():
    with pytest.raises(ValueError):
        bisection(lambda x: x * x + 1.0, -1.0, 1.0)


def test_bisection_root_at_endpoint():
    r = bisection(lambda x: x, 0.0, 1.0)
    assert r.converged and r.root == 0.0 and r.iterations == 0


def test_bisection_accepts_swapped_interval():
    r = bisection(sq2, 2.0, 0.0, tol=1e-10)
    assert r.root == pytest.approx(math.sqrt(2.0), abs=1e-9)


def test_bisection_reports_non_convergence():
    r = bisection(sq2, 0.0, 2.0, tol=1e-15, max_iter=5)
    assert not r.converged and r.iterations == 5


def test_secant_matches_brentq():
    f = lambda x: math.cos(x) - x
    ref = brentq(f, 0.0, 1.0, xtol=1e-14)
    r = secant(f, 0.0, 1.0)
    assert r.converged
    assert r.root == pytest.approx(ref, abs=1e-9)


def test_secant_flat_line_raises():
    with pytest.raises(ZeroDivisionError):
        secant(lambda x: 1.0, 0.0, 1.0)


def test_newton_converges_quadratically():
    r = newton_raphson(sq2, lambda x: 2.0 * x, 1.0, tol=1e-14)
    assert r.converged
    errors = [abs(x - math.sqrt(2.0)) for x in r.history]
    # For f = x^2 - 2 the asymptotic constant is 1 / (2 sqrt 2) ~ 0.354.
    ratios = [errors[k + 1] / errors[k] ** 2
              for k in range(len(errors) - 1) if 1e-6 < errors[k] < 0.5]
    assert ratios and all(ratio < 1.0 for ratio in ratios)


def test_newton_zero_derivative_raises():
    with pytest.raises(ZeroDivisionError):
        newton_raphson(sq2, lambda x: 2.0 * x, 0.0)


def test_newton_diverges_on_cube_root():
    # x_{k+1} = -2 x_k for f(x) = cbrt(x): a classic failure of Newton's method.
    f = lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x)
    df = lambda x: (1.0 / 3.0) * abs(x) ** (-2.0 / 3.0)
    r = newton_raphson(f, df, 1.0, max_iter=30)
    assert not r.converged
    assert abs(r.history[-1]) > abs(r.history[0])


@pytest.mark.parametrize("bad", [0.0, -1e-9])
def test_invalid_tolerance_rejected(bad):
    with pytest.raises(ValueError):
        bisection(sq2, 0.0, 2.0, tol=bad)
