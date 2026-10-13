import numpy as np
import pytest
from scipy.optimize import brentq, root

from nummeth import fd_jacobian, newton_system

# A two-diode network used as a stiff, exponential test problem.
VS, R1, R2, IS, NVT = 5.0, 1000.0, 2000.0, 1e-12, 0.02585


def network(v):
    v1, v2 = v
    i_d1 = IS * np.expm1((v1 - v2) / NVT)
    i_d2 = IS * np.expm1(v1 / NVT)
    return np.array([(v1 - VS) / R1 + i_d1 + i_d2,
                     v2 / R2 - i_d1])


def network_jac(v):
    v1, v2 = v
    g = IS / NVT
    e1 = np.exp((v1 - v2) / NVT)
    e2 = np.exp(v1 / NVT)
    return np.array([[1.0 / R1 + g * e1 + g * e2, -g * e1],
                     [-g * e1, 1.0 / R2 + g * e1]])


def circle_line(v):
    return np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] - v[1]])


def test_circle_and_line():
    r = newton_system(circle_line, [1.0, 1.5])
    assert r.converged
    assert np.allclose(r.x, [np.sqrt(2.0), np.sqrt(2.0)])


def test_matches_scipy_levenberg_marquardt_on_the_diode_network():
    # SciPy's default "hybr" method stalls from this start on this badly scaled
    # exponential system, so "lm" is used as the SciPy reference.
    x0 = [0.5, 0.0]
    mine = newton_system(network, x0, jac=network_jac, tol=1e-12)
    ref = root(network, x0, method="lm")
    assert mine.converged and ref.success
    assert np.allclose(mine.x, ref.x, atol=1e-6)


def test_matches_an_independent_one_dimensional_reduction():
    # Eliminate v2 with an inner root-find, then solve for v1 with an outer one.
    def v2_of(v1):
        return brentq(lambda v2: v2 / R2 - IS * np.expm1((v1 - v2) / NVT),
                      -1.0, v1, xtol=1e-15)

    v1 = brentq(lambda v: network([v, v2_of(v)])[0], 0.1, 1.0, xtol=1e-15)
    mine = newton_system(network, [0.5, 0.0], jac=network_jac, tol=1e-12)
    assert mine.converged
    assert np.allclose(mine.x, [v1, v2_of(v1)], atol=1e-8)


def test_finite_difference_jacobian_matches_analytic():
    v = np.array([0.55, 0.12])
    assert np.allclose(fd_jacobian(network, v), network_jac(v), rtol=1e-5)


def test_numerical_and_analytic_jacobians_agree_on_the_solution():
    a = newton_system(network, [0.5, 0.0], jac=network_jac)
    b = newton_system(network, [0.5, 0.0])
    assert a.converged and b.converged
    assert np.allclose(a.x, b.x, atol=1e-8)


def test_damping_rescues_a_poor_starting_point():
    r = newton_system(network, [2.0, 1.0], jac=network_jac, damping=True, max_iter=200)
    assert r.converged


def test_singular_jacobian_is_reported_not_raised():
    F = lambda v: np.array([v[0] ** 2 - 1.0, v[1] - 1.0])
    r = newton_system(F, [0.0, 0.0])
    assert not r.converged


def test_residual_history_tracks_iterations():
    r = newton_system(circle_line, [1.0, 1.5])
    assert len(r.residual_history) == len(r.history)
    assert r.residual_history[-1] < 1e-10
