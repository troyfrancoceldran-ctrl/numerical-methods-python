import numpy as np
import pytest
from scipy.linalg import hilbert

from nummeth import (
    SingularMatrixError,
    condition_number,
    determinant,
    gauss_solve,
    inverse,
)


@pytest.mark.parametrize("n", [2, 5, 12])
def test_matches_numpy_solve(n):
    rng = np.random.default_rng(n)
    A = rng.normal(size=(n, n)) + n * np.eye(n)
    b = rng.normal(size=n)
    assert np.allclose(gauss_solve(A, b), np.linalg.solve(A, b))


def test_multiple_right_hand_sides():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    B = rng.normal(size=(4, 3))
    assert np.allclose(gauss_solve(A, B), np.linalg.solve(A, B))


def test_inputs_are_not_modified():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([3.0, 5.0])
    A0, b0 = A.copy(), b.copy()
    gauss_solve(A, b)
    assert np.array_equal(A, A0) and np.array_equal(b, b0)


def test_singular_matrix_raises():
    with pytest.raises(SingularMatrixError):
        gauss_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_non_square_rejected():
    with pytest.raises(ValueError):
        gauss_solve(np.ones((2, 3)), np.ones(2))


def test_pivoting_is_needed_for_a_tiny_leading_pivot():
    A = [[1e-20, 1.0], [1.0, 1.0]]
    b = [1.0, 2.0]
    reference = np.linalg.solve(A, b)
    assert np.allclose(gauss_solve(A, b), reference)
    unpivoted = gauss_solve(A, b, pivoting=False)
    assert not np.allclose(unpivoted, reference, atol=1e-6)


def test_determinant_matches_numpy():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    assert determinant(A) == pytest.approx(np.linalg.det(A), rel=1e-9)


def test_determinant_sign_after_row_swap():
    assert determinant([[0.0, 1.0], [1.0, 0.0]]) == pytest.approx(-1.0)


def test_determinant_of_singular_matrix_is_zero():
    assert determinant([[1.0, 2.0], [2.0, 4.0]]) == 0.0


def test_inverse_times_matrix_is_identity():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(5, 5)) + 5 * np.eye(5)
    assert np.allclose(A @ inverse(A), np.eye(5))


def test_condition_number_of_hilbert_matrix():
    H = hilbert(6)
    assert condition_number(H) == pytest.approx(np.linalg.cond(H, 1), rel=1e-4)


def test_condition_number_of_singular_matrix_is_infinite():
    assert condition_number([[1.0, 2.0], [2.0, 4.0]]) == float("inf")


def test_badly_scaled_but_nonsingular_system_is_solved():
    # Entries span 20 orders of magnitude, like Jacobians of exponential models.
    # A test scaled by the whole matrix would wrongly call this singular.
    A = [[1e20, 1.0], [1.0, 1.0]]
    b = [1e20, 2.0]
    assert np.allclose(gauss_solve(A, b), np.linalg.solve(A, b))
