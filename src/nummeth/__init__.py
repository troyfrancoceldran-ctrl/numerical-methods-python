"""nummeth: from-scratch numerical methods for root-finding, linear systems
and nonlinear systems.

The solvers here do not call SciPy. SciPy and NumPy are used only in the tests
and notebooks to cross-check results.
"""

from .linsys import (
    SingularMatrixError,
    condition_number,
    determinant,
    gauss_solve,
    inverse,
)
from .nonlinear import SystemResult, fd_jacobian, newton_system
from .rootfinding import RootResult, bisection, newton_raphson, secant

__all__ = [
    "RootResult",
    "bisection",
    "secant",
    "newton_raphson",
    "SingularMatrixError",
    "gauss_solve",
    "determinant",
    "inverse",
    "condition_number",
    "SystemResult",
    "fd_jacobian",
    "newton_system",
]

__version__ = "0.1.0"
