# Numerical Methods in Python

![tests](https://github.com/troyfrancoceldran-ctrl/numerical-methods-python/actions/workflows/tests.yml/badge.svg)

Python solvers for root-finding, linear systems and nonlinear systems, validated against NumPy and SciPy and applied to small battery, circuit and network problems.

> **Status:** complete as a learning project. It has the solvers, 36 automated tests, and five notebooks with written observations. It does not include hand derivations of the methods, and I did not write the solvers from scratch (see "Use of AI tools" below).

## Where this comes from

I learned these methods in a Numerical Methods course (EEE102) at Mindanao State University - Iligan Institute of Technology. This repository contains no course handouts, problem statements, diagrams, grading rubrics or exam material, and its example problems are original.

## Methods

- **Root-finding:** bisection, secant, Newton-Raphson
- **Linear systems:** Gaussian elimination with partial pivoting, determinant, inverse and 1-norm condition number
- **Nonlinear systems:** Newton's method with an analytic or finite-difference Jacobian and an optional backtracking line search

The solvers do not call SciPy. SciPy and NumPy appear only in the tests and notebooks, as independent references.

## Notebooks

| # | Notebook | Method | Problem (original) |
|---|----------|--------|--------------------|
| 01 | `01_bisection.ipynb` | Bisection | State of charge from an open-circuit-voltage model (illustrative, not a fitted cell) |
| 02 | `02_secant.ipynb` | Secant | Operating point of a series diode circuit |
| 03 | `03_newton_raphson.ipynb` | Newton-Raphson | The same SOC inversion, convergence order, and two failure cases |
| 04 | `04_linear_systems.ipynb` | Gaussian elimination | Node voltages of a resistor network; pivoting; Hilbert-matrix conditioning |
| 05 | `05_nonlinear_systems.ipynb` | Newton for systems | A two-diode network; analytic vs finite-difference Jacobian; damping |

Each notebook ends with an **Observations** cell that answers questions about the results. I wrote the first drafts, and an AI assistant corrected errors in them (see "Use of AI tools").

## Results

What the notebooks measured:

- **Bisection:** the iteration count matches the bound ceil(log2((b - a)/tol)) exactly (7, 14, 20, 27, 34 and 40 iterations for tolerances from 1e-2 to 1e-12). The actual error always stays below the (b - a)/2^k bound but zigzags instead of falling steadily.
- **Secant versus bisection:** on the diode circuit (4.4259 mA, 0.5741 V across the diode) the secant method needed 4 iterations and bisection needed 33.
- **Convergence order:** Newton-Raphson measured about 2.02 and 2.00 (quadratic). Secant measured noisy values (3.24, 1.33, 1.76, 1.57, 0.93; theory: 1.618). The middle estimates are the meaningful ones, and the last is dominated by rounding near 1e-16.
- **Failure modes:** Newton diverges on the cube root (each step gives x -> -2x) and raises an error when the derivative is zero.
- **Linear systems:** without pivoting a 2-by-2 system with a 1e-20 pivot returns [0, 1] instead of [1, 1]. On Hilbert matrices the error grows with the condition number and stays below kappa times machine epsilon. At size 12 kappa is about 4e16 and the solver refuses the matrix. The resistor network has a condition number of about 52, and a 1% resistor change moves node voltages by about 0.06%.
- **Nonlinear systems:** analytic and finite-difference Jacobians both needed 7 iterations. Damping helped from the flat side of the exponential (7 iterations against 180 from [0, 0]) and made no difference from the steep side (59 iterations from [2, 1] either way). The start [1, -1] fails because the Jacobian is numerically singular (entries near 1.5e23), not because of overflow. SciPy's `lm` converged from the standard start where `hybr` stopped with a "not making good progress" message.

## Quick start

```bash
git clone https://github.com/troyfrancoceldran-ctrl/numerical-methods-python.git
cd numerical-methods-python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs the dependencies and the nummeth package
pytest                           # runs the test suite
jupyter lab                      # then open the notebooks/ folder
```

Python 3.10 or newer.

## Repository layout

```
numerical-methods-python/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── notebooks/        # one notebook per method
├── src/nummeth/      # rootfinding.py, linsys.py, nonlinear.py
├── tests/            # pytest checks against SciPy and NumPy
└── .github/workflows/tests.yml
```

## Testing

The test suite checks, among other things:

- bisection's iteration count against its theoretical bound, and its refusal to run without a sign change
- Newton's quadratic convergence on a simple root, and its divergence on a cube root
- Gaussian elimination against `numpy.linalg.solve`, and a case that fails without pivoting
- the Newton system solver against SciPy and against an independent one-dimensional reduction

## Known limitations

- Dense matrices only, written with plain Python loops, so it is slow beyond a few hundred unknowns.
- The singular-matrix test uses a pivot tolerance relative to each row, so a severely ill-conditioned matrix (a Hilbert matrix of size 12, for example) is rejected as singular.
- Newton's method can still fail from a poor start. Damping helps from some starting points and not others (see notebook 05).
- How SciPy's `hybr` method behaves on the two-diode network may differ between SciPy versions.
- There are no hand derivations of the methods in this repository.

## Use of AI tools

The code scaffold (package structure, solvers, tests and starter notebooks) was generated with an AI assistant (Claude, Anthropic). I did not write the solvers from scratch. I ran the notebooks, which cross-check each solver against SciPy or NumPy, and I wrote the first drafts of the observation cells. Claude then corrected errors in those observations, answered questions I had skipped, and added the code cells that support them (the condition number and 1% sensitivity check in notebook 04, and the Jacobian and SciPy-message cells in notebook 05). This is disclosed in line with the MSU Policy on the Fair and Ethical Use of AI and Its Applications.

## Acknowledgments

The EEE102 course at MSU-IIT introduced these methods.

## License

MIT License (see `LICENSE`).

## Contact

Troy Franco G. Celdran - [LinkedIn](https://www.linkedin.com/in/troy-franco-celdran-aa614838a/) - [GitHub](https://github.com/troyfrancoceldran-ctrl)
