# AGENTS.md

Guidance for AI agents and contributors working in this repository.

## Project

FEM-based analysis tool for Yagi antenna elements (telescoping aluminum tubes).
Goal: compute loads along an element (bending moment, shear, stress, deflection)
under wind (per a recognized standard), self-weight, ice, and point loads, and
report the maximum wind speed the element survives for each failure mode.

## Development workflow

- **TDD, small steps.** Write a failing test against a known analytic/closed-form
  result, implement, make it green, refactor. Commit after each step.
- Commit often; each commit should be green (tests pass).
- Work incrementally — do not one-shot entire features.

## Commands

All commands use the project virtual environment:

```
.venv\Scripts\python.exe -m pytest          # run tests
.venv\Scripts\python.exe -m pytest tests\test_geometry.py -k tube   # subset
```

On non-Windows shells the venv python is `.venv/bin/python`.

## Conventions

- Package code lives in `elementload/`, tests in `tests/`.
- Tests use pytest, plain asserts, no external test deps.
- No comments in code unless needed to explain non-obvious physics.
- Units: SI (m, N, Pa, m/s). Wind speed in m/s internally, km/h only in output.
- Solver is decoupled behind a clean interface so the backend (scipy vs. OpenSees)
  can be swapped without touching geometry/wind/failure code.

## Physical models (authoritative references)

- Beam FEM: 2-node Euler-Bernoulli elements, stepped sections with own EI.
- Wind: drag per unit length q = 0.5*rho*V^2*Cd*D, Re-dependent Cd per
  EN 1991-1-4 circular cylinder chart; gust/height/terrain factors.
- Telescoping joints: analyze rigid-overlap and slipped/pinned-overlap bounds.
- Supports: fixed / pin / roller at arbitrary nodes (multiple allowed).

## Roadmap

1. geometry: tube section properties (I, Z, A, mass/len) — in progress
2. wind: drag per unit length, Re-dependent Cd
3. solver: beam FEM assemble/solve, V/M/delta (closed-form golden tests)
4. supports: arbitrary support list, redundant-reaction tests
5. assembly: stepped/telescoping element, overlap stiffness bounds
6. failure: yield, local wall crippling, joint pin shear
7. report: survival wind speed per mode + diagrams (CLI acceptance test)
8. later: dynamics (modal / vortex shedding), fatigue, multi-element model
