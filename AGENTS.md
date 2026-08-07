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
.venv-osees\Scripts\python.exe -m pytest tests\test_opensees_crosscheck.py  # OpenSees cross-check
```

On non-Windows shells the venv python is `.venv/bin/python`.

The OpenSees cross-check needs its own venv because the `openseespywin` wheel ships
a `.pyd` built against `python312.dll`; run it under the Python 3.12 venv
(`.venv-osees`, created with `C:\Python312\python.exe -m venv .venv-osees`).
The test skips automatically under other interpreters.

## Conventions

- Package code lives in `elementload/`, tests in `tests/`.
- Tests use pytest, plain asserts, no external test deps.
- No comments in code unless needed to explain non-obvious physics.
- Units: SI (m, N, Pa, m/s). Wind speed in m/s internally, km/h only in output.
- Solver is decoupled behind a clean interface so the backend (scipy vs. OpenSees)
  can be swapped without touching geometry/wind/failure code.

## Physical models (authoritative references)

- Beam FEM: 2-node Euler-Bernoulli elements, stepped sections with own EI.
  Cross-checked against an independent FE engine (OpenSees 3.8 elasticBeamColumn)
  for a stepped telescoping cantilever under wind+weight load; node deflections
  and max end moment agree to <1e-6 relative (both rigid and slipped joint models).
- Wind: drag per unit length q = 0.5*rho*V^2*Cd*D, Re-dependent Cd per
  EN 1991-1-4 circular cylinder chart; gust/height/terrain factors.
  Cd chart in `elementload/wind.py` follows EN 1991-1-4 Fig. 7.28 for a smooth
  circular cylinder: subcritical plateau 1.2 (Wieselsberger, per Schewe 1983),
  drag crisis down to ~0.4, transcritical recovery. Anchored to the Dlubal
  Eurocode verification example VE0308 (Cf,0 = 0.55 at Re = 2e6).
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
