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

1. geometry: tube section properties (I, Z, A, mass/len) — done
2. wind: drag per unit length, Re-dependent Cd — done
3. solver: beam FEM assemble/solve, V/M/delta (closed-form golden tests) — done
4. supports: arbitrary support list, redundant-reaction tests — done
5. assembly: stepped/telescoping element, overlap stiffness bounds — done
6. failure: yield, local wall crippling, joint pin shear — done
7. report: survival wind speed per mode + diagrams (CLI acceptance test) — done
8. later: dynamics (modal / vortex shedding), fatigue, multi-element model

CLI: `python -m elementload --tube D,t,L [--tube ...] --overlap M --joint rigid|slipped
--support POS:KIND [--support ...] [--wind MPS] [--material MAT] [--pin-diameter M] [--n-pins N]`
or the installed `elementload` entry point. Report gives per-mode survival wind speeds
(yield, local buckling, joint pin shear) in km/h plus ASCII deflection/moment diagrams.
Open items flagged: local-buckling constant (0.3) and joint overlap stiffness models are
not yet calibrated against measured telescoping data; geometry not cross-checked against
a tube catalog. Survival wind speed bisection caps at 500 m/s.

## Dynamic loading plan (roadmap item 8)

Physical targets (all testable against closed-form or an independent solver):

### 8.1 Modal analysis — natural frequencies and mode shapes
- Add consistent mass matrix to `solver.py` (2-node Euler-Bernoulli, 4x4, mass per
  length rho*A from `tube_mass_per_length`). In overlap segments both tubes are
  physically present, so use rho*(A_prev + A_cur) regardless of joint stiffness model
  (the rigid/slipped choice only affects EI, not mass).
- Assemble global M, apply support boundary conditions (constrained DOFs removed),
  solve generalized eigenproblem (K - w^2 M) phi = 0 via `scipy.linalg.eigh`.
- Output natural frequencies f_i (Hz) and mode shapes; report first N modes.
- Golden tests: uniform cantilever — f_1 = (1.8751^2/(2 pi)) sqrt(EI/(m L^4)), and the
  (beta_n L)^2 pattern (1.8751, 4.6941, 7.8548); double cantilever uses half-span.
  Cross-check telescoping element frequencies against OpenSees eigen (run under
  `.venv-osees`).
- Damping ratio zeta: configurable input (CLI `--damping`, default ~0.01 for aluminum).
  — done (mass in `dynamics.py`, per-mode f + lock-in V_crit + Scruton in CLI report)

### 8.2 Vortex shedding / cross-wind response
- Strouhal number St = 0.18 (EN 1991-1-4, smooth circular cylinder).
- Shedding frequency f_s = St * V / D; lock-in when f_s matches a natural frequency.
- Critical wind speed per mode: V_crit,i = f_i * D / St.
- Scruton number Sc = 4*pi*m*zeta/(rho_air*D^2) — small Sc => strong lock-in response.
- Crosswind response per EN 1991-1-4 Annex E, approach 1:
  - E.7 y_F,max/b = (1/St^2)(1/Sc) c_lat K_w K; c_lat from Table E.3 at the lock-in
    ratio V_crit/V_m = 1 (c_lat = 0.75 c_lat,0), c_lat,0 from Fig. E.2 digitized
    (0.7 subcritical, 0.35 transcritical, log-Re transition 2e5..1e6).
  - K_w from E.8 for a cantilever (correlation length L_j = 6 b at the tip, capped
    at 0.6); K = 0.13 from Table E.5 (cantilever fundamental).
  - E.6-equivalent static load F(s) = m_e(s) (2 pi f_i)^2 phi_i(s) y_F,max gives the
    cyclic bending stress; the reported range is 2x the peak bending stress.
  - Validated: uniform-cantilever base moment M = m w^2 y_max h^2/4; equivalent mass
    m_e (F.14) against closed forms; modal shapes normalized to unit max displacement.
  - Report: per-mode y_max and bending stress range, lock-in window (0.83..1.25 of
    V_crit, Table E.3), and a NOTE when the design wind falls inside a lock-in window.
  — done (f_s, V_crit, Scruton, crosswind amplitude, stress range, lock-in window, --damping)

### 8.3 Along-wind dynamic amplification
- Dynamic amplification factor for buffeting response (gust turbulence) —
  EN 1991-1-4 structural factor c_s c_d or an equivalent DAF based on the fundamental
  frequency, mass, and damping. Applies as a multiplier on the static wind load or as
  a report note when fundamental f is below ~1 Hz.

### 8.4 Fatigue (follow-on, depends on 8.2/8.3)
- S-N curve data for 6061-T6 / 6063-T6, cyclic stress range from VIV and gust
  oscillations, cumulative damage (Miner's rule), fatigue life estimate in hours.

### 8.5 Multi-element model (follow-on)
- Load multiple telescoping elements (e.g., a Yagi boom with several elements) as a
  combined FE model; reuse per-element geometry/failure checks.

Design notes:
- Keep dynamic code decoupled: new `elementload/dynamics.py` (mass matrix, eigen,
  shedding, Scruton) and `elementload/fatigue.py`; report/CLI additions thin.
- Mass per length already available (`geometry.tube_mass_per_length`).
- Solver interface unchanged; modal assembly reuses existing node/element/support types.
