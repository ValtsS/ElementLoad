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
--support POS:KIND [--support ...] [--wind MPS] [--material MAT] [--pin-diameter M] [--n-pins N]
[--damping ZETA] [--height M] [--z0 M]`
or, for a YagiMechanical antenna design file:
`python -m elementload --stm FILE [--element NAME] [--material MAT] [--joint ...] [--support ...]
[--wind MPS] [--damping ZETA] [--height M] [--z0 M]`
or the installed `elementload` entry point. Report gives per-mode survival wind speeds
(yield, local buckling, joint pin shear) in km/h plus ASCII deflection/moment diagrams,
and EN 1991-1-4 Annex E crosswind response and 6.3.1/Annex B along-wind dynamic factor
sections.

## .stm reader (`elementload/stm.py`)

DXEngineering YagiMechanical .stm files (Format Version 2.0) store each telescoping
element as a symmetric double cantilever: per tube segment, IS/IE are the inner/outer
END coordinates from the element centre on one side, X/T are outer diameter / wall
(inches), YS is the yield stress (psi) and MAT a free-form label. `parse_stm`/`read_stm`
convert to SI; the `B*` block is the boom and is identified and skipped. Per element,
`element_tube_extents` yields exact per-side extents; `assembly.build_from_extents`
builds the mesh with overlap intervals carrying the combined inertia of every tube
present (handles e.g. the triple-nested 20+16+12 sleeve in the 15m RE). The CLI runs
the full per-element analysis/survival/dynamics report. Material: the file MAT label is
mapped to a known key when possible (6060-T6 -> al6060-t6); the real build uses
al6060-t66 (`--material al6060-t66`, 190 MPa yield). Unknown/custom labels fall back to
the file YS (21756 psi = 150 MPa). Fatigue is computed for al6061-t6/al6063-t6/
al6060-t6/al6060-t66 (S-N curves in `fatigue.py` SN_DATA); unknown labels skip fatigue
with a NOTE in the report.
The `ES` field on an element line is the Element Support Distance (distance from the boom
to where guys/trussing attach; 0 = none). `StmElement.support_distance` holds it in metres;
the CLI adds a support at +ES and -ES from centre on top of the default centre-fixed
mount (e.g. a1.stm 40m elements: ES = 216.53543 in = 5.5 m -> supports center:fixed,
5.5:pin, -5.5:pin). Explicit `--support` overrides the file-derived supports.
Format facts confirmed against the YM6 UI: the ES support is a vertical pipe + dielectric
rope/thread to both element sides (sag support; rotation is free, so it behaves like a
pin only for loads in the sag plane). Two ES models are available via `--es-model`
(default `rigid`): `pin` restrains the wind load like a rigid truss; `sag` restrains
self-weight only (the rope is nearly horizontal — at ES = 5.5 m with a ~1 m pipe the
angle is ~10 deg, so it gives negligible wind restraint) and drops the ES supports for
all wind-driven results (static wind stress, natural frequencies, VIV, along-wind
dynamic factor, fatigue). `sag` is implemented in report.py as `_analyze_sag`: wind
and self-weight solves use different support sets and are combined orthogonally
(`math.hypot`), since wind acts in one plane and the sag rope in the perpendicular
plane. Survival wind speed ordering: center-only < sag < rigid pin (a1.stm 40m RE
yield: 76.0 / 91.7 / 173.5 km/h).
a1.stm's boom is two square tubes (80x80x4 + 70x70x3) and is skipped; header ice IT = 0.
YM6 custom material yield values (T6 150 MPa, T66 170 MPa, matching the file YS) are NOT
used for known materials: the CLI keeps the material-table allowables (T6 160 MPa, T66
190 MPa) for .stm analysis; file YS is only the fallback for unknown/custom MAT labels.
Open items flagged: local-buckling constant (0.3) and joint overlap stiffness models are
not yet calibrated against measured telescoping data; geometry not cross-checked against
a tube catalog. Survival wind speed bisection caps at 500 m/s.

Reference results (`data/mechanical2.stm`, design wind 145.9 km/h, center-fixed, rigid
joints): yield is ALWAYS the limiting mode (buckling ~700+ km/h, pin shear ~1800 km/h).
With al6060-t6 (160 MPa) 4/7 elements FAIL the design wind (15m RE 136.8, 15m DE 143.7,
10m RE 136.4, 10m D1 142.7 km/h); with al6060-t66 (190 MPa) all pass (149.5..172.9 km/h).
Fatigue is a non-issue for this antenna: gust buffeting stress range ~4-7 MPa and VIV
~0.1-0.2 MPa vs 67-70 MPa endurance -> unlimited life. Useful as a sanity-check fixture
when changing the solver or material data.

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
  EN 1991-1-4 structural factor c_s c_d (6.3.1 eq 6.1, Annex B procedure 1):
  - c_s c_d = (1 + 2 k_p I_v sqrt(B^2 + R^2))/(1 + 7 I_v); c_s = (1 + 7 I_v B^2)/(1 + 7 I_v).
  - Components per Annex B: I_v = k_I/(c_0 ln(z/z0)) (4.7); L(z) = 300 (z/200)^0.67 (B.1);
    B^2 = 1/(1 + 0.9 ((b+h)/L)^0.63) (B.2); R^2 = (pi^2/2 delta) S_L R_h R_b (B.2);
    S_L = 6.8 f_L/(1 + 10.2 f_L)^(5/3); R(eta) = 1/eta - (1-e^-2eta)/(2 eta^2),
    eta = 4.6 (b or h)/L f_L; k_p = max(sqrt(2 ln(nu T)) + 0.6/sqrt(2 ln(nu T)), 3), T = 600 s.
  - Damping delta = delta_s + delta_a; delta_s = 2 pi zeta (F.5),
    delta_a = c_f rho b v_m/(2 n_1 m_e) (F.18).
  - Validated: peak factor, admittance, background and spectrum against closed forms;
    c_s c_d assembly against eq 6.1/6.2; no-turbulence and high-frequency limits.
  - CLI `--height` (reference height, default 10 m) and `--z0` (terrain roughness,
    default 0.05 m, terrain category II). Report shows c_s, c_d, B^2, R^2, k_p, I_v and
    the decrements; a NOTE is printed when the fundamental frequency is below 1 Hz.
  — done (c_s c_d + components in `dynamics.py`, `along_wind_dynamic_factor` in report.py,
    CLI output and --height/--z0). Not yet applied as a multiplier on the static wind load
    (survival wind speeds still use the undamped static wind).

### 8.4 Fatigue (follow-on, depends on 8.2/8.3)
- S-N curve data for 6061-T6 / 6063-T6 / 6060-T6 / 6060-T66, cyclic stress range
  from VIV and gust oscillations, cumulative damage (Miner's rule), fatigue life
  estimate in hours.
  - S-N (R = -1, stress AMPLITUDE): closed-form fit to 6061-T6 data of
    Yahr (1997, ASME JPVT) S_a = A/sqrt(N) + S_e with A = 14 479 MPa,
    S_e = 96.5 MPa (14 ksi); treated as infinite life below S_e. 6063-T6 anchored
    to the ASM fatigue strength 69 MPa, 6060-T6 to the MakeItFrom fatigue strength
    70 MPa, 6060-T66 to 67 MPa; A scaled by endurance/96.5 for each (approximation).
  - `elementload/fatigue.py`: `stress_amplitude_at`, `cycles_to_failure`,
    `stress_range_to_failure` (amplitude = range/2), `damage_fraction`,
    `damage_per_hour`, `miner_life_hours`; validated against the published curve
    points (241.29 MPa @ 1e4, 110.979 MPa @ 1e6) and life-scaling identities.
  - Report `fatigue_report` combines two continuous cyclic sources: VIV lock-in
    of the FUNDAMENTAL mode (stress range from 8.2) and gust buffeting resonance
    (range = 2 * sigma_mean * 2 k_p I_v (sqrt(B^2+R^2) - sqrt(B^2))/(1+7 I_v),
    sigma_mean = wind-only max bending stress at the design wind, from 8.3).
    Cycles/s = the source frequency; D = sum of n/N per hour; life = 1/D (hours).
    VIV is limited to the fundamental mode because the E.7 factors K (Table E.5)
    and K_w ((s/h)^2 shape) are only defined for the fundamental; applying them to
    higher modes overestimates the stress (open item).
  — done (fatigue.py + fatigue_report + CLI Fatigue block)
  Open items: no fatigue concentration factor (Kt); welded-joint/mean-stress
  knock-downs (Yahr factor of 2/20) not applied; 6063-T6 curve is a scaled estimate.
  Note: 6060-T66 (67 MPa) is LOWER than 6060-T6 (70 MPa) per MakeItFrom despite the
  higher UTS — do NOT assume T66 scales fatigue from T6 by UTS ratio.
  Potential inconsistency: failure.py allowables (al6060-t6 160 MPa, al6060-t66 190 MPa)
  exceed EN 755-2 minimums (150/160 MPa yield, 190/215 MPa UTS); the .stm file YS
  (21756 psi = 150 MPa) matches EN 755-2 better. Decide whether to use file YS for
  .stm analysis instead of the material-name table.

### 8.5 Multi-element model (follow-on)
- Load multiple telescoping elements (e.g., a Yagi boom with several elements) as a
  combined FE model; reuse per-element geometry/failure checks.

Design notes:
- Keep dynamic code decoupled: new `elementload/dynamics.py` (mass matrix, eigen,
  shedding, Scruton) and `elementload/fatigue.py`; report/CLI additions thin.
- Mass per length already available (`geometry.tube_mass_per_length`).
- Solver interface unchanged; modal assembly reuses existing node/element/support types.
