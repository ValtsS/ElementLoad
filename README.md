# ElementLoad

FEM-based load analysis of Yagi antenna elements (telescoping aluminium tubes).

The tool computes the structural response of an antenna element under wind (per
EN 1991-1-4), self-weight, ice and point loads, and reports:

- bending moment, shear, stress and deflection along the element,
- the **maximum wind speed** the element survives for each failure mode
  (yield, local wall buckling, joint pin shear),
- natural frequencies, vortex-shedding (cross-wind) response and the
  along-wind dynamic factor,
- fatigue life (Miner's rule) from VIV and gust buffeting.

It uses a 2-node Euler-Bernoulli beam FEM with stepped sections (own EI per
segment) and supports the telescoping-joint physics of real Yagi elements:
rigid-overlap and slipped/pinned-overlap bounds.

## Installation

Requires Python 3.11+. Install in editable mode from the repository root:

```
pip install -e .
```

The `elementload` command and the `python -m elementload` module entry point
both work. The solver depends on NumPy and SciPy.

### Windows without activating the venv

Two convenience wrappers live at the repository root and pick up the project
venv automatically (no `Activate.ps1` needed):

- `elementload.cmd` — from `cmd.exe`: `elementload.cmd --stm data/a1.stm`
- `elementload.ps1` — from PowerShell: `.\elementload.ps1 --stm data/a1.stm`

Both resolve `.venv\Scripts\python.exe` and add the repository root to
`PYTHONPATH`, so they work from any directory. To call `elementload` by short
name from anywhere, add the repository root to your `PATH`.

## Quick start

### Analyze a simple telescoping element (command line)

```
elementload.cmd \
  --tube 0.030,0.002,0.6 \
  --tube 0.025,0.002,0.6 \
  --tube 0.020,0.002,0.6 \
  --tube 0.016,0.0015,0.6 \
  --overlap 0.05 --joint rigid \
  --support center:fixed \
  --wind 40
```

(The `elementload.cmd` wrapper runs the venv python for you; `python -m
elementload` works identically.)

Each `--tube D,t,L` adds a tube: outer diameter (m), wall thickness (m),
length (m). `--overlap` is the telescoping overlap length between consecutive
tubes, `--joint` selects the overlap stiffness model, and `--support POS:KIND`
places supports (see below).

### Analyze a YagiMechanical .stm antenna design file

DXEngineering YagiMechanical (.stm, Format Version 2.0) files are supported
directly. The boom block is identified and skipped; each element is analyzed
individually:

```
elementload.cmd --stm data/a1.stm --element "40m RE"
```

Without `--element`, every element in the file is analyzed. The file's design
wind speed, reference height, tube geometry and (known) material are used
automatically. The `ES` (Element Support Distance) field adds support points
on each side of the centre mount — see "ES support model" below.

## Understanding the report

The report is split into blocks:

| Block | Meaning |
|-------|---------|
| **Tubes / Overlap / Supports / Material / Pins** | model definition |
| **Analysis at wind speed** | max moment, stress, deflection, self-weight sag at the chosen wind speed |
| **Survival wind speed per failure mode** | the wind speed at which each mode reaches its limit; the minimum is the LIMITING MODE |
| **Dynamics** | natural frequencies (Hz), per-mode lock-in wind speed (V_crit), effective mass, Scruton number |
| **Along-wind dynamic factor** | EN 1991-1-4 6.3.1/Annex B: c_s*c_d with components B^2, R^2, k_p, I_v |
| **Fatigue** | Miner-rule life from VIV lock-in and gust buffeting stress ranges |
| **Vortex shedding crosswind response** | EN 1991-1-4 Annex E: y_max, bending stress range, lock-in window per mode |
| **ASCII diagrams** | deflection and bending moment along the element |

Survival wind speeds are reported in **m/s** (the tool uses SI units
throughout).

## Graphical output

`--plot QUANTITY` writes a matplotlib figure of the element colored by a
response quantity along its length, with a colorbar:

```
elementload.cmd --stm data/a1.stm --element "40m RE" --plot stress
elementload.cmd --tube 0.030,0.002,0.6 --tube 0.025,0.002,0.6 \
  --tube 0.020,0.002,0.6 --tube 0.016,0.0015,0.6 --overlap 0.05 \
  --joint rigid --support center:fixed --wind 40 --plot moment --plot-file element.png
```

Available quantities: `stress` (the default when `--plot` is used), `moment`,
`load`, `wind_load`, `pressure`, `deflection`.

The figure draws the telescoping tube geometry as a horizontal band — the
diameters are exaggerated and labelled *not to scale*, since a true 1:1
drawing would be invisible for a thin, long element — with a colored
centerline gradient, marked supports, a colorbar and a dashed deformed-shape
overlay (magnification factor stated on the plot). The plot is evaluated at
the analysis wind speed (shown in the title). Colorbar units match the report:
stress in MPa, deflection in mm.

For an element with an ES support, the default `--es-model both` writes a
two-panel figure showing the **rigid-pin and sag-rope models side by side** on
a shared color scale, so the two support assumptions can be compared visually.
`--plot-file PATH` chooses the output file (default
`<element>_<quantity>.png`). Requires matplotlib (an optional dependency,
declared in pyproject.toml).

## Element optimization

`--optimize` (with `--stm` and exactly one `--element`) re-sizes the element's
tubes from a tube stock catalog so that it just survives a target wind speed,
and prints a stock-vs-optimized comparison (mass, saving, tube chain diff,
yield survival):

```
elementload.cmd --stm data/a1.stm --element "20m RE" --optimize --wind 36.6
```

The optimizer is a deterministic envelope sizer: it computes the bending
moment each stock tube position must carry at the target wind, then picks the
lightest catalog tube per position that (a) survives that moment (yield and
local buckling) and (b) can telescope — each tube's bore must host the tube
outside it plus a clearance. Because drag grows with diameter, the envelope is
re-solved to a fixed point.

The fixed point is not unique in general (the envelope map is monotone on the
feasible telescoping lattice but not on all chains), so a single starting chain
can trap the optimizer on a heavier-but-still-consistent solution. The optimizer
therefore runs the iteration from several diverse starts — the given chain, the
lightest telescoping chain, and the heaviest catalog chain — and returns the
lightest feasible result.

The telescoping fit is what actually constrains the chain: the root diameter is
forced by the tip (you cannot skip sizes), not by strength alone. For the 20 m
RE this means the 45 mm root is real — the only improvement over stock is a
20x2.0 -> 20x1.5 swap (2.4% mass), because stock's 20x2.0 bore (16 mm) is a
zero-clearance fit that cannot telescope over the 16 mm tube.

Options:

| Option | Description |
|--------|-------------|
| `--catalog FILE` | tube stock catalog CSV (`D_mm, t_mm` rows; default `data/tube_catalog.csv`) |
| `--clearance M` | telescoping clearance between tubes (default 0.001 = 1 mm) |
| `--length-step M` | snap per-side tube lengths to this increment, e.g. `0.005` for 5 mm (default 0 = off) |
| `--es-sweep` | sweep the ES support position, re-optimizing the tubes at each point |
| `--es-range START END` | ES position range for the sweep (m from centre; default 0.25x..0.95x half-length) |
| `--es-steps N` | sweep points (default 15) |
| `--layout-search` | brute-force the layout design space (tube count, taper, sleeve; overlaps derived from diameter) and print the lightest element per survival wind speed |
| `--layout-workers N` | worker processes for `--layout-search` (default: all CPU cores) |
| `--layout-n NMIN NMAX` | tube-count range for `--layout-search` (default 3 10) |

`--wind MPS` is the target survival wind speed (default: the .stm design WS).

### ES support position sweep

The ES support is usually the one thing a builder can actually move (the truss
attachment point). The `--es-sweep` mode re-runs the whole catalog optimization
at each ES position and prints mass, root tube and yield survival vs. ES:

```
elementload.cmd --stm data/a1.stm --element "40m RE" --optimize --es-sweep --wind 35
```

The ES pin moves the moment peak away from the centre mount (for the 40 m RE at
35 m/s it cuts the centre moment from 1860 to ~140 N·m), so the section can thin
toward the centre. But the telescoping bore chain can mask the benefit: the root
diameter is locked by the tip chain, so a range of ES positions give the *same*
mass and only differ in survival. Positions too close to centre leave the outer
cantilever unsupported (infeasible); positions near the tip put a huge moment at
the pin and force a heavier chain.

### Layout design-space search

`--layout-search` brute-forces the layout itself — how many tubes, the exposed-
length taper, and an optional short sleeve collar at the root — sizes each layout
from the catalog at the design wind, and prints the lightest element per survival
wind speed, marking the lightest that meets the design wind. Joint overlaps are
derived from the sized tube diameters rather than brute-forced: each joint gets
max(100 mm, 3 x tube diameter) using the outer/smaller tube, and the tip joint
keeps at least 125 mm so the clamp has tuning room. The grid is spread over
`--layout-workers` processes (default: all cores):

```
elementload.cmd --stm data/a1.stm --element "20m RE" --optimize --layout-search
```

For the 20 m RE at the 36.6 m/s design wind the search finds a 4053.9 g element
(`[40x2;35x2;30x2;25x2;20x1.5;12x1]`: a 150 mm 40 mm collar at the root, the 16
mm size skipped, 12 mm at the tip, joint overlaps 105/100/100/100/125 mm) vs
4870.6 g stock and 4751.7 g for the fixed-geometry optimizer. Because a telescope
must host every tube inside its bore, the root diameter is locked by the tip
chain — strength alone never fixes it, which is why the layout (not just the
section sizes) is the real design lever.

## Failure modes

- **yield** — peak bending stress reaches the material yield stress
  (`max stress` is shown; the yield stress comes from the material table, or
  from the .stm file's YS field for unknown materials).
- **buckling** — local wall buckling of the thin tubes:
  `sigma_cr = 0.3 E t / r_mean`.
- **pin_shear** — double-shear failure of the joint pins (needs
  `--pin-diameter` / `--n-pins`; the default 2 x 4 mm pins).

## Supports

Supports are given as `POS:KIND` and may be repeated. `POS` is the position in
metres from the element centre (or the literal `center`); `KIND` is one of:

- `fixed` — clamped (translation and rotation restrained),
- `pin` — pinned (translation restrained, rotation free),
- `roller` — roller (same as `pin` in this 2-D transverse model),
- `sag` — a rope/thread support that carries **self-weight only** (no wind
  restraint; used for the ES model).

For a telescoping element mounted at its centre you usually want at least
`center:fixed`. Examples:

```
--support center:fixed
--support center:fixed --support 2.5:pin --support -2.5:pin
```

## ES support model

The `.stm` `ES` field (Element Support Distance) is the distance from the boom
to where guys/rope attach. In the YM6 UI this is a vertical pipe on the boom
plus a dielectric rope to each element side. The rope is nearly horizontal, so
how much it restrains the element depends on the construction:

- **rigid pin** — the support restrains the wind load like a rigid truss
  (only correct if the element is genuinely trussed, e.g. multiple wires to
  separate anchors — a "V" of two wires to two boom pipes does restrain the
  wind in the V plane).
- **sag rope** — the rope restrains self-weight only (correct for a single
  near-horizontal rope; at ES = 5.5 m with a ~1 m pipe the angle is ~10 deg,
  so wind restraint is negligible).
- **both** (default) — computes both and prints them side by side so you see
  the range.

`--es-model rigid|sag|both` selects the model. Explicit `--support` overrides
the file-derived supports.

For the 40 m RE element of `data/a1.stm` the yield survival speed is
21.1 m/s (centre only) < 25.5 m/s (sag rope) < 48.2 m/s (rigid pin).

## Materials

`--material` selects the material table (yield / shear stress):

| Key | Yield (MPa) | Shear (MPa) |
|-----|-------------|-------------|
| `al6061-t6` | 276 | 207 |
| `al6063-t6` | 170 | 128 |
| `al6060-t6` | 160 | 120 |
| `al6060-t66` | 190 | 142.5 |

Fatigue S-N curves exist for all four materials (R = -1, stress amplitude):
6061-T6 (endurance 96.5 MPa), 6063-T6 (69), 6060-T6 (70), 6060-T66 (67).
For `.stm` files the MAT label is mapped to the matching key when known;
unknown labels fall back to the file YS.

## Reference examples

- `data/mechanical2.stm` — 7-element Yagi, design wind 40.5 m/s, centre-fixed.
  Yield is always the limiting mode. With `al6060-t6` 4/7 elements fail the
  design wind; with `al6060-t66` all pass (41.5..48.0 m/s).
- `data/a1.stm` / `data/a.stm` — 40 m / 20 m elements with ES supports,
  design wind 36.6 m/s. All pass; 40 m RE yield at 48.2 m/s (rigid pin).

## CLI reference

```
usage: elementload [-h] [--stm FILE] [--element NAME] [--optimize]
                   [--catalog FILE] [--clearance M] [--length-step M]
                   [--es-sweep] [--es-range START END] [--es-steps N]
                   [--layout-search] [--layout-workers N]
                   [--tube D,t,L] [--overlap M] [--joint {rigid,slipped}]
                   [--support POS:KIND] [--wind MPS]
                   [--material {al6061-t6,al6063-t6,al6060-t6,al6060-t66}]
                   [--pin-diameter M] [--n-pins N] [--damping ZETA]
                   [--height M] [--z0 M] [--es-model {rigid,sag,both}]
                   [--plot QUANTITY] [--plot-file PATH]
```

| Option | Description |
|--------|-------------|
| `--stm FILE` | analyze each element in a YagiMechanical .stm file |
| `--element NAME` | with `--stm`: analyze only this element |
| `--optimize` | with `--stm` + `--element`: re-size tubes from the catalog for the target wind |
| `--catalog FILE` | tube stock catalog for `--optimize` (default `data/tube_catalog.csv`) |
| `--clearance M` | telescoping clearance (default 0.001 m) |
| `--length-step M` | snap per-side tube lengths to this increment (default 0 = off) |
| `--es-sweep` | with `--optimize`: sweep the ES support position, re-optimizing at each point |
| `--es-range START END` | ES range for the sweep (m; default 0.25x..0.95x half-length) |
| `--es-steps N` | ES sweep points (default 15) |
| `--layout-search` | with `--optimize`: brute-force the layout design space, print lightest per survival |
| `--layout-workers N` | worker processes for `--layout-search` (default: all CPU cores) |
| `--tube D,t,L` | tube outer diameter, wall, length (m); repeatable |
| `--overlap M` | telescoping overlap length (m) |
| `--joint rigid|slipped` | overlap stiffness model |
| `--support POS:KIND` | support specification; repeatable |
| `--wind MPS` | analysis wind speed (m/s; default 30, or the .stm design WS) |
| `--material MAT` | material key (default al6061-t6, or the .stm MAT) |
| `--pin-diameter M` | joint pin diameter (m; default 0.004) |
| `--n-pins N` | pins per joint (default 2) |
| `--damping ZETA` | structural damping ratio (default 0.01) |
| `--height M` | reference height above ground (default 10, or the .stm H) |
| `--z0 M` | terrain roughness length (default 0.05, terrain category II) |
| `--es-model rigid|sag|both` | ES support model (default `both`) |
| `--plot QUANTITY` | write a matplotlib figure colored by stress, moment, load, wind_load, pressure or deflection |
| `--plot-file PATH` | output path for `--plot` (default `<element>_<quantity>.png`) |

## Physics background

- **Beam FEM**: 2-node Euler-Bernoulli elements, stepped sections with their
  own EI. Cross-checked against OpenSees for a stepped telescoping cantilever
  (deflections and max end moment agree to < 1e-6 relative).
- **Wind**: drag per unit length `q = 0.5*rho*V^2*Cd*D` with a Re-dependent
  Cd for smooth circular cylinders per EN 1991-1-4 (subcritical 1.2, drag
  crisis ~0.4, transcritical recovery), plus gust/height/terrain factors.
- **Dynamics**: consistent mass matrix; Strouhal 0.18 for vortex shedding;
  cross-wind response per EN 1991-1-4 Annex E; along-wind c_s*c_d per
  6.3.1/Annex B; fatigue per Miner's rule.
- **Telescoping joints**: rigid-overlap and slipped/pinned-overlap bounds.

## Development

Run the tests:

```
.venv\Scripts\python.exe -m pytest            # Windows
.venv/bin/python -m pytest                    # non-Windows
```

The OpenSees cross-check (`.venv-osees`, Python 3.12) runs separately and
skips automatically under other interpreters.
