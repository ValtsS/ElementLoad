import argparse
import math
import os

from elementload.report import build_model, analyze, survival_wind_speed, build_model_from_geometry
from elementload.stm import read_stm, element_tube_extents
from elementload.solver import internal_moment
from elementload.optimize import element_mass, read_catalog, size_with_tip_extension

_MODES = ("yield", "buckling", "pin_shear")


def _tube_spec(s):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected D,t,L got {s!r}")
    return tuple(parts)


def _extent_spec(s):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected D,t,start,end got {s!r}")
    return tuple(parts)


def _support_spec(s):
    pos, _, kind = s.partition(":")
    if kind not in ("fixed", "pin", "roller", "sag"):
        raise argparse.ArgumentTypeError(f"expected pos:kind got {s!r}")
    if pos != "center":
        try:
            pos = float(pos)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected numeric position or 'center', got {pos!r}")
    return (pos, kind)


def _tip_extension_spec(s):
    if s == "auto":
        return s
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a length in m, got {s!r}")
    if v < 0.0:
        raise argparse.ArgumentTypeError("tip extension must be >= 0")
    return v


def _ascii_chart(points, width=40, height=11):
    """points: list of (x, value); returns a list of strings, one per row."""
    if len(points) < 2:
        return ["(no data)"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for x, y in points:
        col = int(round((x - xmin) / xspan * (width - 1)))
        row = height - 1 - int(round((y - ymin) / yspan * (height - 1)))
        grid[row][col] = "#"
    return ["".join(r) for r in grid]


def _moment_samples(model, result, per_element=4):
    points = []
    for i, seg in enumerate(model.segments):
        L = seg.x1 - seg.x0
        for k in range(per_element + 1):
            s = L * k / per_element
            x = seg.x0 + s
            m = internal_moment(model.elements, i, model.nodes,
                                result.displacements, model.loads, s)
            points.append((x, abs(m)))
    return points


def _deflection_samples(model, result):
    return [(n.x, abs(w)) for n, (w, _) in zip(model.nodes, result.displacements)]


def render_report(model, result, wind_speed, damping=0.01, height=10.0, z_0=0.05, title=None):
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    else:
        lines.append("Element load analysis")
        lines.append("=====================")
    lines.append("")
    lines.append("Tubes (D x t, length):")
    for i, (d, t, L) in enumerate(model.tube_specs, start=1):
        lines.append(f"  {i}: {d * 1000:.1f} x {t * 1000:.2f} mm, {L * 1000:.0f} mm")
    if isinstance(model.overlap, (list, tuple)):
        ov = ", ".join(f"{o * 1000:.0f}" for o in model.overlap)
        lines.append(f"Overlap per joint: {ov} mm, joint model: {model.joint}")
    else:
        lines.append(f"Overlap: {model.overlap * 1000:.0f} mm, joint model: {model.joint}")
    supports = ", ".join(f"{p:g}:{k}" if isinstance(p, float) else f"{p}:{k}"
                         for p, k in model.support_specs)
    lines.append(f"Supports: {supports}")
    lines.append(f"Material: {model.material}")
    lines.append(f"Pins: {model.n_pins} x {model.pin_diameter * 1000:.1f} mm at each joint")
    lines.append("")
    lines.append(f"Analysis at wind speed {wind_speed:.1f} m/s:")
    lines.append(f"  max moment:   {result.max_moment:9.1f} N.m at x = {result.max_moment_location:+.2f} m")
    lines.append(f"  max stress:   {result.max_stress / 1e6:9.1f} MPa at x = {result.max_stress_location:+.2f} m")
    lines.append(f"  max deflection: {result.max_deflection * 1000:8.1f} mm")
    lines.append(f"  self-weight sag: {result.max_sag * 1000:8.1f} mm")
    lines.append("")
    lines.append("Survival wind speed per failure mode:")
    survivals = {}
    for mode in _MODES:
        v = survival_wind_speed(model, mode=mode)
        survivals[mode] = v
        lines.append(f"  {mode:>10}: {v:7.1f} m/s")
    limiting = min(survivals, key=survivals.get)
    lines.append(f"  LIMITING MODE: {limiting} at {survivals[limiting]:.1f} m/s")
    lines.append("")
    lines.append(f"Dynamics (damping zeta = {damping:.3f}):")
    from elementload.report import natural_frequencies, vortex_crosswind_response
    from elementload.dynamics import critical_wind_speed, scruton_number
    from elementload.report import _model_mass
    freqs = natural_frequencies(model, n_modes=3)
    for i, f in enumerate(freqs, start=1):
        v_crit = critical_wind_speed(f, model.tube_specs[0][0])
        lines.append(f"  mode {i}: f = {f:6.2f} Hz, lock-in at V_crit = {v_crit:6.1f} m/s")
    m_eff = _model_mass(model)
    d_ref = model.tube_specs[0][0]
    sc = scruton_number(m_eff, damping, d_ref)
    lines.append(f"  effective mass {m_eff:.3f} kg/m, Scruton Sc = {sc:.1f}")
    lines.append("")
    from elementload.report import along_wind_dynamic_factor
    daf = along_wind_dynamic_factor(model, damping=damping, height=height, z_0=z_0)
    lines.append(f"Along-wind dynamic factor (EN 1991-1-4 6.3.1/Annex B, reference height {height:.0f} m, "
                 f"z0 = {z_0:.2f} m):")
    lines.append(
        f"  c_s*c_d = {daf['c_s_cd']:.3f} (c_s = {daf['c_s']:.3f}, c_d = {daf['c_d']:.3f}, "
        f"B^2 = {daf['background']:.3f}, R^2 = {daf['resonance']:.3f}, "
        f"k_p = {daf['peak_factor']:.2f}, I_v = {daf['turbulence_intensity']:.3f})"
    )
    lines.append(
        f"  damping decrements: delta_s = {daf['delta_s']:.3f}, delta_a = {daf['delta_a']:.3f}, "
        f"delta = {daf['delta']:.3f}"
    )
    if daf["frequency"] < 1.0:
        lines.append(
            f"  NOTE: fundamental frequency {daf['frequency']:.2f} Hz < 1 Hz, "
            "along-wind dynamic amplification is significant"
        )
    lines.append("")
    from elementload.report import fatigue_report
    fat = fatigue_report(model, damping=damping, height=height, z_0=z_0)
    if fat.get("skipped"):
        lines.append(f"Fatigue (Miner's rule, R = -1, S-N {model.material}): "
                     "no S-N data for this material - skipped")
    else:
        lines.append(f"Fatigue (Miner's rule, R = -1, S-N {model.material}):")
        for s in fat["sources"]:
            if s["damage_per_hour"] > 0.0:
                life = 1.0 / s["damage_per_hour"]
                lines.append(
                    f"  {s['name']:>22}: range = {s['stress_range'] / 1e6:6.1f} MPa @ "
                    f"{s['frequency']:.2f} Hz -> life {life:.3g} h"
                )
            else:
                lines.append(
                    f"  {s['name']:>22}: range = {s['stress_range'] / 1e6:6.1f} MPa @ "
                    f"{s['frequency']:.2f} Hz -> no fatigue damage (below endurance)"
                )
        if fat["life_hours"] == float("inf"):
            lines.append("  combined life: unlimited (all sources below endurance)")
        else:
            lines.append(f"  combined Miner life: {fat['life_hours']:.3g} h")
    lines.append("")
    lines.append("Vortex shedding crosswind response (EN 1991-1-4 Annex E, approach 1):")
    v_resp = vortex_crosswind_response(model, damping=damping, n_modes=3)
    design_mps = wind_speed
    for i, r in enumerate(v_resp, start=1):
        lo_mps = r["lock_in"][0]
        hi_mps = r["lock_in"][1]
        lines.append(
            f"  mode {i}: y_max = {r['y_max'] * 1000:7.2f} mm, "
            f"bending stress range = {r['stress_range'] / 1e6:7.1f} MPa, "
            f"lock-in window {lo_mps:6.1f}..{hi_mps:6.1f} m/s"
        )
        if lo_mps <= design_mps <= hi_mps:
            lines.append(f"  NOTE: design wind {design_mps:.1f} m/s is inside the mode {i} lock-in window")
    lines.append("")
    lines.append("Deflection |w| along element (mm):")
    samples = _deflection_samples(model, result)
    chart = _ascii_chart([(x, y * 1000) for x, y in samples])
    lines.extend("  " + row for row in chart)
    lines.append("")
    lines.append("Bending moment |M| along element (N.m):")
    chart = _ascii_chart(_moment_samples(model, result))
    lines.extend("  " + row for row in chart)
    return "\n".join(lines)


def render_es_comparison(analyses, wind_speed):
    """Side-by-side summary of the rigid-pin and sag ES support models.

    `analyses` is a list of (label, model, result) with label "pin" or "sag".
    Shows max moment/stress/deflection and per-mode survival wind speeds for both."""
    names = {"pin": "rigid pin", "sag": "sag rope"}
    lines = []
    lines.append(f"ES support model comparison (analysis at {wind_speed:.1f} m/s):")
    labels = [names[l] for l, _, _ in analyses]
    lines.append(f"  {'':22}{labels[0]:>16}{labels[1]:>16}")
    for name, get in (("max moment (N.m)", lambda r: r.max_moment),
                      ("max stress (MPa)", lambda r: r.max_stress / 1e6),
                      ("max defl (mm)", lambda r: r.max_deflection * 1000)):
        row = f"  {name:<20}"
        for _, _, r in analyses:
            row += f"{get(r):16.1f}"
        lines.append(row)
    lines.append("  survival wind speed (m/s):")
    for mode in _MODES:
        row = f"    {mode:<16}"
        for _, model, _ in analyses:
            v = survival_wind_speed(model, mode=mode)
            row += f"{v:16.1f}"
        lines.append(row)
    row = f"    {'limiting':<16}"
    for _, model, _ in analyses:
        survivals = {m: survival_wind_speed(model, mode=m) for m in _MODES}
        limiting = min(survivals, key=survivals.get)
        row += f"{limiting:>16}"
    lines.append(row)
    return "\n".join(lines)


def _analyze_es_pair(el, wind, args, parser):
    """Build and analyze the rigid-pin and sag models for an .stm element with an ES
    support. Returns (labels, models, results) aligned lists, label "pin" or "sag"."""
    mat = args.material or el.material_key
    yield_stress = None if mat is not None else el.yield_stress
    labels, models, results = [], [], []
    for label in ("pin", "sag"):
        supports = [("center", "fixed")]
        if el.support_distance > 0.0:
            es_kind = "pin" if label == "pin" else "sag"
            supports += [(el.support_distance, es_kind), (-el.support_distance, es_kind)]
        model = build_model_from_geometry(
            element_tube_extents(el), args.joint or "rigid",
            supports, wind, mat or el.material,
            args.pin_diameter, args.n_pins, yield_stress)
        labels.append(label)
        models.append(model)
        results.append(analyze(model))
    return labels, models, results


def render_optimization(opt_tubes, stock_tubes, wind_speed, material, yield_stress,
                        joint, supports, pin_diameter, n_pins, clearance,
                        tip_extension=0.0, tip_extension_kind="fixed"):
    """Comparison of the stock element vs. the catalog-optimized tube set."""
    from elementload.optimize import element_mass
    from elementload.report import build_model_from_geometry
    stock_model = build_model_from_geometry(
        stock_tubes, joint, supports, wind_speed, material,
        pin_diameter, n_pins, yield_stress)
    opt_model = build_model_from_geometry(
        opt_tubes, joint, supports, wind_speed, material,
        pin_diameter, n_pins, yield_stress)
    lines = []
    lines.append(f"Optimization at target wind {wind_speed:.1f} m/s "
                 f"(catalog-constrained, clearance {clearance:g} m):")
    if tip_extension > 0.0:
        label = f"tip extension {tip_extension * 1000:.0f} mm/side ({tip_extension_kind})"
        lines.append(f"  {label}")
    elif tip_extension_kind.startswith("auto"):
        lines.append(f"  tip extension: {tip_extension_kind} (applied {tip_extension * 1000:.0f} mm/side)")
    lines.append(f"  {'':8}{'stock':>22}{'optimized':>22}")
    stock_mass = element_mass(stock_tubes)
    opt_mass = element_mass(opt_tubes)
    lines.append(f"  {'mass':<8}{stock_mass * 1000:22.1f}{opt_mass * 1000:22.1f} g")
    lines.append(f"  {'saving':<8}{'':22}{100 * (1 - opt_mass / stock_mass):21.1f} %")
    lines.append("  tube chain (centre -> tip):")
    for a, b in zip(stock_tubes, opt_tubes):
        mark = " " if (round(a.outer_d, 6), round(a.wall, 6)) == \
            (round(b.outer_d, 6), round(b.wall, 6)) else "*"
        lines.append(
            f"    {mark} {a.outer_d * 1000:5.1f} x {a.wall * 1000:3.2f} mm  ->  "
            f"{b.outer_d * 1000:5.1f} x {b.wall * 1000:3.2f} mm  "
            f"({a.inner_start * 1000:.0f}..{a.inner_end * 1000:.0f} mm)")
    lines.append("  survival wind speed (yield):")
    for label, model in (("stock", stock_model), ("optimized", opt_model)):
        v = survival_wind_speed(model, "yield")
        lines.append(f"    {label:<10}{v:22.1f} m/s")
    return "\n".join(lines)


def render_es_sweep(el, wind_speed, args, parser):
    """Sweep the ES support position, re-optimizing the tube chain at each point.
    Returns (lines, rows) for the sweep table. Each row: (es, mass, root_d, survival)
    or (es, None, None, None) when infeasible."""
    from elementload.optimize import element_mass
    mat = args.material or el.material_key
    yield_stress = None if mat is not None else el.yield_stress
    stock_tubes = element_tube_extents(el)
    half = max(max(t.inner_end for t in stock_tubes), abs(min(t.inner_start for t in stock_tubes)))
    if args.es_range is None:
        lo, hi = 0.25 * half, 0.95 * half
    else:
        lo, hi = sorted(args.es_range)
    n = max(args.es_steps, 2)
    catalog = read_catalog(args.catalog or os.path.join(
        os.path.dirname(__file__), "..", "data", "tube_catalog.csv"))
    lines = []
    lines.append(f"ES position sweep at target wind {wind_speed:.1f} m/s "
                 f"({n} points, {lo:.2f}..{hi:.2f} m from centre, rigid ES, "
                 f"tip extension {_tip_extension_label(args.tip_extension)}):")
    lines.append(f"  {'ES [m]':>8}  {'mass [kg]':>10}  {'root [mm]':>9}  "
                 f"{'survival [m/s]':>14}   chain (centre -> tip)")
    rows = []
    for es in [lo + (hi - lo) * k / (n - 1) for k in range(n)]:
        supports = [("center", "fixed"), (es, "pin"), (-es, "pin")]
        try:
            opt, _ = size_with_tip_extension(
                stock_tubes, stock_tubes, args.joint or "rigid", supports,
                wind_speed, mat or el.material, yield_stress,
                catalog=catalog, pin_diameter=args.pin_diameter,
                n_pins=args.n_pins, clearance=args.clearance,
                length_step=args.length_step,
                tip_extension=args.tip_extension)
            model = build_model_from_geometry(opt, args.joint or "rigid", supports,
                                              wind_speed, mat or el.material,
                                              yield_stress=yield_stress)
            v = survival_wind_speed(model, "yield")
            chain = " ".join(f"{t.outer_d * 1000:.0f}x{t.wall * 1000:g}" for t in opt)
            rows.append((es, element_mass(opt), opt[0].outer_d * 1000, v))
            lines.append(f"  {es:8.2f}  {rows[-1][1]:10.2f}  {rows[-1][2]:9.0f}  "
                         f"{v:14.1f}   {chain}")
        except ValueError:
            rows.append((es, None, None, None))
            lines.append(f"  {es:8.2f}  {'infeasible':>10}")
    return "\n".join(lines), rows


def render_layout_search(el, wind_speed, args, parser):
    """Brute-force the layout design space (N, overlap, taper, sleeve) and print
    the lightest catalog-sized element at each survival wind speed."""
    from elementload.optimize import search_layouts
    mat = args.material or el.material_key
    yield_stress = None if mat is not None else el.yield_stress
    stock_tubes = element_tube_extents(el)
    supports = args.support
    if supports is None:
        supports = [("center", "fixed")]
        if el.support_distance > 0.0:
            es_kind = "sag" if args.es_model == "sag" else "pin"
            supports += [(el.support_distance, es_kind), (-el.support_distance, es_kind)]
    catalog = read_catalog(args.catalog or os.path.join(
        os.path.dirname(__file__), "..", "data", "tube_catalog.csv"))
    results = search_layouts(
        stock_tubes, args.joint or "rigid", supports, wind_speed,
        mat or el.material, yield_stress, catalog=catalog,
        pin_diameter=args.pin_diameter, n_pins=args.n_pins,
        clearance=args.clearance, length_step=args.length_step,
        tip_extension=args.tip_extension, ref_tubes=stock_tubes,
        n_range=args.layout_n, workers=args.layout_workers)
    lines = []
    lines.append(f"Layout design-space search at target wind {wind_speed:.1f} m/s "
                 f"(N={args.layout_n[0]}..{args.layout_n[1]}, "
                 f"{len(results)} feasible of the grid, {args.layout_workers or 'all cores'} workers):")
    lines.append("")
    supports_note = ", ".join(f"{p:g}:{k}" if isinstance(p, float) else f"{p}:{k}"
                              for p, k in supports)
    if any(k == "sag" for _, k in supports):
        supports_note += " (sag: wind restrained by non-sag supports only)"
    lines.append(f"  supports: {supports_note}")
    tip_note = f"tip extension {_tip_extension_label(args.tip_extension)}"
    if results and any(r["tip_ext"] > 0.0 for r in results):
        exts = [r["tip_ext"] for r in results if r["tip_ext"] > 0.0]
        tip_note += f" (applied {min(exts) * 1000:.0f}..{max(exts) * 1000:.0f} mm/side)"
    lines.append(f"  {tip_note}")
    lines.append("")
    design_met = [r for r in results if r["survival"] >= wind_speed]
    lightest_ok = min(design_met, key=lambda r: r["mass"]) if design_met else None

    lines.append("  Pareto frontier (lightest non-dominated design at each survival):")
    lines.append(f"    {'survival':>8}  {'mass':>9}  {'N':>2}  {'ov':>4}  {'taper':>5}  "
                 f"{'sleeve':>6}   chain (centre -> tip)")
    for r in _pareto_frontier(results):
        mark = " <- lightest at/above design wind" if r is lightest_ok else ""
        lines.append(
            f"    {r['survival']:8.1f}  {r['mass'] * 1000:8.1f} g  {r['n']:2d}  "
            f"{r['overlap']:4.2f}  {r['ratio']:5.2f}  "
            f"{r['sleeve'] if r['sleeve'] is not None else '-':>6}   [{r['chain']}]{mark}")
    if lightest_ok is None:
        lines.append("    (no layout reaches the design wind speed)")
    lines.append("")
    lines.append("  Materialized Pareto designs (ready-to-paste elementload command):")
    lines.append("  Per-side tube extents D,t,start,end (m from centre -> tip, innermost")
    lines.append("  first); the command re-runs the full report + --plot on this layout:")
    for i, r in enumerate(_pareto_frontier(results), start=1):
        mark = " <- lightest at/above design wind" if r is lightest_ok else ""
        cmd = _materialize_command(r, args.joint or "rigid", supports, wind_speed,
                                   material=mat)
        lines.append(f"  [{i}] {r['survival']:5.1f} m/s  {r['mass'] * 1000:7.1f} g{mark}")
        lines.append(f"      {cmd}")
        lines.append(_materialize_table(r))
    lines.append("")
    lines.append("  Best design per root tube diameter (centre-side OD):")
    lines.append(f"    {'root':>5}  {'best survival':>13}  {'mass':>9}  "
                 f"{'meets wind?':>11}   chain (centre -> tip)")
    for root, r in sorted(_best_per_root(results, wind_speed).items()):
        meets = "yes" if r["survival"] >= wind_speed else "NO"
        lines.append(
            f"    {root:4.0f} mm  {r['survival']:11.1f} m/s  {r['mass'] * 1000:7.1f} g  "
            f"{meets:>11}   [{r['chain']}]")
    return "\n".join(lines)


def _tip_extension_label(tip_extension):
    if tip_extension == "auto":
        return "auto (compensates mean-diameter loss)"
    if tip_extension == 0:
        return "none"
    return "fixed"


def _materialize_command(r, joint, supports, wind_speed, material=None):
    """Elementload CLI invocation that reproduces a layout-search result's exact
    geometry and material, via per-side tube extents (same path the .stm reader
    uses). Extents are rounded to 1 mm (0.001 m) for readability and buildability."""
    parts = [f"--extent {d:g},{t:g},{a:.3f},{b:.3f}" for d, t, a, b in r["extents"]]
    cmd = ["elementload"] + parts
    if material is not None:
        cmd.append(f"--material {material}")
    cmd.append(f"--joint {joint}")
    cmd.extend(f"--support {p}:{k}" if isinstance(p, str)
               else f"--support {p:g}:{k}" for p, k in supports)
    cmd.append(f"--wind {wind_speed:g}")
    return " ".join(cmd)


def _materialize_table(r):
    """Compact per-tube table for a layout-search result: Dxwall, start, stop,
    exposed (per-side), and full length, all in metres."""
    lines = ["    tubes (centre -> tip):"]
    lines.append("    {:<12} {:>6} {:>6} {:>8} {:>8}".format(
        "Dxwall", "start", "stop", "exposed", "full"))
    for d, t, a, b in r["extents"]:
        exposed = b - a
        full = 2.0 * exposed
        lines.append("    {:<12} {:>6.3f} {:>6.3f} {:>8.3f} {:>8.3f}".format(
            f"{d*1000:.0f}x{t*1000:g}", a, b, exposed, full))
    return "\n".join(lines)


def _pareto_frontier(results, bucket=0.1):
    """Non-dominated layouts: keep a result unless some other result has both a
    lower (or equal) mass and a higher (or equal) survival, where survival is
    bucketed to `bucket` m/s (default 0.1, matching the display) so bisection
    floating-point noise does not create near-duplicate frontier points.
    Sorted by survival."""
    front = []
    for i, r in enumerate(results):
        s_r = round(r["survival"] / bucket)
        dominated = any(
            round(o["survival"] / bucket) >= s_r and o["mass"] <= r["mass"]
            and (round(o["survival"] / bucket) > s_r or o["mass"] < r["mass"])
            for j, o in enumerate(results) if j != i)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["survival"])


def _best_per_root(results, wind_speed, bucket=0.1):
    """For each root tube OD (mm, centre-side), the layout with the highest
    survival wind speed (ties broken by lightest mass). Survival is bucketed to
    `bucket` m/s so bisection floating-point noise does not pick a heavier
    layout at a nominally-equal wind speed."""
    best = {}
    for r in results:
        root = float(r["chain"].split(";")[0].split("x")[0])
        cur = best.get(root)
        s_r = round(r["survival"] / bucket)
        s_cur = round(cur["survival"] / bucket) if cur is not None else None
        if cur is None or (s_r > s_cur
                           or (s_r == s_cur and r["mass"] < cur["mass"])):
            best[root] = r
    return best


def _maybe_plot(args, model, result, label, wind_speed, pair=None):
    """Write a plot of `model`/`result` colored by args.plot. If `pair` is given
    (the sag model when `--es-model both`), draw both ES support models side by
    side. Returns the written path or None."""
    if args.plot is None:
        return None
    from elementload.plot import plot_element, plot_es_pair
    path = args.plot_file or f"{label}_{args.plot}.png"
    if pair is not None:
        plot_es_pair((model, result), pair, path, quantity=args.plot,
                     title=label, wind_speed=wind_speed)
    else:
        plot_element(model, result, path, quantity=args.plot, title=label,
                     wind_speed=wind_speed)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="elementload",
        description="FEM load analysis of a telescoping Yagi antenna element",
    )
    parser.add_argument("--stm", metavar="FILE",
                        help="YagiMechanical .stm file: analyze each telescoping element "
                             "(boom identified and skipped)")
    parser.add_argument("--element", metavar="NAME",
                        help="with --stm: only analyze the element with this name")
    parser.add_argument("--optimize", action="store_true",
                        help="with --stm and --element: re-size the element tubes from the "
                             "tube catalog so it just survives the target wind speed")
    parser.add_argument("--catalog", metavar="FILE", default=None,
                        help="tube stock catalog CSV (D_mm, t_mm rows); used by --optimize "
                             "(default data/tube_catalog.csv)")
    parser.add_argument("--clearance", type=float, default=1.0e-3, metavar="M",
                        help="telescoping clearance between tubes (default 1 mm)")
    parser.add_argument("--length-step", type=float, default=0.0, metavar="M",
                        help="snap per-side tube lengths to this increment (e.g. 0.005 "
                             "for 5 mm); 0 disables snapping")
    parser.add_argument("--tip-extension", default="auto", type=_tip_extension_spec,
                        metavar="AUTO|M",
                        help="extend the tip tube to compensate the resonance up-shift "
                             "from thinner tubes: 'auto' (default) scales by the element's "
                             "mean-diameter loss vs the stock element, a length in m is a "
                             "fixed per-side extension, 0 disables")
    parser.add_argument("--es-sweep", action="store_true",
                        help="with --optimize: sweep the ES support position, re-optimizing "
                             "the tubes at each position, and print mass vs ES")
    parser.add_argument("--layout-search", action="store_true",
                        help="with --optimize: brute-force the layout design space "
                             "(tube count, overlap, taper, sleeve collar) and print the "
                             "lightest catalog-sized element per survival wind speed")
    parser.add_argument("--layout-workers", type=int, default=None, metavar="N",
                        help="worker processes for --layout-search (default: os.cpu_count())")
    parser.add_argument("--layout-n", type=int, nargs=2, default=(3, 10),
                        metavar=("NMIN", "NMAX"),
                        help="tube-count range for --layout-search (default 3 10)")
    parser.add_argument("--es-range", type=float, nargs=2, default=None, metavar=("START", "END"),
                        help="ES position range for --es-sweep (m from centre; default "
                             "0.25 x to 0.95 x element half-length)")
    parser.add_argument("--es-steps", type=int, default=15, metavar="N",
                        help="number of sweep points for --es-sweep (default 15)")
    parser.add_argument("--tube", action="append", type=_tube_spec,
                        metavar="D,t,L", help="tube outer diameter, wall, length (SI); repeatable")
    parser.add_argument("--extent", action="append", type=_extent_spec,
                        metavar="D,t,START,END",
                        help="tube outer diameter, wall, per-side extents (m from centre); "
                             "repeatable, innermost first (same convention as .stm files)")
    parser.add_argument("--overlap", type=float, metavar="M",
                        help="telescoping overlap length (m)")
    parser.add_argument("--joint", choices=("rigid", "slipped"),
                        help="joint stiffness model")
    parser.add_argument("--support", action="append", type=_support_spec,
                        metavar="POS:KIND", help="support, e.g. center:fixed; repeatable")
    parser.add_argument("--wind", type=float, default=None, metavar="MPS",
                        help="analysis wind speed in m/s (default 30, or the .stm design WS)")
    parser.add_argument("--material", default=None,
                        choices=("al6061-t6", "al6063-t6", "al6060-t6", "al6060-t66"))
    parser.add_argument("--pin-diameter", type=float, default=0.004, metavar="M")
    parser.add_argument("--n-pins", type=int, default=2, metavar="N")
    parser.add_argument("--damping", type=float, default=0.01, metavar="ZETA",
                        help="structural damping ratio for dynamic response (default 0.01)")
    parser.add_argument("--height", type=float, default=None, metavar="M",
                        help="reference height above ground for turbulence (default 10, or the .stm H)")
    parser.add_argument("--z0", type=float, default=0.05, metavar="M",
                        help="terrain roughness length z0 (default 0.05, terrain category II)")
    parser.add_argument("--es-model", choices=("rigid", "sag", "both"), default="both",
                        help="how the .stm ES element support is modelled: rigid pin, "
                             "sag rope (restrains self-weight only, no wind restraint), "
                             "or both (default: report the two results side by side)")
    parser.add_argument("--plot", choices=("stress", "moment", "load", "wind_load",
                                           "pressure", "deflection"), default=None,
                        metavar="QUANTITY",
                        help="write a matplotlib figure of the element colored by the "
                             "given quantity (stress, moment, load, wind_load, pressure, "
                             "deflection)")
    parser.add_argument("--plot-file", default=None, metavar="PATH",
                        help="output path for --plot (default <element>_<quantity>.png)")
    args = parser.parse_args(argv)

    if args.stm is not None:
        if args.tube is not None or args.overlap is not None or args.extent is not None:
            parser.error("--tube/--extent/--overlap are not used with --stm "
                         "(geometry comes from the file)")
        return _main_stm(args, parser)

    if args.extent is not None:
        if args.tube is not None or args.overlap is not None:
            parser.error("--tube/--overlap are not used with --extent "
                         "(per-side extents already include the overlaps)")
        if args.joint is None or args.support is None:
            parser.error("--extent requires --joint and --support")
        return _main_extent(args, parser)

    if args.tube is None or args.overlap is None or args.joint is None or args.support is None:
        parser.error("either --stm or --tube/--overlap/--joint/--support or "
                     "--extent/--joint/--support must be given")

    wind = 30.0 if args.wind is None else args.wind
    height = 10.0 if args.height is None else args.height
    model = build_model(args.tube, args.overlap, args.joint, args.support,
                        wind, args.material or "al6061-t6",
                        args.pin_diameter, args.n_pins)
    result = analyze(model)
    print(render_report(model, result, wind, damping=args.damping,
                        height=height, z_0=args.z0))
    path = _maybe_plot(args, model, result, "element", wind)
    if path:
        print(f"Plot saved: {path}")


def _main_extent(args, parser):
    from elementload.assembly import TubeExtent
    from elementload.report import build_model_from_geometry
    wind = 30.0 if args.wind is None else args.wind
    height = 10.0 if args.height is None else args.height
    extents = [TubeExtent(*spec) for spec in args.extent]
    model = build_model_from_geometry(
        extents, args.joint, args.support, wind,
        args.material or "al6061-t6", args.pin_diameter, args.n_pins)
    result = analyze(model)
    print(render_report(model, result, wind, damping=args.damping,
                        height=height, z_0=args.z0))
    path = _maybe_plot(args, model, result, "element", wind)
    if path:
        print(f"Plot saved: {path}")


def _main_stm(args, parser):
    stm = read_stm(args.stm)
    print(f"YagiMechanical file: {args.stm}")
    print(f"Antenna: {stm.antenna}, height H = {stm.height:.2f} m, design wind "
          f"WS = {stm.wind_speed:.1f} m/s")
    if stm.boom is not None:
        b = stm.boom.segments[0]
        print(f"Boom ({stm.boom.key}): {b.outer_d * 1000:.1f} x {b.wall * 1000:.2f} mm, "
              f"length {b.inner_end * 1000:.0f} mm - identified, skipped (per-element analysis)")
    elements = stm.elements
    if args.element is not None:
        wanted = args.element.lower()
        matches = [e for e in stm.elements if e.name.lower() == wanted]
        if not matches:
            parser.error(f"no element named {args.element!r} in {args.stm}")
        elements = matches
    print(f"Elements: {', '.join(e.name for e in elements)}")
    wind = args.wind if args.wind is not None else stm.wind_speed
    height = args.height if args.height is not None else stm.height
    if args.optimize and len(elements) != 1:
        parser.error("--optimize requires exactly one element (use --element NAME)")
    if args.optimize:
        el = elements[0]
        mat = args.material or el.material_key
        yield_stress = None if mat is not None else el.yield_stress
        if args.es_sweep:
            print(render_es_sweep(el, wind, args, parser)[0])
            return
        if args.layout_search:
            print(render_layout_search(el, wind, args, parser))
            return
        supports = args.support
        if supports is None:
            supports = [("center", "fixed")]
            if el.support_distance > 0.0:
                es_kind = "sag" if args.es_model == "sag" else "pin"
                supports += [(el.support_distance, es_kind), (-el.support_distance, es_kind)]
        catalog_path = args.catalog
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "tube_catalog.csv")
        catalog = read_catalog(catalog_path)
        stock_tubes = element_tube_extents(el)
        opt_tubes, applied_ext = size_with_tip_extension(
            stock_tubes, stock_tubes, args.joint or "rigid", supports, wind, mat or el.material,
            yield_stress, catalog=catalog, pin_diameter=args.pin_diameter,
            n_pins=args.n_pins, clearance=args.clearance,
            length_step=args.length_step, tip_extension=args.tip_extension)
        print(render_optimization(
            opt_tubes, stock_tubes, wind, mat or el.material, yield_stress,
            args.joint or "rigid", supports, args.pin_diameter, args.n_pins,
            args.clearance, tip_extension=applied_ext,
            tip_extension_kind=_tip_extension_label(args.tip_extension)))
        return
    for el in elements:
        mat = args.material or el.material_key
        yield_stress = None if mat is not None else el.yield_stress
        supports = args.support
        if supports is None:
            supports = [("center", "fixed")]
            if el.support_distance > 0.0:
                es_kind = "pin" if args.es_model == "rigid" else "sag"
                supports += [(el.support_distance, es_kind), (-el.support_distance, es_kind)]
        model = build_model_from_geometry(
            element_tube_extents(el), args.joint or "rigid",
            supports, wind, mat or el.material,
            args.pin_diameter, args.n_pins, yield_stress)
        result = analyze(model)
        title = f"{el.name} ({el.freq / 1e6:.2f} MHz, mount {el.mount_pos * 1000:.0f} mm)"
        print()
        pair = None
        if args.es_model == "both" and el.support_distance > 0.0 and args.support is None:
            _, models, results = _analyze_es_pair(el, wind, args, parser)
            analyses = list(zip(("pin", "sag"), models, results))
            print(render_es_comparison(analyses, wind))
            print()
            # the full detail report shows the rigid-pin model; the sag summary
            # is already in the comparison block above
            model, result = models[0], results[0]
            pair = (models[1], results[1])
            print(render_report(model, result, wind, damping=args.damping,
                                height=height, z_0=args.z0, title=title))
        else:
            print(render_report(model, result, wind, damping=args.damping,
                                height=height, z_0=args.z0, title=title))
        path = _maybe_plot(args, model, result, el.name, wind, pair=pair)
        if path:
            print(f"Plot saved: {path}")


if __name__ == "__main__":
    main()
