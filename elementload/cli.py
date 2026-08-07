import argparse
import math

from elementload.report import build_model, analyze, survival_wind_speed, build_model_from_geometry
from elementload.stm import read_stm, element_tube_extents
from elementload.solver import internal_moment

_MODES = ("yield", "buckling", "pin_shear")


def _tube_spec(s):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected D,t,L got {s!r}")
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
    parser.add_argument("--tube", action="append", type=_tube_spec,
                        metavar="D,t,L", help="tube outer diameter, wall, length (SI); repeatable")
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
    args = parser.parse_args(argv)

    if args.stm is not None:
        if args.tube is not None or args.overlap is not None:
            parser.error("--tube/--overlap are not used with --stm (geometry comes from the file)")
        return _main_stm(args, parser)

    if args.tube is None or args.overlap is None or args.joint is None or args.support is None:
        parser.error("either --stm or --tube/--overlap/--joint/--support must be given")

    wind = 30.0 if args.wind is None else args.wind
    height = 10.0 if args.height is None else args.height
    model = build_model(args.tube, args.overlap, args.joint, args.support,
                        wind, args.material or "al6061-t6",
                        args.pin_diameter, args.n_pins)
    result = analyze(model)
    print(render_report(model, result, wind, damping=args.damping,
                        height=height, z_0=args.z0))


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
        if args.es_model == "both" and el.support_distance > 0.0 and args.support is None:
            _, models, results = _analyze_es_pair(el, wind, args, parser)
            analyses = list(zip(("pin", "sag"), models, results))
            print(render_es_comparison(analyses, wind))
            print()
            # the full detail report shows the rigid-pin model; the sag summary
            # is already in the comparison block above
            model, result = models[0], results[0]
            print(render_report(model, result, wind, damping=args.damping,
                                height=height, z_0=args.z0, title=title))
        else:
            print(render_report(model, result, wind, damping=args.damping,
                                height=height, z_0=args.z0, title=title))


if __name__ == "__main__":
    main()
