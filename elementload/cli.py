import argparse
import math

from elementload.report import build_model, analyze, survival_wind_speed
from elementload.solver import internal_moment

KMH_PER_MPS = 3.6
_MODES = ("yield", "buckling", "pin_shear")


def _tube_spec(s):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected D,t,L got {s!r}")
    return tuple(parts)


def _support_spec(s):
    pos, _, kind = s.partition(":")
    if kind not in ("fixed", "pin", "roller"):
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


def render_report(model, result, wind_speed):
    lines = []
    lines.append("Element load analysis")
    lines.append("=====================")
    lines.append("")
    lines.append("Tubes (D x t, length):")
    for i, (d, t, L) in enumerate(model.tube_specs, start=1):
        lines.append(f"  {i}: {d * 1000:.1f} x {t * 1000:.2f} mm, {L * 1000:.0f} mm")
    lines.append(f"Overlap: {model.overlap * 1000:.0f} mm, joint model: {model.joint}")
    supports = ", ".join(f"{p}:{k}" for p, k in model.support_specs)
    lines.append(f"Supports: {supports}")
    lines.append(f"Material: {model.material}")
    lines.append(f"Pins: {model.n_pins} x {model.pin_diameter * 1000:.1f} mm at each joint")
    lines.append("")
    lines.append(f"Analysis at wind speed {wind_speed:.1f} m/s ({wind_speed * KMH_PER_MPS:.0f} km/h):")
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
        lines.append(f"  {mode:>10}: {v * KMH_PER_MPS:7.1f} km/h ({v:5.1f} m/s)")
    limiting = min(survivals, key=survivals.get)
    lines.append(f"  LIMITING MODE: {limiting} at {survivals[limiting] * KMH_PER_MPS:.1f} km/h")
    lines.append("")
    lines.append("Dynamics (damping zeta = 0.01):")
    from elementload.report import natural_frequencies
    from elementload.dynamics import critical_wind_speed, scruton_number
    from elementload.report import _model_mass
    freqs = natural_frequencies(model, n_modes=3)
    for i, f in enumerate(freqs, start=1):
        v_crit = critical_wind_speed(f, model.tube_specs[0][0])
        lines.append(f"  mode {i}: f = {f:6.2f} Hz, lock-in at V_crit = {v_crit * KMH_PER_MPS:6.0f} km/h")
    m_eff = _model_mass(model)
    d_ref = model.tube_specs[0][0]
    sc = scruton_number(m_eff, 0.01, d_ref)
    lines.append(f"  effective mass {m_eff:.3f} kg/m, Scruton Sc = {sc:.1f}")
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="elementload",
        description="FEM load analysis of a telescoping Yagi antenna element",
    )
    parser.add_argument("--tube", action="append", type=_tube_spec, required=True,
                        metavar="D,t,L", help="tube outer diameter, wall, length (SI); repeatable")
    parser.add_argument("--overlap", type=float, required=True, metavar="M",
                        help="telescoping overlap length (m)")
    parser.add_argument("--joint", choices=("rigid", "slipped"), required=True,
                        help="joint stiffness model")
    parser.add_argument("--support", action="append", type=_support_spec, required=True,
                        metavar="POS:KIND", help="support, e.g. center:fixed; repeatable")
    parser.add_argument("--wind", type=float, default=30.0, metavar="MPS",
                        help="analysis wind speed in m/s (default 30)")
    parser.add_argument("--material", default="al6061-t6",
                        choices=("al6061-t6", "al6063-t6"))
    parser.add_argument("--pin-diameter", type=float, default=0.004, metavar="M")
    parser.add_argument("--n-pins", type=int, default=2, metavar="N")
    args = parser.parse_args(argv)

    model = build_model(args.tube, args.overlap, args.joint, args.support,
                        args.wind, args.material, args.pin_diameter, args.n_pins)
    result = analyze(model)
    print(render_report(model, result, args.wind))


if __name__ == "__main__":
    main()
