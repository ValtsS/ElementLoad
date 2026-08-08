"""Structural optimization of a telescoping element against a tube stock catalog.

The optimizer minimizes the element mass subject to survival-wind-speed and
telescoping-fit constraints. Two stages:

- Stage 1 (deterministic envelope sizing): for a fixed wind speed, the bending
  moment envelope along the element fixes the required section modulus at each
  station; pick the smallest catalog tube whose modulus clears the demand.
- Stage 2 (iterative drag coupling): the envelope itself depends on the chosen
  diameters (drag grows with D), so re-solve and re-size to a fixed point.

Design variables: per-side tube (D, t) drawn from the catalog, one tube per stock
segment length. The number of tubes and their lengths stay fixed (from the .stm);
only the section is re-picked.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import itertools
import os

from elementload.assembly import TubeExtent
from elementload.failure import local_buckling_stress
from elementload.geometry import tube_area, tube_section_modulus
from elementload.report import (
    _wind_loads,
    _wind_supports,
    build_model_from_geometry,
    element_end_forces,
    solve,
    survival_wind_speed,
)

MM_TO_M = 1.0e-3
MAX_ITER = 20
BORE_EPS = 1.0e-9
TIP_EXTENSION_COEFF = 0.1


def parse_catalog_lines(lines):
    """Parse catalog text lines 'D_mm, t_mm' into a sorted list of (outer_d, wall) in m."""
    sizes = {}
    for lineno, line in enumerate(lines, 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            d_mm, t_mm = (float(x) for x in text.split(","))
        except ValueError:
            raise ValueError(f"tube catalog line {lineno}: expected 'D_mm, t_mm': {text!r}")
        key = (d_mm * MM_TO_M, t_mm * MM_TO_M)
        if key in sizes:
            raise ValueError(f"tube catalog line {lineno}: duplicate size {d_mm}x{t_mm}")
        sizes[key] = True
    tubes = sorted(sizes)
    if not tubes:
        raise ValueError("tube catalog is empty")
    return tubes


def read_catalog(path):
    """Read a tube stock catalog CSV file (D_mm, t_mm rows)."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_catalog_lines(f.read().splitlines())


def tube_mass(tube, length):
    """Mass (kg) of one tube of given outer_d/wall and length (m)."""
    return tube_area(tube[0], tube[1]) * 2700.0 * length


def element_mass(tubes):
    """Total mass (kg) of both sides of a symmetric telescoping element."""
    return sum(tube_mass((t.outer_d, t.wall), 2.0 * (t.inner_end - t.inner_start))
               for t in tubes)


def per_tube_max_moment(tubes, joint, supports, wind_speed, material,
                        yield_stress=None, pin_diameter=0.004, n_pins=2):
    """Max bending moment carried by each tube (same order as `tubes`) under the
    wind+weight load at wind_speed. Wind is only restrained by the non-sag
    supports; the moment is the orthogonal sum of wind and self-weight solves."""
    model = build_model_from_geometry(tubes, joint, supports, wind_speed, material,
                                      pin_diameter, n_pins, yield_stress)
    wind_supports = _wind_supports(model)
    u_wind = solve(model.nodes, model.elements, wind_supports, _wind_loads(model))
    from elementload.report import self_weight_loads
    u_weight = solve(model.nodes, model.elements, model.supports, self_weight_loads(model))
    moments = [0.0] * len(tubes)
    for i, seg in enumerate(model.segments):
        v_wi, m_wi, v_wj, m_wj = element_end_forces(
            model.elements, i, model.nodes, u_wind, _wind_loads(model))
        v_gi, m_gi, v_gj, m_gj = element_end_forces(
            model.elements, i, model.nodes, u_weight, self_weight_loads(model))
        m_i = math.hypot(m_wi, m_gi)
        m_j = math.hypot(m_wj, m_gj)
        for k, t in enumerate(tubes):
            if seg.outer_d == t.outer_d:
                moments[k] = max(moments[k], m_i, m_j)
    return moments


def required_allowable_stress(outer_d, wall, yield_stress, e_modulus=69.0e9):
    """Design stress limit of a tube = min(yield, local buckling)."""
    return min(yield_stress, local_buckling_stress(outer_d, wall, e_modulus))


def pick_tube(moment, catalog, yield_stress):
    """Lightest catalog tube that carries `moment` without yielding or local
    buckling. Returns the (outer_d, wall) pair, or None if nothing suffices."""
    best = None
    best_mass = float("inf")
    for d, t in catalog:
        z = tube_section_modulus(d, t)
        if z <= 0.0:
            continue
        sigma = required_allowable_stress(d, t, yield_stress)
        if moment / z <= sigma:
            mass = tube_mass((d, t), 1.0)
            if mass < best_mass:
                best_mass = mass
                best = (d, t)
    return best


def _d(thing):
    return thing[0] if not hasattr(thing, "outer_d") else thing.outer_d


def _t(thing):
    return thing[1] if not hasattr(thing, "wall") else thing.wall


def fits_inside(inner_tube, outer_tube, clearance=1.0e-3):
    """True if `outer_tube` (the smaller, tip-side tube) telescopes into
    `inner_tube` (larger, centre-side): its outer diameter must clear the inner
    tube's bore by `clearance`."""
    bore = _d(inner_tube) - 2.0 * _t(inner_tube)
    return _d(outer_tube) <= bore - clearance + BORE_EPS


def size_from_extents(tubes, joint, supports, wind_speed, material,
                      yield_stress=None, catalog=None, pin_diameter=0.004,
                      n_pins=2, clearance=1.0e-3, length_step=0.0):
    """Deterministic envelope sizing: choose the lightest catalog tube for each
    stock tube position so that no tube yields or locally buckles at wind_speed,
    keeping the telescoping fit (each tube fits inside its inner neighbour with
    `clearance` slop). Re-solves to fixed point because the envelope depends on
    the diameters (drag grows with D). With `length_step > 0`, per-side tube
    lengths are snapped to that increment for practical manufacturing. Returns a
    new TubeExtent list.

    The fixed-point map is monotone on the feasible telescoping lattice but not
    on the full product lattice, so a single starting chain can converge to a
    suboptimal fixed point (a heavier chain is still self-consistent). To avoid
    that, the iteration runs from several diverse starts -- the given chain, the
    bottom (lightest) feasible chain, and the heaviest catalog chain -- and the
    lightest feasible result is returned."""
    from elementload.report import allowables
    if yield_stress is None:
        yield_stress = allowables(material)["yield"]
    starts = [list(tubes)]
    if catalog is not None:
        try:
            starts.append(bottom_feasible_chain(tubes, catalog, clearance))
        except ValueError:
            pass
        top = [TubeExtent(d, t_w, tube.inner_start, tube.inner_end)
               for (d, t_w), tube in zip([catalog[-1]] * len(tubes), tubes)]
        starts.append(top)
    best = None
    first_error = None
    for start in starts:
        try:
            candidate = _fixed_point_iteration(
                start, joint, supports, wind_speed, material, yield_stress,
                catalog, pin_diameter, n_pins, clearance, length_step)
        except ValueError as e:
            if first_error is None:
                first_error = e
            continue
        if best is None or element_mass(candidate) < element_mass(best):
            best = candidate
    if best is None:
        raise first_error if first_error is not None else ValueError(
            "no catalog tube can carry the required loads at this wind speed")
    return best


def bottom_feasible_chain(tubes, catalog, clearance):
    """The lightest catalog chain that telescopes with `clearance`, built
    tip-to-root: the smallest tube at each position whose bore can host the tube
    already chosen outside it. This is the bottom of the feasible-chain lattice;
    the fixed-point map is monotone there, so iterating from it reaches the
    least (lightest) fixed point."""
    n = len(tubes)
    picks = [None] * n
    required_bore = 0.0
    for k in range(n - 1, -1, -1):
        best = None
        best_mass = float("inf")
        for d, t in catalog:
            if d - 2.0 * t < required_bore - BORE_EPS:
                continue
            mass = tube_mass((d, t), 1.0)
            if mass < best_mass:
                best_mass = mass
                best = (d, t)
        if best is None:
            raise ValueError(
                f"no catalog tube has a bore large enough to host the telescoping "
                f"chain (needs {required_bore * 1000:.1f} mm)")
        picks[k] = best
        required_bore = _d(best) + clearance
    return [TubeExtent(d, t_w, tube.inner_start, tube.inner_end)
            for (d, t_w), tube in zip(picks, tubes)]


def _fixed_point_iteration(start, joint, supports, wind_speed, material,
                           yield_stress, catalog, pin_diameter, n_pins,
                           clearance, length_step):
    """Run the envelope sizing to a fixed point starting from `start`."""
    current = snap_lengths(start, length_step) if length_step > 0.0 else list(start)
    for _ in range(MAX_ITER):
        moments = per_tube_max_moment(current, joint, supports, wind_speed,
                                      material, yield_stress, pin_diameter, n_pins)
        picks = _size_telescoped(moments, catalog, yield_stress, clearance)
        next_tubes = [TubeExtent(d, t_w, tube.inner_start, tube.inner_end)
                      for (d, t_w), tube in zip(picks, current)]
        if all(n.outer_d == c.outer_d and n.wall == c.wall
               for n, c in zip(next_tubes, current)):
            return next_tubes
        current = next_tubes
    return current


def snap_lengths(tubes, step):
    """Snap each per-side tube length to a multiple of `step`, keeping the inner
    (centre-side) joint fixed. The tip tube keeps its exact original end, so the
    total element length is preserved and the tip absorbs the rounding residual
    of the inner tubes (only the non-tip tubes are rounded to the increment)."""
    snapped = []
    for i, t in enumerate(tubes):
        if i == len(tubes) - 1:
            inner_end = t.inner_end
        else:
            inner_end = round((t.inner_end - t.inner_start) / step) * step + t.inner_start
        snapped.append(TubeExtent(t.outer_d, t.wall, t.inner_start, inner_end))
    return snapped


def effective_diameter(tubes):
    """Length-weighted mean outer diameter (m) over one side of the element."""
    total_len = sum(t.inner_end - t.inner_start for t in tubes)
    if total_len <= 0.0:
        return 0.0
    return sum(_d(t) * (t.inner_end - t.inner_start) for t in tubes) / total_len


def tip_extension_for(tubes, ref_tubes, coeff=TIP_EXTENSION_COEFF):
    """Per-side tip extension (m) compensating the resonance up-shift when the
    optimized element's mean diameter is smaller than the reference: coeff *
    (1 - d_opt/d_ref) * half_length. Returns 0 when the optimized element is
    not thinner (mean diameter via `effective_diameter`)."""
    d_opt = effective_diameter(tubes)
    d_ref = effective_diameter(ref_tubes)
    if d_ref <= 0.0 or d_opt >= d_ref:
        return 0.0
    half = max(t.inner_end for t in tubes)
    return coeff * (1.0 - d_opt / d_ref) * half


def extend_tip(tubes, extension):
    """Return a copy of `tubes` with the tip tube (last in the list, outermost)
    extended by `extension` per side; all other tubes unchanged."""
    if extension <= 0.0 or not tubes:
        return list(tubes)
    out = [TubeExtent(t.outer_d, t.wall, t.inner_start, t.inner_end) for t in tubes]
    tip = out[-1]
    out[-1] = TubeExtent(tip.outer_d, tip.wall, tip.inner_start, tip.inner_end + extension)
    return out


def size_with_tip_extension(tubes, ref_tubes, joint, supports, wind_speed, material,
                            yield_stress=None, catalog=None, pin_diameter=0.004,
                            n_pins=2, clearance=1.0e-3, length_step=0.0,
                            tip_extension=None, coeff=TIP_EXTENSION_COEFF):
    """Size `tubes` from the catalog and optionally extend the tip to compensate
    the resonance up-shift from thinner tubes. `tip_extension`:
      - None: plain sizing, no extension;
      - a number: fixed per-side extension (m);
      - "auto": coeff * (1 - d_opt/d_ref) * half_length with d_ref from
        `ref_tubes`, iterated to a fixed point because extending the tip changes
        the drag envelope and the envelope changes the diameters the extension
        is derived from.
    Returns (opt_tubes, applied_extension_m)."""
    if tip_extension is None:
        opt = size_from_extents(tubes, joint, supports, wind_speed, material,
                                yield_stress, catalog=catalog,
                                pin_diameter=pin_diameter, n_pins=n_pins,
                                clearance=clearance, length_step=length_step)
        return opt, 0.0
    if tip_extension != "auto":
        ext = float(tip_extension)
        opt = size_from_extents(extend_tip(tubes, ext), joint, supports, wind_speed,
                                material, yield_stress, catalog=catalog,
                                pin_diameter=pin_diameter, n_pins=n_pins,
                                clearance=clearance, length_step=length_step)
        return opt, ext
    ext = 0.0
    opt = None
    for _ in range(MAX_ITER):
        opt = size_from_extents(extend_tip(tubes, ext), joint, supports, wind_speed,
                                material, yield_stress, catalog=catalog,
                                pin_diameter=pin_diameter, n_pins=n_pins,
                                clearance=clearance, length_step=length_step)
        new_ext = tip_extension_for(opt, ref_tubes, coeff)
        if abs(new_ext - ext) <= 1e-3 * max(1.0, abs(ext)):
            # re-size at the converged extension so the returned chain matches it
            if new_ext != ext:
                opt = size_from_extents(extend_tip(tubes, new_ext), joint, supports,
                                        wind_speed, material, yield_stress,
                                        catalog=catalog, pin_diameter=pin_diameter,
                                        n_pins=n_pins, clearance=clearance,
                                        length_step=length_step)
            return opt, new_ext
        ext = new_ext
    return opt, ext


def _size_telescoped(moments, catalog, yield_stress, clearance):
    """Tip-to-root sizing: start from the tip (smallest moment) and work inward.
    Each tube is the smallest catalog tube that (a) carries its own moment and
    (b) has a bore large enough to host the tube already chosen outside it
    (with `clearance`). The required bore propagates inward, which is exactly
    how a real telescoping element must be built."""
    n = len(moments)
    picks = [None] * n
    required_bore = 0.0
    failed_bore = None
    for k in range(n - 1, -1, -1):
        best = None
        best_mass = float("inf")
        for d, t in catalog:
            if d - 2.0 * t < required_bore - BORE_EPS:
                continue
            z = tube_section_modulus(d, t)
            if z <= 0.0:
                continue
            if moments[k] / z > required_allowable_stress(d, t, yield_stress):
                continue
            mass = tube_mass((d, t), 1.0)
            if mass < best_mass:
                best_mass = mass
                best = (d, t)
        if best is None and failed_bore is None:
            failed_bore = required_bore
        picks[k] = best if best is not None else (0.0, 0.0)
        required_bore = _d(picks[k]) + clearance
    if any(_d(p) == 0.0 for p in picks):
        failed = next(k for k, p in enumerate(picks) if _d(p) == 0.0)
        need = moments[failed]
        max_d, max_t = catalog[-1]
        if max_d - 2.0 * max_t < failed_bore - BORE_EPS:
            cause = (f"the telescoping bore chain needs {failed_bore * 1000:.1f} mm "
                     f"but the largest catalog tube has only "
                     f"{(max_d - 2.0 * max_t) * 1000:.1f} mm of bore")
        else:
            cause = (f"the largest catalog tube cannot carry the "
                     f"{need:.1f} N m demand at this position")
        raise ValueError(
            f"no catalog tube can carry the required loads at this wind speed: "
            f"{cause} (tip-side position {failed})")
    return picks


def layout_from_exposed(exposed, overlap, half_length):
    """Tube extents (innermost first) from per-tube exposed lengths: exposed[k] is
    how far tube k protrudes beyond the tube inside it (exposed[0] = root tube
    length), consecutive tubes overlapping by `overlap`. Returns a list of
    TubeExtent, or None if the geometry is degenerate (negative start)."""
    if any(e <= 0.0 for e in exposed):
        return None
    if overlap < 0.0:
        return None
    end = 0.0
    tubes = []
    for k, e in enumerate(exposed):
        start = 0.0 if k == 0 else end - overlap
        if start < -1e-9:
            return None
        end += e
        tubes.append(TubeExtent(0.0, 0.002, max(start, 0.0), end))
    if abs(end - half_length) > 1e-6:
        return None
    return tubes


def geom_exposed(n, ratio, half_length, first=None):
    """Exposed lengths of `n` telescoping tubes in a geometric taper ratio^k,
    normalized to `half_length`. With `first` (a sleeve collar), exposed[0] is
    that length and the rest are a geometric series over the remainder."""
    xs = [ratio ** k for k in range(n)]
    if first is not None:
        rest = half_length - first
        if rest <= 0.0 or n < 2:
            return None
        s = sum(xs[1:])
        xs = [first] + [rest * x / s for x in xs[1:]]
    else:
        xs = [half_length * x / sum(xs) for x in xs]
    return xs


def _layout_search_job(job):
    """One independent layout-search job, picklable for ProcessPoolExecutor.
    Returns a result dict, or None when the layout/sizing is infeasible."""
    n, overlap, ratio, sleeve, wind, half_length, material, yield_stress, \
        joint, supports, catalog, pin_diameter, n_pins, clearance, length_step, \
        tip_extension, ref_tubes, coeff = job
    exposed = geom_exposed(n, ratio, half_length, first=sleeve)
    if exposed is None:
        return None
    layout = layout_from_exposed(exposed, overlap, half_length)
    if layout is None:
        return None
    try:
        opt, applied = size_with_tip_extension(
            layout, ref_tubes, joint, supports, wind, material,
            yield_stress, catalog=catalog, pin_diameter=pin_diameter,
            n_pins=n_pins, clearance=clearance, length_step=length_step,
            tip_extension=tip_extension, coeff=coeff)
    except ValueError:
        return None
    model = build_model_from_geometry(opt, joint, supports, wind, material,
                                      pin_diameter, n_pins, yield_stress)
    survival = survival_wind_speed(model, "yield")
    chain = ";".join(f"{t.outer_d * 1000:.0f}x{t.wall * 1000:g}" for t in opt)
    extents = [(t.outer_d, t.wall, t.inner_start, t.inner_end) for t in opt]
    return {
        "n": n, "overlap": overlap, "ratio": ratio, "sleeve": sleeve,
        "wind": wind, "mass": element_mass(opt), "survival": survival,
        "chain": chain, "tip_ext": applied, "extents": extents,
    }


def search_layouts(tubes, joint, supports, wind_speed, material,
                   yield_stress=None, catalog=None, pin_diameter=0.004, n_pins=2,
                   clearance=1.0e-3, length_step=0.0,
                   tip_extension=None, ref_tubes=None, coeff=TIP_EXTENSION_COEFF,
                   n_range=(3, 10), overlaps=(0.10, 0.15, 0.20),
                   ratios=(0.6, 0.8, 1.0, 1.2, 1.4, 1.6),
                   sleeves=(None, 0.150, 0.250, 0.400),
                   target_winds=None, workers=None):
    """Brute-force the layout design space: tube count, overlap, exposed-length
    taper and a sleeve collar, sizing each layout from the catalog at the target
    winds and reporting the mass and yield-survival of every feasible candidate.
    `tip_extension` is passed through to each sizing (None = none, a length in m
    = fixed per-side, "auto" = mean-diameter-based compensation using
    `ref_tubes`, defaulting to `tubes`). The grid is embarrassingly parallel
    and is spread over `workers` processes (default: os.cpu_count()). Returns a
    list of result dicts sorted by survival wind speed."""
    half_length = max(t.inner_end for t in tubes)
    if ref_tubes is None:
        ref_tubes = tubes
    if target_winds is None:
        target_winds = [wind_speed]
    jobs = []
    for n in range(n_range[0], n_range[1] + 1):
        for overlap, ratio, sleeve in itertools.product(overlaps, ratios, sleeves):
            for wind in target_winds:
                jobs.append((n, overlap, ratio, sleeve, wind, half_length,
                             material, yield_stress, joint, tuple(supports),
                             catalog, pin_diameter, n_pins, clearance, length_step,
                             tip_extension, ref_tubes, coeff))
    if workers is None:
        workers = os.cpu_count() or 1
    results = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_layout_search_job, jobs, chunksize=4):
                if r is not None:
                    results.append(r)
    else:
        for job in jobs:
            r = _layout_search_job(job)
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: r["survival"])
    return results
