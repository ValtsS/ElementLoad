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
from dataclasses import replace

from elementload.assembly import TubeExtent
from elementload.failure import local_buckling_stress
from elementload.geometry import tube_area, tube_section_modulus
from elementload.report import (
    _wind_loads,
    _wind_supports,
    build_model_from_geometry,
    element_end_forces,
    solve,
)

MM_TO_M = 1.0e-3
MAX_ITER = 20
BORE_EPS = 1.0e-9


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
    new TubeExtent list."""
    from elementload.report import allowables
    if yield_stress is None:
        yield_stress = allowables(material)["yield"]
    current = snap_lengths(tubes, length_step) if length_step > 0.0 else list(tubes)
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
    (centre-side) joint fixed and absorbing the rounding at the tip."""
    snapped = []
    for i, t in enumerate(tubes):
        if i == len(tubes) - 1:
            inner_end = max(round((t.inner_end - t.inner_start) / step) * step
                            + t.inner_start, t.inner_start + step)
        else:
            inner_end = round((t.inner_end - t.inner_start) / step) * step + t.inner_start
        snapped.append(TubeExtent(t.outer_d, t.wall, t.inner_start, inner_end))
    return snapped


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
