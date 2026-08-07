from dataclasses import dataclass

from elementload.geometry import tube_inertia, tube_mass_per_length


@dataclass(frozen=True)
class Tube:
    outer_d: float
    wall: float
    length: float
    density: float = 2700.0
    e_modulus: float = 69.0e9


@dataclass(frozen=True)
class Segment:
    x0: float
    x1: float
    outer_d: float
    wall: float
    inertia: float
    e_modulus: float = 69.0e9
    is_overlap: bool = False
    mass_per_length: float = 0.0


@dataclass(frozen=True)
class TubeExtent:
    outer_d: float
    wall: float
    inner_start: float
    inner_end: float
    e_modulus: float = 69.0e9
    density: float = 2700.0


def _overlap_inertia(prev, cur, joint):
    i_prev = tube_inertia(prev.outer_d, prev.wall)
    i_cur = tube_inertia(cur.outer_d, cur.wall)
    if joint == "rigid":
        return i_prev + i_cur
    if joint == "slipped":
        return i_prev
    raise ValueError(f"unknown joint model: {joint}")


def _build_half(tubes, overlap, joint):
    segments = []
    center = tubes[0]
    inner_end = center.length / 2 - overlap
    if inner_end > 0:
        segments.append(
            Segment(
                0,
                inner_end,
                center.outer_d,
                center.wall,
                tube_inertia(center.outer_d, center.wall),
                center.e_modulus,
                False,
                tube_mass_per_length(center.outer_d, center.wall, center.density),
            )
        )
    x = inner_end if inner_end > 0 else 0
    for i in range(1, len(tubes)):
        prev = tubes[i - 1]
        cur = tubes[i]
        segments.append(
            Segment(
                x,
                x + overlap,
                max(prev.outer_d, cur.outer_d),
                max(prev.wall, cur.wall),
                _overlap_inertia(prev, cur, joint),
                prev.e_modulus,
                True,
                tube_mass_per_length(prev.outer_d, prev.wall, prev.density)
                + tube_mass_per_length(cur.outer_d, cur.wall, cur.density),
            )
        )
        x += overlap
        exposed = cur.length - overlap
        segments.append(
            Segment(
                x,
                x + exposed,
                cur.outer_d,
                cur.wall,
                tube_inertia(cur.outer_d, cur.wall),
                cur.e_modulus,
                False,
                tube_mass_per_length(cur.outer_d, cur.wall, cur.density),
            )
        )
        x += exposed
    return segments


def build_telescoping(tubes, overlap, joint):
    right = _build_half(tubes, overlap, joint)
    left = [
        Segment(-s.x1, -s.x0, s.outer_d, s.wall, s.inertia, s.e_modulus,
                s.is_overlap, s.mass_per_length)
        for s in reversed(right)
    ]
    segments = left + right
    nodes = sorted({0.0} | {s.x0 for s in segments} | {s.x1 for s in segments})
    return nodes, segments


def build_from_extents(tubes, joint):
    """Exact telescoping layout from per-side tube extents: each tube spans
    (inner_start, inner_end) measured from the element centre on one side, ordered
    from the innermost (largest) tube outward. Overlap intervals carry the combined
    (rigid) or outer-tube-only (slipped) inertia of every tube present; the visible
    outer diameter at a station is the largest tube present there."""
    if joint not in ("rigid", "slipped"):
        raise ValueError(f"unknown joint model: {joint}")
    tubes = sorted(tubes, key=lambda t: (t.inner_start, -t.outer_d))
    bounds = sorted({t.inner_start for t in tubes} | {t.inner_end for t in tubes})
    if bounds[0] > 0.0:
        raise ValueError("innermost tube must start at the element centre (0)")
    right = []
    for a, b in zip(bounds, bounds[1:]):
        present = [t for t in tubes if t.inner_start <= a and b <= t.inner_end]
        if not present:
            continue
        outer = max(present, key=lambda t: t.outer_d)
        if joint == "rigid":
            inertia = sum(tube_inertia(t.outer_d, t.wall) for t in present)
        else:
            inertia = max(tube_inertia(t.outer_d, t.wall) for t in present)
        mass = sum(tube_mass_per_length(t.outer_d, t.wall, t.density) for t in present)
        right.append(Segment(a, b, outer.outer_d, outer.wall, inertia, outer.e_modulus,
                             len(present) > 1, mass))
    left = [
        Segment(-s.x1, -s.x0, s.outer_d, s.wall, s.inertia, s.e_modulus,
                s.is_overlap, s.mass_per_length)
        for s in reversed(right)
    ]
    segments = left + right
    nodes = sorted({0.0} | {s.x0 for s in segments} | {s.x1 for s in segments})
    return nodes, segments
