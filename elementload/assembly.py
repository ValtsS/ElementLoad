from dataclasses import dataclass

from elementload.geometry import tube_inertia


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
            )
        )
        x += exposed
    return segments


def build_telescoping(tubes, overlap, joint):
    right = _build_half(tubes, overlap, joint)
    left = [
        Segment(-s.x1, -s.x0, s.outer_d, s.wall, s.inertia, s.e_modulus) for s in reversed(right)
    ]
    segments = left + right
    nodes = sorted({0.0} | {s.x0 for s in segments} | {s.x1 for s in segments})
    return nodes, segments
