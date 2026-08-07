"""Reader for DXEngineering YagiMechanical .stm antenna files (Format Version 2.0).

Each telescoping element is stored as a symmetric double cantilever: for every tube
segment, IS/IE are the inner and outer END coordinates (inches) measured from the
element centre on one side, and X/T are the outer diameter and wall thickness. A
'B*' block is the antenna boom (single non-telescoping tube) and is not part of the
per-element analysis. YS (psi) is the segment yield stress and MAT a free-form
material label, so the yield strength is taken from the file, not from the label.
"""

import re
from dataclasses import dataclass, field

from elementload.assembly import TubeExtent

IN_TO_M = 0.0254
FT_TO_M = 0.3048
MPH_TO_MS = 0.44704
PSI_TO_PA = 6894.757293168361
MHZ_TO_HZ = 1.0e6

_UNITS = {
    "in": IN_TO_M,
    "ft": FT_TO_M,
    "mph": MPH_TO_MS,
    "psi": PSI_TO_PA,
    "mhz": MHZ_TO_HZ,
}

_MATERIAL_KEYS = {
    "6060-t6": "al6060-t6",
    "6060-t66": "al6060-t66",
    "6061-t6": "al6061-t6",
    "6063-t6": "al6063-t6",
}


@dataclass
class StmSegment:
    outer_d: float
    wall: float
    inner_start: float
    inner_end: float
    yield_stress: float
    material: str


@dataclass
class StmElement:
    key: str
    name: str
    mount_pos: float
    element_spacing: float
    segment_count: int
    freq: float
    driven: bool
    segments: list = field(default_factory=list)

    @property
    def material(self):
        return self.segments[0].material if self.segments else ""

    @property
    def material_key(self):
        """Known material key for the label, or None for a free-form/custom label."""
        return _MATERIAL_KEYS.get(self.material.lower())

    @property
    def yield_stress(self):
        return max((s.yield_stress for s in self.segments), default=0.0)


@dataclass
class StmFile:
    antenna: str
    height: float
    wind_speed: float
    elements: list
    boom: object


def _parse_token(token):
    key, _, value = token.partition("=")
    key = key.strip()
    value = value.strip()
    if not value:
        return key, None
    if value.startswith('"'):
        return key, value.strip('"')
    m = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([A-Za-z]*)\s*", value)
    if not m:
        return key, value
    return key, float(m.group(1)) * _UNITS.get(m.group(2).lower(), 1.0)


def _parse_fields(text):
    fields = {}
    for token in text.split(";"):
        k, v = _parse_token(token)
        if v is not None:
            fields[k] = v
    return fields


def parse_stm(text):
    header = {}
    elements = []
    boom = None
    by_key = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("DXEngineering"):
            continue
        if line.startswith("AN="):
            header.update(_parse_fields(line))
            continue
        if line.startswith("EC="):
            continue
        m = re.match(r'^([EB]\d+)="([^"]*)"', line)
        if m:
            key, name = m.group(1), m.group(2)
            fields = _parse_fields(line.split(";", 1)[1])
            el = StmElement(
                key,
                name,
                fields.get("MP", 0.0),
                fields.get("ES", 0.0),
                int(fields.get("SC", 0)),
                fields.get("RF", 0.0),
                str(fields.get("DE", "False")).lower() == "true",
                [],
            )
            by_key[key] = el
            if key.startswith("B"):
                boom = el
            else:
                elements.append(el)
            continue
        m = re.match(r"^([EB]\d+)(S\d+)\s+(.*)$", line)
        if m:
            fields = _parse_fields(m.group(3))
            seg = StmSegment(
                fields["X"],
                fields["T"],
                fields["IS"],
                fields["IE"],
                fields["YS"],
                str(fields["MAT"]),
            )
            by_key[m.group(1)].segments.append(seg)
    return StmFile(header.get("AN", ""), header.get("H", 10.0),
                   header.get("WS", 30.0), elements, boom)


def read_stm(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_stm(f.read())


def element_tube_extents(el):
    """Per-side tube extents of an element, innermost (largest) tube first."""
    ordered = sorted(el.segments, key=lambda s: (s.inner_start, -s.outer_d))
    return [TubeExtent(s.outer_d, s.wall, s.inner_start, s.inner_end) for s in ordered]
