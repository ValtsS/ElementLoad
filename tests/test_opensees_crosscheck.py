import pytest

from elementload.assembly import Tube, build_telescoping
from elementload.geometry import tube_area
from elementload.report import wind_and_weight_load
from elementload.solver import (
    Node,
    Element,
    Support,
    DistributedLoad,
    solve,
    element_end_forces,
)

try:
    import openseespy.opensees as openseespy
except (ImportError, RuntimeError):
    pytest.skip("openseespy not importable (needs Python 3.12 venv .venv-osees)",
                allow_module_level=True)

E = 69.0e9


def _right_half_segments(tubes, overlap, joint):
    _, segments = build_telescoping(tubes, overlap, joint)
    return [s for s in segments if s.x0 >= 0.0]


def _nodes(segments):
    xs = sorted({0.0} | {s.x0 for s in segments} | {s.x1 for s in segments})
    return xs


def _our_cantilever(segments, loads):
    xs = _nodes(segments)
    nodes = [Node(x) for x in xs]
    index = {x: k for k, x in enumerate(xs)}
    elements = [Element(index[s.x0], index[s.x1], s.e_modulus, s.inertia) for s in segments]
    supports = [Support(0, "fixed")]
    u = solve(nodes, elements, supports, loads)
    moments = []
    for i in range(len(segments)):
        _, m_i, _, m_j = element_end_forces(elements, i, nodes, u, loads)
        moments.extend([abs(m_i), abs(m_j)])
    return u, max(moments)


def _opensees_cantilever(segments, loads):
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    xs = _nodes(segments)
    for k, x in enumerate(xs, start=1):
        ops.node(k, x, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    for k, seg in enumerate(segments, start=1):
        i = xs.index(seg.x0) + 1
        j = xs.index(seg.x1) + 1
        ops.element(
            "elasticBeamColumn", k, i, j,
            tube_area(seg.outer_d, seg.wall), seg.e_modulus, seg.inertia, 1,
        )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for k, load in enumerate(loads, start=1):
        ops.eleLoad("-ele", k, "-type", "-beamUniform", -load.q)
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    disp = [ops.nodeDisp(k, 2) for k in range(1, len(xs) + 1)]
    moments = []
    for k in range(1, len(segments) + 1):
        f = ops.eleResponse(k, "localForce")
        moments.extend([abs(f[2]), abs(f[5])])
    ops.wipe()
    return disp, max(moments)


def _distributed_loads(segments, wind_speed):
    return [DistributedLoad(i, wind_and_weight_load(seg, wind_speed, "al6061-t6"))
            for i, seg in enumerate(segments)]


def test_opensees_crosscheck_telescoping_cantilever():
    tubes = [Tube(0.030, 0.002, 0.600),
             Tube(0.025, 0.002, 0.600),
             Tube(0.020, 0.002, 0.600),
             Tube(0.016, 0.0015, 0.600)]
    for joint in ("rigid", "slipped"):
        segments = _right_half_segments(tubes, 0.05, joint)
        loads = _distributed_loads(segments, 50.0)
        u_ours, m_ours = _our_cantilever(segments, loads)
        disp_osees, m_osees = _opensees_cantilever(segments, loads)
        for (w, _), w_osees in zip(u_ours, disp_osees):
            assert abs(w) == pytest.approx(abs(w_osees), rel=1e-6)
        assert m_ours == pytest.approx(m_osees, rel=1e-6)
