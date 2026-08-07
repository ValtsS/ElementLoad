from pathlib import Path

import pytest

from elementload.assembly import build_from_extents, Segment
from elementload.geometry import tube_inertia
from elementload.report import build_model_from_geometry, analyze, wind_and_weight_load
from elementload.stm import (
    parse_stm,
    read_stm,
    element_tube_extents,
    IN_TO_M,
    PSI_TO_PA,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "mechanical2.stm"

HEADER = """DXEngineering YagiMechanical Antenna File, Format Version 2.0
AN="Antenna";    ML=0.00000 in;    SS=0.00000 in;    WLS=1.0;    H=33.00000 ft;    WS=90.65327 mph;    IT=0.00000 in;
EC=7;
"""

ELEMENT = """E0="15m RE";    MP=1.96850 in;    ES=0.00000 in;    SC=4;    RF=20.2792 MHz;    DE=False;
 E0S0    SHAPE=Round;    X=0.47244 in;    Y=0.47244 in;    T=0.03937 in;    IS=88.58268 in;    IE=144.72441 in;    YS=21756 psi;    MAT="6060-T6";
 E0S3    SHAPE=Round;    X=0.98425 in;    Y=0.98425 in;    T=0.07874 in;    IS=0.00000 in;    IE=39.37008 in;    YS=21756 psi;    MAT="6060-T6";
"""

BOOM = """B0="";    MP=118.11024 in;    RS=0.00000 in;    DS=0.00000 in;    SC=1;
 B0S0    SHAPE=Round;    X=1.96850 in;    Y=1.96850 in;    T=0.07874 in;    IS=0.00000 in;    IE=236.22047 in;    YS=21756 psi;    MAT="6060-T6";
"""


def test_parse_header_metric_conversion():
    stm = parse_stm(HEADER + ELEMENT + BOOM)
    assert stm.antenna == "Antenna"
    assert stm.height == pytest.approx(33 * 0.3048)
    assert stm.wind_speed == pytest.approx(90.65327 * 0.44704)


def test_parse_elements_and_boom():
    stm = parse_stm(HEADER + ELEMENT + BOOM)
    assert [e.name for e in stm.elements] == ["15m RE"]
    assert stm.boom is not None
    assert stm.boom.key == "B0"
    assert stm.boom.name == ""
    assert len(stm.boom.segments) == 1


def test_parse_segment_metric_conversion():
    stm = parse_stm(HEADER + ELEMENT + BOOM)
    seg = stm.elements[0].segments[0]
    assert seg.outer_d == pytest.approx(0.47244 * IN_TO_M)
    assert seg.wall == pytest.approx(0.03937 * IN_TO_M)
    assert seg.inner_start == pytest.approx(88.58268 * IN_TO_M)
    assert seg.inner_end == pytest.approx(144.72441 * IN_TO_M)
    assert seg.yield_stress == pytest.approx(21756 * PSI_TO_PA)
    assert seg.material == "6060-T6"


def test_element_freq_mount_and_yield():
    stm = parse_stm(HEADER + ELEMENT + BOOM)
    el = stm.elements[0]
    assert el.freq == pytest.approx(20.2792e6)
    assert el.mount_pos == pytest.approx(1.96850 * IN_TO_M)
    assert el.segment_count == 4
    assert el.material == "6060-T6"
    assert el.yield_stress == pytest.approx(21756 * PSI_TO_PA)


def test_read_stm_file_element_list():
    stm = read_stm(DATA)
    assert [e.name for e in stm.elements] == [
        "15m RE", "15m DE", "15m D1", "10m RE", "10m DE", "10m D1", "10m D2",
    ]
    assert stm.boom is not None


def test_element_tube_extents_15m_re():
    stm = read_stm(DATA)
    el = next(e for e in stm.elements if e.name == "15m RE")
    extents = element_tube_extents(el)
    for (e, expect) in zip(extents, [
        (0.025, 0.002, 0.0, 1.0),
        (0.020, 0.0015, 0.9, 2.35),
        (0.016, 0.0015, 2.25, 2.38),
        (0.012, 0.001, 2.25, 3.676),
    ]):
        assert (e.outer_d, e.wall, e.inner_start, e.inner_end) == pytest.approx(expect, abs=1e-6)


def test_build_from_extents_exact_nodes():
    stm = read_stm(DATA)
    el = next(e for e in stm.elements if e.name == "15m RE")
    nodes, segments = build_from_extents(element_tube_extents(el), "rigid")
    assert nodes == pytest.approx(
        [-3.676, -2.38, -2.35, -2.25, -1.0, -0.9, 0.0, 0.9, 1.0, 2.25, 2.35, 2.38, 3.676]
    )
    assert segments[-1].outer_d == pytest.approx(0.012, abs=1e-6)
    assert segments[0].outer_d == pytest.approx(0.012, abs=1e-6)


def test_triple_overlap_combines_all_present_tubes():
    stm = read_stm(DATA)
    el = next(e for e in stm.elements if e.name == "15m RE")
    nodes, segments = build_from_extents(element_tube_extents(el), "rigid")
    seg = next(s for s in segments if s.x0 == pytest.approx(2.25) and s.x1 == pytest.approx(2.35))
    i20 = tube_inertia(0.020, 0.0015)
    i16 = tube_inertia(0.016, 0.0015)
    i12 = tube_inertia(0.012, 0.001)
    assert seg.inertia == pytest.approx(i20 + i16 + i12)
    assert seg.outer_d == pytest.approx(0.020, abs=1e-6)
    assert seg.is_overlap


def test_slipped_overlap_uses_outer_tube_only():
    stm = read_stm(DATA)
    el = next(e for e in stm.elements if e.name == "15m RE")
    nodes, segments = build_from_extents(element_tube_extents(el), "slipped")
    seg = next(s for s in segments if s.x0 == pytest.approx(2.25) and s.x1 == pytest.approx(2.35))
    assert seg.inertia == pytest.approx(max(tube_inertia(d, t)
                                            for d, t in ((0.020, 0.0015), (0.016, 0.0015),
                                                         (0.012, 0.001))))


def test_single_tube_double_cantilever_golden():
    # one uniform 12mm tube, span 2 m, centre fixed: M = q*L_half^2/2 per side,
    # delta_tip = q*L_half^4/(8 E I)
    model = build_model_from_geometry([(0.012, 0.001, 0.0, 1.0)], "rigid",
                                      [("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    q = wind_and_weight_load(model.segments[1], 40.0, "al6061-t6")
    assert result.max_moment == pytest.approx(q / 2.0, rel=1e-9)
    e_mod = model.segments[1].e_modulus
    inertia = tube_inertia(0.012, 0.001)
    assert result.max_deflection == pytest.approx(q / (8.0 * e_mod * inertia), rel=1e-6)


def test_survival_wind_speed_uses_exact_geometry():
    from elementload.report import survival_wind_speed
    stm = read_stm(DATA)
    el = next(e for e in stm.elements if e.name == "15m RE")
    model = build_model_from_geometry(element_tube_extents(el), "rigid",
                                      [("center", "fixed")], stm.wind_speed,
                                      el.material_key, yield_stress=None)
    for mode in ("yield", "buckling", "pin_shear"):
        v = survival_wind_speed(model, mode=mode)
        assert 1.0 < v < 500.0
