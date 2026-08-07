import pytest

from elementload.assembly import Tube, build_telescoping, Segment
from elementload.geometry import tube_inertia


def test_two_tube_mesh_nodes():
    # Center tube 30mm, tip tube 25mm, 150mm overlap -> symmetric mesh
    tubes = [Tube(0.030, 0.002, 1.0), Tube(0.025, 0.002, 0.8)]
    nodes, segments = build_telescoping(tubes, overlap=0.15, joint="rigid")
    assert nodes == pytest.approx([-1.15, -0.5, -0.35, 0.0, 0.35, 0.5, 1.15])
    assert len(segments) == 6


def test_rigid_overlap_uses_combined_inertia():
    tubes = [Tube(0.030, 0.002, 1.0), Tube(0.025, 0.002, 0.8)]
    nodes, segments = build_telescoping(tubes, overlap=0.15, joint="rigid")
    i0 = tube_inertia(0.030, 0.002)
    i1 = tube_inertia(0.025, 0.002)
    # segment index 1 is the right-side overlap
    assert segments[1].inertia == pytest.approx(i0 + i1)
    assert segments[1].outer_d == pytest.approx(0.030)


def test_slipped_overlap_uses_outer_tube_only():
    tubes = [Tube(0.030, 0.002, 1.0), Tube(0.025, 0.002, 0.8)]
    nodes, segments = build_telescoping(tubes, overlap=0.15, joint="slipped")
    i0 = tube_inertia(0.030, 0.002)
    assert segments[1].inertia == pytest.approx(i0)


def test_single_tube_mesh():
    tubes = [Tube(0.030, 0.002, 1.0)]
    nodes, segments = build_telescoping(tubes, overlap=0.0, joint="rigid")
    assert nodes == pytest.approx([-0.5, 0.0, 0.5])
    assert len(segments) == 2
    assert segments[0].outer_d == pytest.approx(0.030)


def test_three_tube_tip_diameter_chain():
    tubes = [
        Tube(0.030, 0.002, 1.0),
        Tube(0.025, 0.002, 0.7),
        Tube(0.020, 0.002, 0.6),
    ]
    nodes, segments = build_telescoping(tubes, overlap=0.1, joint="rigid")
    # last segment (rightmost) is the 20mm tip
    assert segments[-1].outer_d == pytest.approx(0.020)
    # center-adjacent segments are 30mm
    assert segments[4].outer_d == pytest.approx(0.030)
