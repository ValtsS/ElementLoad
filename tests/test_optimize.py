import os

import pytest

from elementload.optimize import (
    element_mass,
    fits_inside,
    parse_catalog_lines,
    pick_tube,
    read_catalog,
    size_from_extents,
    snap_lengths,
)
from elementload.report import build_model_from_geometry, survival_wind_speed
from elementload.stm import element_tube_extents, read_stm

CATALOG = [
    (0.012, 0.001),
    (0.016, 0.0015),
    (0.020, 0.0015),
    (0.020, 0.002),
    (0.025, 0.002),
    (0.030, 0.002),
    (0.035, 0.002),
    (0.040, 0.002),
    (0.045, 0.002),
    (0.050, 0.002),
    (0.060, 0.002),
]
YIELD = 190.0e6


def _path(name):
    return os.path.join(os.path.dirname(__file__), "..", "data", name)


def test_read_catalog_parses_stock_sizes():
    tubes = read_catalog(_path("tube_catalog.csv"))
    assert tubes[0] == (0.012, 0.001)
    assert tubes[-1] == (0.060, 0.002)
    assert len(tubes) == 11


def test_read_catalog_sorted_by_diameter():
    tubes = read_catalog(_path("tube_catalog.csv"))
    d = [t[0] for t in tubes]
    assert d == sorted(d)


def test_read_catalog_detects_duplicates():
    with pytest.raises(ValueError):
        parse_catalog_lines(["20.0, 2.0", "20.0, 2.0"])


def test_pick_tube_selects_lightest_adequate_tube():
    assert pick_tube(250.0, CATALOG, YIELD) == (0.035, 0.002)
    assert pick_tube(100.0, CATALOG, YIELD) == (0.025, 0.002)
    assert pick_tube(20.0, CATALOG, YIELD) == (0.016, 0.0015)


def test_pick_tube_none_for_impossible_load():
    assert pick_tube(1.0e9, CATALOG, YIELD) is None


def test_pick_tube_respects_buckling_limit():
    # 25x2.0 yields at ~146 Nm; 150 Nm needs 30x2.0 (buckling 2.96 GPa is not
    # binding for these tubes — yield governs in every case here)
    assert pick_tube(150.0, CATALOG, YIELD) == (0.030, 0.002)
    assert pick_tube(150.0, CATALOG, YIELD)[0] >= 0.025


def test_fits_inside_telescoping_clearance():
    assert fits_inside((0.045, 0.002), (0.035, 0.002), 1.0e-3)
    assert not fits_inside((0.040, 0.002), (0.045, 0.002), 1.0e-3)
    assert not fits_inside((0.045, 0.002), (0.045, 0.002), 1.0e-3)
    assert fits_inside((0.045, 0.002), (0.040, 0.002), 1.0e-3)


def test_size_from_extents_keeps_feasible_element():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    opt = size_from_extents(tubes, "rigid", [("center", "fixed")],
                            stm.wind_speed, el.material_key, el.yield_stress,
                            catalog=CATALOG)
    assert len(opt) == len(tubes)
    for i, t in enumerate(opt):
        assert (t.outer_d, t.wall) in CATALOG
        if i:
            assert fits_inside(opt[i - 1], t, 1.0e-3)


def test_size_from_extents_matches_stock_at_design_wind():
    # The stock a1.stm 20m RE is near-optimal: the root must host the 40mm tube,
    # so it cannot shrink. But the 20mm section must be 20x1.5 (bore 17mm) to
    # telescope over the 16mm tube with 1mm clearance — stock's 20x2.0 (bore
    # 16mm) is a zero-clearance fit. The optimizer finds the lighter 20x1.5.
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    opt = size_from_extents(tubes, "rigid", [("center", "fixed")],
                            stm.wind_speed, el.material_key, el.yield_stress,
                            catalog=CATALOG)
    assert [(round(t.outer_d, 6), round(t.wall, 6)) for t in opt] == \
        [(0.045, 0.002), (0.04, 0.002), (0.035, 0.002), (0.03, 0.002),
         (0.025, 0.002), (0.02, 0.0015), (0.016, 0.0015), (0.012, 0.001)]


def test_optimized_element_survives_target_wind_and_is_lighter():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    opt = size_from_extents(tubes, "rigid", [("center", "fixed")],
                            stm.wind_speed, el.material_key, el.yield_stress,
                            catalog=CATALOG)
    model = build_model_from_geometry(opt, "rigid", [("center", "fixed")],
                                      stm.wind_speed, el.material_key,
                                      yield_stress=el.yield_stress)
    survival = survival_wind_speed(model, "yield")
    assert survival >= stm.wind_speed
    assert element_mass(opt) <= element_mass(tubes)


def test_size_from_extents_raises_when_chain_infeasible():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    with pytest.raises(ValueError, match="bore"):
        size_from_extents(tubes, "rigid", [("center", "fixed")], 55.0,
                          el.material_key, el.yield_stress, catalog=CATALOG)


def test_size_from_extents_raises_when_catalog_too_small_for_chain():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    small = [t for t in CATALOG if t[0] <= 0.040]
    with pytest.raises(ValueError, match="bore"):
        size_from_extents(tubes, "rigid", [("center", "fixed")], stm.wind_speed,
                          el.material_key, el.yield_stress, catalog=small)


def test_snap_lengths_rounds_to_step():
    from elementload.assembly import TubeExtent
    tubes = [TubeExtent(0.045, 0.002, 0.0, 0.55),
             TubeExtent(0.040, 0.002, 0.40, 1.15),
             TubeExtent(0.012, 0.001, 0.45, 1.60)]
    snapped = snap_lengths(tubes, 0.005)
    for t in snapped:
        length = t.inner_end - t.inner_start
        assert abs(round(length / 0.005) - length / 0.005) < 1e-9
    # inner joints stay fixed; the tip absorbs the rounding
    assert snapped[0].inner_start == 0.0
    assert snapped[0].inner_end == 0.55
    assert snapped[1].inner_start == 0.40
    assert snapped[2].inner_end >= 1.60


def test_size_from_extents_with_length_step_survives():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    opt = size_from_extents(tubes, "rigid", [("center", "fixed")],
                            stm.wind_speed, el.material_key, el.yield_stress,
                            catalog=CATALOG, length_step=0.005)
    model = build_model_from_geometry(opt, "rigid", [("center", "fixed")],
                                      stm.wind_speed, el.material_key,
                                      yield_stress=el.yield_stress)
    assert survival_wind_speed(model, "yield") >= stm.wind_speed
