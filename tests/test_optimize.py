import os

import pytest

from elementload.optimize import (
    _size_telescoped,
    element_mass,
    fits_inside,
    geom_exposed,
    layout_from_exposed,
    parse_catalog_lines,
    pick_tube,
    read_catalog,
    search_layouts,
    size_from_extents,
    snap_lengths,
    tube_mass,
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


def test_geom_exposed_normalizes_to_half_length():
    L = 5.705
    for n in range(3, 8):
        for ratio in (0.6, 1.0, 1.6):
            xs = geom_exposed(n, ratio, L)
            assert xs is not None
            assert sum(xs) == pytest.approx(L, rel=1e-9)
            assert all(x > 0 for x in xs)


def test_geom_exposed_sleeve_sets_first_exposed():
    L = 5.705
    xs = geom_exposed(6, 1.2, L, first=0.15)
    assert xs[0] == pytest.approx(0.15)
    assert sum(xs) == pytest.approx(L, rel=1e-9)


def test_layout_from_exposed_builds_overlapping_extents():
    L = 5.705
    xs = geom_exposed(6, 1.2, L, first=0.15)
    tubes = layout_from_exposed(xs, 0.10, L)
    assert tubes is not None
    assert tubes[0].inner_start == pytest.approx(0.0)
    assert tubes[0].inner_end == pytest.approx(0.15)
    # consecutive tubes overlap by 0.10 m
    assert tubes[1].inner_start == pytest.approx(tubes[0].inner_end - 0.10)
    assert tubes[-1].inner_end == pytest.approx(L)


def test_layout_from_exposed_rejects_degenerate():
    assert layout_from_exposed([1.0, -0.5], 0.1, 1.5) is None
    # overlap longer than the root tube makes the next start negative
    assert layout_from_exposed([0.5, 0.5], 0.6, 1.0) is None
    # exposed lengths must sum to the half length
    assert layout_from_exposed([0.5, 0.5], 0.1, 1.3) is None


def test_search_layouts_finds_lighter_than_stock():
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    results = search_layouts(
        tubes, "rigid", [("center", "fixed")], stm.wind_speed,
        el.material_key, None, catalog=CATALOG,
        n_range=(6, 6), overlaps=(0.10,), ratios=(1.2,),
        sleeves=(0.15,), target_winds=[stm.wind_speed], workers=1)
    assert results
    assert all(r["survival"] >= stm.wind_speed for r in results)
    assert min(r["mass"] for r in results) < element_mass(tubes)
    # the known lightest-at-design layout
    lightest = min(results, key=lambda r: r["mass"])
    assert [int(d) for d in lightest["chain"].split(";")] == [40, 35, 30, 25, 20, 12]


def test_search_layouts_infeasible_layouts_skipped():
    # a single tube far below the catalog minimum cannot telescope the full span
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    results = search_layouts(
        tubes, "rigid", [("center", "fixed")], stm.wind_speed,
        el.material_key, el.yield_stress, catalog=CATALOG,
        n_range=(1, 1), overlaps=(0.10,), ratios=(0.6,),
        sleeves=(None,), target_winds=[stm.wind_speed], workers=1)
    # a 1-tube element longer than the largest catalog tube cannot be built
    assert results == []


def test_layout_search_root_bounds_survival():
    # the 30mm root cannot reach the 35 m/s the 35mm root reaches
    stm = read_stm(_path("a1.stm"))
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    tubes = element_tube_extents(el)
    small = [t for t in CATALOG if t[0] <= 0.030]
    results = search_layouts(
        tubes, "rigid", [("center", "fixed")], 35.0,
        el.material_key, None, catalog=small,
        n_range=(3, 5), overlaps=(0.10, 0.20), ratios=(0.8, 1.2),
        sleeves=(None, 0.15), target_winds=[30, 35, 40], workers=1)
    assert results
    best = max(results, key=lambda r: r["survival"])
    assert best["survival"] < 31.0
    assert int(best["chain"].split(";")[0]) == 30


def test_pareto_frontier_and_best_per_root():
    from elementload.cli import _best_per_root, _pareto_frontier
    results = [
        {"survival": 30.0, "mass": 3.0, "chain": "30;25;20;12"},
        {"survival": 34.0, "mass": 3.5, "chain": "35;30;25;20;12"},
        # dominated: heavier than the 34 m/s design but no faster
        {"survival": 34.0, "mass": 4.0, "chain": "35;30;25;16"},
        {"survival": 36.8, "mass": 3.97, "chain": "40;35;30;25;20;12"},
    ]
    front = _pareto_frontier(results)
    assert len(front) == 3
    assert all(r["survival"] <= 34.0 for r in front if r["mass"] < 3.97)
    best = _best_per_root(results, 35.0)
    assert best[30.0]["survival"] == 30.0
    assert best[40.0]["chain"].endswith("12")


def test_pareto_frontier_collapses_bisection_noise():
    # two layouts at the same survival (differing only in bisection FP noise)
    # collapse to the lighter one
    from elementload.cli import _pareto_frontier
    results = [
        {"survival": 35.756546703040826, "mass": 3.7529315, "chain": "35;30;25;20;12"},
        {"survival": 35.75654670316001, "mass": 3.8723198, "chain": "35;30;25;20;12"},
        {"survival": 36.8405, "mass": 3.9659, "chain": "40;35;30;25;20;12"},
    ]
    front = _pareto_frontier(results)
    assert len(front) == 2
    assert [r["chain"] for r in front] == ["35;30;25;20;12", "40;35;30;25;20;12"]
    assert front[0]["mass"] == pytest.approx(3.7529315)


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


def _tube_capacity(d, t, yield_stress):
    from elementload.failure import local_buckling_stress
    from elementload.geometry import tube_section_modulus
    z = tube_section_modulus(d, t)
    if z <= 0.0:
        return 0.0
    return min(yield_stress, local_buckling_stress(d, t, 69.0e9)) * z


def _brute_force_chain(moments, catalog, yield_stress, clearance):
    """Exhaustive search for the minimum-mass telescoping chain. Returns the
    minimum mass (or None if infeasible). Position k's bore must host the tube
    at k+1 (tip side) plus clearance."""
    import itertools
    best = None
    for combo in itertools.product(catalog, repeat=len(moments)):
        ok = True
        for k in range(len(combo) - 1):
            d_in, t_in = combo[k]
            d_out = combo[k + 1][0]
            if d_out > (d_in - 2.0 * t_in) - clearance + 1e-9:
                ok = False
                break
        if not ok:
            continue
        mass = 0.0
        for (d, t), m in zip(combo, moments):
            if m > _tube_capacity(d, t, yield_stress):
                ok = False
                break
            mass += tube_mass((d, t), 1.0)
        if not ok:
            continue
        best = mass if best is None else min(best, mass)
    return best


def test_greedy_sizing_matches_brute_force():
    """The tip-to-root greedy is provably optimal for this catalog (mass and
    capacity are monotone in diameter), so it must agree with exhaustive search
    over all telescoping chains. Exercises both feasible and infeasible cases
    and a range of clearances."""
    import random
    rng = random.Random(20240808)
    for _ in range(150):
        n = rng.randint(1, 4)
        moments = [rng.uniform(2.0, 300.0) for _ in range(n)]
        clearance = rng.choice([0.0, 0.0005, 0.001, 0.002])
        try:
            picks = _size_telescoped(moments, CATALOG, YIELD, clearance)
            greedy_mass = sum(tube_mass(p, 1.0) for p in picks)
        except ValueError:
            greedy_mass = None
        brute_mass = _brute_force_chain(moments, CATALOG, YIELD, clearance)
        if greedy_mass is None:
            assert brute_mass is None, (moments, clearance, brute_mass)
        else:
            assert brute_mass is not None, (moments, clearance, greedy_mass)
            assert abs(greedy_mass - brute_mass) < 1e-9, (moments, clearance, greedy_mass, brute_mass)


def test_greedy_sizing_matches_brute_force_real_element():
    """Greedy vs exhaustive search on the 10m RE's actual moment envelope."""
    from elementload.optimize import per_tube_max_moment
    stm = read_stm(_path("mechanical2.stm"))
    el = [e for e in stm.elements if e.name == "10m RE"][0]
    tubes = element_tube_extents(el)
    moments = per_tube_max_moment(tubes, "rigid", [("center", "fixed")],
                                  stm.wind_speed, el.material_key, el.yield_stress)
    picks = _size_telescoped(moments, CATALOG, YIELD, 1.0e-3)
    greedy_mass = sum(tube_mass(p, 1.0) for p in picks)
    brute_mass = _brute_force_chain(moments, CATALOG, YIELD, 1.0e-3)
    assert brute_mass is not None
    assert abs(greedy_mass - brute_mass) < 1e-9


def test_bottom_feasible_chain_is_lightest_telescoping():
    """The bottom of the feasible-chain lattice: the lightest chain that
    telescopes with clearance, built tip-to-root."""
    from elementload.optimize import bottom_feasible_chain
    from elementload.assembly import TubeExtent
    stm = read_stm(_path("mechanical2.stm"))
    el = [e for e in stm.elements if e.name == "10m RE"][0]
    tubes = element_tube_extents(el)
    bottom = bottom_feasible_chain(tubes, CATALOG, 1.0e-3)
    assert [(round(t.outer_d, 6), round(t.wall, 6)) for t in bottom] == \
        [(0.020, 0.0015), (0.016, 0.0015), (0.012, 0.001)]
    for i, t in enumerate(bottom):
        if i:
            assert fits_inside(bottom[i - 1], t, 1.0e-3)


def test_size_from_extents_finds_lightest_fixed_point():
    """The fixed-point iteration is start-dependent (the 10m RE at 37 m/s has
    two self-consistent chains: light (20,16,12) and heavy (25,20,16)). The
    multi-start optimizer must return the lightest feasible one regardless of the
    starting chain."""
    stm = read_stm(_path("mechanical2.stm"))
    el = [e for e in stm.elements if e.name == "10m RE"][0]
    tubes = element_tube_extents(el)
    opt = size_from_extents(tubes, "rigid", [("center", "fixed")],
                            37.0, el.material_key, el.yield_stress,
                            catalog=CATALOG)
    assert [(round(t.outer_d, 6), round(t.wall, 6)) for t in opt] == \
        [(0.020, 0.0015), (0.016, 0.0015), (0.012, 0.001)]
    light_mass = element_mass(opt)
    assert light_mass < element_mass(tubes) or abs(light_mass - element_mass(tubes)) < 1e-9
    # the heavy fixed point exists at this wind speed and must NOT be returned
    assert opt[0].outer_d < 0.025


def test_fixed_point_iteration_is_start_independent_after_multistart():
    """Same light result even when the element is handed in as a heavy uniform
    chain: the multi-start searches from the bottom of the feasible lattice too,
    so a suboptimal starting chain cannot trap the optimizer on the heavy fixed
    point."""
    from elementload.assembly import TubeExtent
    stm = read_stm(_path("mechanical2.stm"))
    el = [e for e in stm.elements if e.name == "10m RE"][0]
    tubes = element_tube_extents(el)
    heavy = [TubeExtent(0.060, 0.002, t.inner_start, t.inner_end) for t in tubes]
    opt = size_from_extents(heavy, "rigid", [("center", "fixed")],
                            37.0, el.material_key, el.yield_stress,
                            catalog=CATALOG)
    assert [(round(t.outer_d, 6), round(t.wall, 6)) for t in opt] == \
        [(0.020, 0.0015), (0.016, 0.0015), (0.012, 0.001)]
