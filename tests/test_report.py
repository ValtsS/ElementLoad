import pytest

from elementload.report import build_model, analyze, survival_wind_speed
from elementload.geometry import tube_inertia, tube_section_modulus

E = 69.0e9


def test_build_model_single_tube_clamped_center():
    # 30mm tube, 2m long, clamped at center -> 2 cantilever halves of 1m
    tubes = [(0.030, 0.002, 2.0)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    assert len(model.nodes) == 3
    assert model.nodes[1].x == pytest.approx(0.0)
    assert len(model.elements) == 2


def test_clamp_moment_matches_qL2_over_8():
    # uniform load q on symmetric clamped beam: M at clamp = q*L^2/8 (L=full span)
    D, t, length = 0.030, 0.002, 2.0
    V = 40.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=V)
    q = model.loads[0].q  # actual distributed load applied (wind + self-weight)
    result = analyze(model)
    assert result.max_moment == pytest.approx(q * length**2 / 8, rel=1e-6)


def test_peak_stress_at_clamp():
    D, t, length = 0.030, 0.002, 2.0
    V = 40.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=V)
    result = analyze(model)
    z = tube_section_modulus(D, t)
    expected = result.max_moment / z
    assert result.max_stress == pytest.approx(expected, rel=1e-9)
    assert result.max_stress_location == pytest.approx(0.0)


def test_survival_wind_speed_yield_plausible():
    # 30mm tube clamped at center: yield-critical survival speed in a sane range
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    v = survival_wind_speed(model, mode="yield")
    assert 50.0 < v < 400.0


def test_survival_wind_speed_zero_margin_matches_analyze():
    # at the survival wind speed the yield factor should be ~1.0
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    v = survival_wind_speed(model, mode="yield")
    model2 = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=v)
    result2 = analyze(model2)
    assert result2.max_stress / 276.0e6 == pytest.approx(1.0, rel=1e-3)
