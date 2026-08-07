import math

import pytest

from elementload.report import (
    build_model,
    analyze,
    survival_wind_speed,
    natural_frequencies,
    vortex_crosswind_response,
    along_wind_dynamic_factor,
    max_wind_stress,
    fatigue_report,
)
from elementload.dynamics import critical_wind_speed, scruton_number, crosswind_amplitude
from elementload.geometry import tube_inertia, tube_section_modulus, tube_area

E = 69.0e9
G = 9.81
AL_DENSITY = 2700.0

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


def test_analyze_utilization_modes_present():
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    result = analyze(model)
    assert set(result.utilization) == {"yield", "buckling", "pin_shear"}


def test_analyze_yield_utilization_equals_stress_ratio():
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    result = analyze(model)
    factor, loc = result.utilization["yield"]
    assert factor == pytest.approx(result.max_stress / 276.0e6, rel=1e-9)
    assert loc == pytest.approx(0.0, abs=1e-6)


def test_survival_wind_speed_buckling_consistent():
    # thin wall tube: at the buckling survival speed the buckling factor is ~1.0
    D, t, length = 0.030, 0.001, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    v = survival_wind_speed(model, mode="buckling")
    assert 10.0 < v < 500.0
    model2 = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=v)
    factor, _ = analyze(model2).utilization["buckling"]
    assert factor == pytest.approx(1.0, rel=1e-3)


def test_survival_wind_speed_pin_shear_consistent():
    # telescoping model with pins: at the pin-shear survival speed the factor is ~1.0
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=150.0,
                        pin_diameter=0.001, n_pins=1)
    v = survival_wind_speed(model, mode="pin_shear")
    assert 1.0 < v < 500.0
    model2 = build_model(tubes, overlap=0.05, joint="rigid",
                         supports=[("center", "fixed")], wind_speed=v,
                         pin_diameter=0.001, n_pins=1)
    factor, _ = analyze(model2).utilization["pin_shear"]
    assert factor == pytest.approx(1.0, rel=1e-3)


def test_limiting_mode_is_min_survival():
    # the limiting survival speed across modes is the minimum of the per-mode values
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=150.0)
    v_yield = survival_wind_speed(model, mode="yield")
    v_buckle = survival_wind_speed(model, mode="buckling")
    v_pin = survival_wind_speed(model, mode="pin_shear")
    assert min(v_yield, v_buckle, v_pin) == v_yield  # yield is the critical mode
    assert v_buckle == pytest.approx(500.0)  # buckling not critical for 2mm wall
    assert v_pin == pytest.approx(500.0)  # no pins (single tube) -> not critical


def test_self_weight_sag_matches_cantilever_formula():
    # single tube clamped at center: each half is a cantilever, tip sag = q*l^4/(8*E*I)
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    q = tube_area(D, t) * AL_DENSITY * G
    l = length / 2  # half span
    expected = q * l**4 / (8 * E * tube_inertia(D, t))
    assert result.max_sag == pytest.approx(expected, rel=1e-6)


def test_self_weight_sag_independent_of_wind():
    # sag at zero wind equals sag at 40 m/s (self-weight only)
    tubes = [(0.030, 0.002, 2.0)]
    m0 = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=0.0)
    m40 = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    assert analyze(m0).max_sag == pytest.approx(analyze(m40).max_sag, rel=1e-9)


def test_sag_support_does_not_restrain_wind():
    # a sag (rope) support carries self-weight only: wind stress matches the
    # center-only model, while the self-weight sag is reduced
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6)]
    kw = dict(overlap=0.05, joint="rigid")
    center_only = build_model(tubes, supports=[("center", "fixed")], wind_speed=40.0, **kw)
    sag = build_model(tubes, supports=[("center", "fixed"), (0.25, "sag"), (-0.25, "sag")],
                      wind_speed=40.0, **kw)
    pin = build_model(tubes, supports=[("center", "fixed"), (0.25, "pin"), (-0.25, "pin")],
                      wind_speed=40.0, **kw)
    # wind-only peak stress: sag and center-only agree (rope gives no wind restraint)
    assert max_wind_stress(sag) == pytest.approx(max_wind_stress(center_only), rel=1e-9)
    # ...while a rigid pin does restrain the wind
    assert max_wind_stress(pin) < max_wind_stress(sag)
    # self-weight sag: the sag support carries it, so sag is reduced below center-only
    assert analyze(sag).max_sag < analyze(center_only).max_sag
    # combined stress (wind + weight, orthogonal) is between the two bounds
    v_center = survival_wind_speed(center_only, mode="yield")
    v_sag = survival_wind_speed(sag, mode="yield")
    v_pin = survival_wind_speed(pin, mode="yield")
    assert v_center < v_sag < v_pin
    assert v_sag > 1.0


def test_natural_frequencies_single_tube_matches_cantilever():
    # single tube clamped at center: each half is a cantilever of length L/2
    D, t, length = 0.030, 0.002, 2.0
    tubes = [(D, t, length)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    m = tube_area(D, t) * AL_DENSITY
    l = length / 2
    expected = (1.8751**2 / (2 * 3.14159)) * (E * tube_inertia(D, t) / (m * l**4)) ** 0.5
    f = natural_frequencies(model, n_modes=1)
    # single segment per half is a coarse mesh; 1-element cantilever is ~0.5% stiff
    assert f[0] == pytest.approx(expected, rel=1e-2)


def test_natural_frequencies_overlap_uses_combined_mass():
    # telescoping half-span (0.9 m) is shorter than single-tube half-span (1.0 m),
    # so its fundamental is higher
    single = [(0.030, 0.002, 2.0)]
    m1 = build_model(single, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    telescoping = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6)]
    m2 = build_model(telescoping, overlap=0.05, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    f1 = natural_frequencies(m1, n_modes=1)[0]
    f2 = natural_frequencies(m2, n_modes=1)[0]
    assert f2 > f1
    assert f2 > 0.0


def test_vortex_crosswind_response_fields():
    # single 30 mm tube clamped at center -> per-mode crosswind response at lock-in
    tubes = [(0.030, 0.002, 2.0)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    resp = vortex_crosswind_response(model, damping=0.01, n_modes=3)
    assert len(resp) == 3
    r = resp[0]
    f = natural_frequencies(model, n_modes=1)[0]
    assert r["frequency"] == pytest.approx(f, rel=1e-9)
    assert r["v_crit"] == pytest.approx(critical_wind_speed(f, 0.030), rel=1e-9)
    lo, hi = r["lock_in"]
    assert lo < r["v_crit"] < hi
    assert r["scruton"] == pytest.approx(
        scruton_number(r["mass_per_length"], 0.01, 0.030), rel=1e-9)
    # amplitude smaller than the diameter, dimensionless amplitude from E.7
    assert 0.0 < r["y_max"] < 0.030
    assert r["stress_range"] > 0.0
    # at lock-in the design amplitude scales 1/Sc; more damping -> smaller response
    resp2 = vortex_crosswind_response(model, damping=0.02, n_modes=3)
    assert resp2[0]["y_max"] == pytest.approx(r["y_max"] / 2.0, rel=1e-9)


def test_along_wind_dynamic_factor():
    tubes = [(0.030, 0.002, 2.0)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    r = along_wind_dynamic_factor(model, damping=0.01, height=10.0, z_0=0.05)
    assert r["frequency"] == pytest.approx(natural_frequencies(model, n_modes=1)[0], rel=1e-9)
    assert r["delta_s"] == pytest.approx(2 * math.pi * 0.01, rel=1e-12)
    assert r["delta"] == pytest.approx(r["delta_s"] + r["delta_a"], rel=1e-12)
    assert r["c_d"] == pytest.approx(r["c_s_cd"] / r["c_s"], rel=1e-12)
    assert 0.0 < r["c_s"] < 1.0      # size factor reduces the load
    assert r["c_d"] > 1.0            # dynamic factor amplifies it
    assert 0.8 < r["c_s_cd"] < 1.5
    # higher damping -> less resonance -> smaller dynamic factor
    r2 = along_wind_dynamic_factor(model, damping=0.03, height=10.0, z_0=0.05)
    assert r2["resonance"] < r["resonance"]
    assert r2["c_s_cd"] < r["c_s_cd"]


def test_fatigue_report_sources_and_consistency():
    tubes = [(0.030, 0.002, 2.0)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    fat = fatigue_report(model, damping=0.01, height=10.0, z_0=0.05)
    names = [s["name"] for s in fat["sources"]]
    assert "VIV mode 1 lock-in" in names
    assert "gust buffeting" in names
    # total damage rate = sum of the per-source rates
    assert fat["damage_per_hour"] == pytest.approx(
        sum(s["damage_per_hour"] for s in fat["sources"]), rel=1e-12)
    # this light element vibrates below the 96.5 MPa endurance limit -> unlimited life
    assert fat["life_hours"] == float("inf")
    assert all(s["damage_per_hour"] == 0.0 for s in fat["sources"])


def test_fatigue_report_gust_range_matches_formula():
    # gust range = 2 * sigma_mean * 2 k_p I_v (sqrt(B^2+R^2) - sqrt(B^2))/(1+7 I_v)
    tubes = [(0.030, 0.002, 2.0)]
    model = build_model(tubes, overlap=0.0, joint="rigid", supports=[("center", "fixed")], wind_speed=40.0)
    fat = fatigue_report(model, damping=0.01, height=10.0, z_0=0.05)
    daf = along_wind_dynamic_factor(model, damping=0.01, height=10.0, z_0=0.05)
    gust = next(s for s in fat["sources"] if s["name"] == "gust buffeting")
    iv, k_p = daf["turbulence_intensity"], daf["peak_factor"]
    b2, r2 = daf["background"], daf["resonance"]
    expected = 2.0 * max_wind_stress(model) * 2.0 * k_p * iv * (
        (b2 + r2) ** 0.5 - b2 ** 0.5) / (1.0 + 7.0 * iv)
    assert gust["stress_range"] == pytest.approx(expected, rel=1e-12)
    # the VIV source range comes straight from the crosswind response
    viv = next(s for s in fat["sources"] if s["name"].startswith("VIV mode 1"))
    resp = vortex_crosswind_response(model, damping=0.01, n_modes=3)[0]
    assert viv["stress_range"] == pytest.approx(resp["stress_range"], rel=1e-12)
