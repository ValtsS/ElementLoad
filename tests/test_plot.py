import os

import pytest

from elementload.plot import (
    QUANTITIES,
    _DISPLAY,
    _sample_quantity,
    plot_element,
    plot_es_pair,
)
from elementload.report import build_model, analyze


def test_sample_quantity_all_quantities():
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    for q in QUANTITIES:
        pts = _sample_quantity(model, result, q)
        assert len(pts) > 10
        xs = [p[0] for p in pts]
        assert min(xs) < 0.0 < max(xs)
        assert all(v >= 0.0 for _, v in pts)


def test_plot_display_units_match_labels():
    # stress samples are Pa -> MPa on the colorbar; deflection m -> mm
    assert _DISPLAY["stress"] == pytest.approx(1e-6)
    assert _DISPLAY["deflection"] == pytest.approx(1e3)
    assert _DISPLAY["moment"] == 1.0


def test_plot_element_writes_file(tmp_path):
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    out = tmp_path / "element.png"
    plot_element(model, result, str(out), quantity="stress", title="test")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_element_deformed_overlay(tmp_path):
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    out = tmp_path / "deflected.png"
    plot_element(model, result, str(out), quantity="deflection", deformed=True)
    assert out.exists()


def test_plot_element_rejects_unknown_quantity(tmp_path):
    tubes = [(0.030, 0.002, 0.6)]
    model = build_model(tubes, overlap=0.0, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    with pytest.raises(ValueError):
        plot_element(model, result, str(tmp_path / "x.png"), quantity="nope")


def test_plot_es_pair_writes_file(tmp_path):
    # a sag support carries self-weight only; the pair plot draws both models
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6)]
    kw = dict(overlap=0.05, joint="rigid")
    pin_m = build_model(tubes, supports=[("center", "fixed"), (0.25, "pin"), (-0.25, "pin")],
                        wind_speed=40.0, **kw)
    sag_m = build_model(tubes, supports=[("center", "fixed"), (0.25, "sag"), (-0.25, "sag")],
                        wind_speed=40.0, **kw)
    out = tmp_path / "pair.png"
    plot_es_pair((pin_m, analyze(pin_m)), (sag_m, analyze(sag_m)), str(out),
                 quantity="deflection", title="test", wind_speed=40.0)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_es_pair_shared_scale_uses_both_models():
    # the shared colorbar must span both models (sag deflection >> pin)
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6)]
    kw = dict(overlap=0.05, joint="rigid")
    pin_m = build_model(tubes, supports=[("center", "fixed"), (0.25, "pin"), (-0.25, "pin")],
                        wind_speed=40.0, **kw)
    sag_m = build_model(tubes, supports=[("center", "fixed"), (0.25, "sag"), (-0.25, "sag")],
                        wind_speed=40.0, **kw)
    pin_v = [v for _, v in _sample_quantity(pin_m, analyze(pin_m), "deflection")]
    sag_v = [v for _, v in _sample_quantity(sag_m, analyze(sag_m), "deflection")]
    assert max(sag_v) > max(pin_v)
