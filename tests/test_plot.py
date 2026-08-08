import os

import pytest

from elementload.plot import QUANTITIES, _sample_quantity, plot_element
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
