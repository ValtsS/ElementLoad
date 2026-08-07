import pytest

from elementload.wind import (
    drag_per_unit_length,
    drag_coefficient,
    reynolds_number,
)


def test_drag_per_unit_length_basic():
    # q = 0.5 * rho * V^2 * Cd * D
    # V = 30 m/s, Cd = 1.2, D = 30 mm, rho_air = 1.225
    q = drag_per_unit_length(30.0, 1.2, 0.030, 1.225)
    assert q == pytest.approx(19.845, rel=1e-9)


def test_drag_scales_with_velocity_squared():
    q1 = drag_per_unit_length(20.0, 1.2, 0.030, 1.225)
    q2 = drag_per_unit_length(40.0, 1.2, 0.030, 1.225)
    assert q2 == pytest.approx(4 * q1, rel=1e-9)


def test_drag_scales_with_diameter():
    q1 = drag_per_unit_length(30.0, 1.2, 0.020, 1.225)
    q2 = drag_per_unit_length(30.0, 1.2, 0.040, 1.225)
    assert q2 == pytest.approx(2 * q1, rel=1e-9)


def test_reynolds_number():
    # Re = V*D/nu, nu_air ~ 1.5e-5 m^2/s at 15 C
    re = reynolds_number(30.0, 0.030, 1.5e-5)
    assert re == pytest.approx(6.0e4, rel=1e-9)


def test_drag_coefficient_subcritical_plateau():
    # EN 1991-1-4 Fig 7.28 smooth cylinder: Cf,0 ~ 1.2 across the subcritical plateau
    for re in (1.0e4, 5.0e4, 1.0e5, 2.0e5):
        assert drag_coefficient(re) == pytest.approx(1.2, abs=0.02)


def test_drag_coefficient_critical_drop():
    # Cd drops through the drag crisis (critical range) near Re ~ 4e5
    low = drag_coefficient(1.0e4)
    assert drag_coefficient(4.0e5) < 0.6 * low
    assert drag_coefficient(4.0e5) > 0.2 * low
    assert drag_coefficient(5.0e5) < drag_coefficient(4.0e5)


def test_drag_coefficient_dlubal_verification_anchor():
    # Dlubal Eurocode verification example VE0308: Cf,0 = 0.55 at Re = 2e6
    assert drag_coefficient(2.0e6) == pytest.approx(0.55, abs=0.02)


def test_drag_coefficient_supercritical_recovery():
    # Cd partially recovers toward the transcritical regime
    assert drag_coefficient(1.0e7) > drag_coefficient(4.0e5)
    assert drag_coefficient(1.0e7) <= 1.0

