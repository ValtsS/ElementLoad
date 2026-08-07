import pytest

from elementload.geometry import (
    tube_area,
    tube_inertia,
    tube_mass_per_length,
    tube_section_modulus,
)


def test_tube_area_closed_form():
    # D = 30 mm, t = 2 mm -> d = 26 mm
    a = tube_area(0.030, 0.002)
    assert a == pytest.approx(1.759291886e-4, rel=1e-9)


def test_tube_inertia_closed_form():
    # I = pi/64 * (D^4 - d^4)
    i = tube_inertia(0.030, 0.002)
    assert i == pytest.approx(1.732902508e-8, rel=1e-9)


def test_tube_section_modulus_from_inertia():
    # Z = I / (D/2)
    z = tube_section_modulus(0.030, 0.002)
    assert z == pytest.approx(1.155268338e-6, rel=1e-9)


def test_tube_mass_per_length():
    # 6061-T6 aluminium rho = 2700 kg/m^3
    m = tube_mass_per_length(0.030, 0.002, 2700.0)
    assert m == pytest.approx(0.4750, rel=1e-3)


def test_solid_rod_matches_tube_with_zero_wall():
    # As wall -> D/2 the tube approaches a solid rod of diameter 2*(D/2) ... use
    # t = D/2 - eps to keep positive wall.
    d = 0.020
    t = d / 2 - 1e-9
    a = tube_area(d, t)
    assert a == pytest.approx(3.14159e-4, rel=1e-3)
