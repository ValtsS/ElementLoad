import pytest

from elementload.failure import (
    yield_factor,
    local_buckling_stress,
    pin_shear_factor,
    allowables,
)


def test_yield_factor_margin():
    # M = 100 N.m, Z = 1.155e-6 m^3 -> sigma = 86.6 MPa, yield 276 MPa
    factor = yield_factor(100.0, 1.155268338e-6, 276.0e6)
    assert factor == pytest.approx(100.0 / (1.155268338e-6 * 276.0e6))


def test_yield_factor_exact():
    # stress exactly at yield -> factor 1.0
    factor = yield_factor(276.0e6 * 1.0e-6, 1.0e-6, 276.0e6)
    assert factor == pytest.approx(1.0)


def test_local_buckling_stress_thin_tube():
    # thin wall -> low critical stress (crippling)
    sigma_cr = local_buckling_stress(0.030, 0.002, 69.0e9)
    # thicker tube -> higher critical stress
    sigma_thick = local_buckling_stress(0.030, 0.004, 69.0e9)
    assert sigma_cr < sigma_thick


def test_local_buckling_stress_scale():
    # for a given tube, critical stress scales with E
    sigma_alu = local_buckling_stress(0.030, 0.002, 69.0e9)
    sigma_steel = local_buckling_stress(0.030, 0.002, 200.0e9)
    assert sigma_steel == pytest.approx(sigma_alu * 200.0 / 69.0, rel=1e-9)


def test_pin_shear_factor():
    # axial load F on n pins, each with cross-section area pi*d^2/4
    F = 500.0
    d = 0.004
    n = 2
    tau_u = 190.0e6  # 6061-T6 shear strength approx
    factor = pin_shear_factor(F, d, n, tau_u)
    area = n * 3.14159 * d**2 / 4
    assert factor == pytest.approx(F / (area * tau_u), rel=1e-3)


def test_allowables_returns_dict():
    # material allowables convenience
    a = allowables("al6061-t6")
    assert "yield" in a
    assert "shear" in a
    assert a["yield"] == pytest.approx(276.0e6, rel=1e-3)
