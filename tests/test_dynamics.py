import math

import pytest

from elementload.solver import Node, Element, Support, assemble_stiffness
from elementload.dynamics import (
    assemble_mass,
    natural_frequencies,
    mass_per_length,
    shedding_frequency,
    critical_wind_speed,
    scruton_number,
    lateral_force_coefficient,
    lateral_force_coefficient_base,
    crosswind_amplitude,
    correlation_length_factor_cantilever,
    lock_in_window,
)

E = 69.0e9


def test_mass_per_length_from_tube():
    m = mass_per_length(0.030, 0.002, 2700.0)
    expected = math.pi / 4 * (0.030**2 - 0.026**2) * 2700.0
    assert m == pytest.approx(expected, rel=1e-9)


def test_assemble_mass_total_mass():
    # rigid-body translation mode: sum of consistent mass matrix entries = m*L
    L = 2.0
    m = 1.0
    nodes = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, 1.0e-8)]
    M = assemble_mass(nodes, elements, [m])
    ones = [1.0, 0.0, 1.0, 0.0]  # unit translation, zero rotation
    assert ones @ M @ ones == pytest.approx(m * L, rel=1e-9)


def test_natural_frequencies_uniform_cantilever_first_mode():
    # f1 = (1.8751^2/(2*pi)) * sqrt(EI/(m*L^4)) for a cantilever
    L = 2.0
    m = 1.0
    I = 1.0e-8
    n = 10
    nodes = [Node(x) for x in (i * L / n for i in range(n + 1))]
    elements = [Element(i, i + 1, E, I) for i in range(n)]
    supports = [Support(0, "fixed")]
    f = natural_frequencies(nodes, elements, supports, [m] * n, n_modes=3)
    expected = (1.8751**2 / (2 * math.pi)) * math.sqrt(E * I / (m * L**4))
    assert f[0] == pytest.approx(expected, rel=1e-3)


def test_natural_frequencies_cantilever_mode_ratios():
    # (beta_n * L)^2 = 1.8751, 4.6941, 7.8548 for a cantilever
    L = 2.0
    m = 1.0
    I = 1.0e-8
    n = 20
    nodes = [Node(x) for x in (i * L / n for i in range(n + 1))]
    elements = [Element(i, i + 1, E, I) for i in range(n)]
    supports = [Support(0, "fixed")]
    f = natural_frequencies(nodes, elements, supports, [m] * n, n_modes=3)
    base = 1.8751**2
    ratios = [(1.8751**2 / base), (4.6941**2 / base), (7.8548**2 / base)]
    for got, exp in zip(f, [r * f[0] for r in ratios]):
        assert got == pytest.approx(exp, rel=1e-3)


def test_natural_frequencies_double_cantilever():
    # clamped at center = two cantilevers of half length; f scales as 1/(L/2)^2 = 4x
    L = 4.0
    m = 1.0
    I = 1.0e-8
    n = 20
    xs = [i * L / n for i in range(n + 1)]
    nodes = [Node(x) for x in xs]
    elements = [Element(i, i + 1, E, I) for i in range(n)]
    center = min(range(n + 1), key=lambda k: abs(xs[k] - L / 2))
    supports = [Support(center, "fixed")]
    f = natural_frequencies(nodes, elements, supports, [m] * n, n_modes=1)
    expected = (1.8751**2 / (2 * math.pi)) * math.sqrt(E * I / (m * (L / 2) ** 4))
    assert f[0] == pytest.approx(expected, rel=1e-3)


def test_shedding_frequency_scales_with_wind():
    f1 = shedding_frequency(10.0, 0.030)
    f2 = shedding_frequency(20.0, 0.030)
    assert f2 == pytest.approx(2 * f1, rel=1e-9)
    assert f1 == pytest.approx(0.18 * 10.0 / 0.030, rel=1e-9)


def test_critical_wind_speed_lockin():
    # V_crit = f_i * D / St; at V_crit shedding frequency equals the natural frequency
    f = 5.0
    D = 0.030
    v = critical_wind_speed(f, D)
    assert v == pytest.approx(f * D / 0.18, rel=1e-9)
    assert shedding_frequency(v, D) == pytest.approx(f, rel=1e-9)


def test_scruton_number():
    # Sc = 4*pi*m*zeta/(rho*D^2); small Sc -> strong lock-in response
    m = 0.5
    zeta = 0.01
    D = 0.030
    sc = scruton_number(m, zeta, D)
    assert sc == pytest.approx(4 * math.pi * m * zeta / (1.225 * D**2), rel=1e-9)


def test_lateral_force_coefficient_base_plateaus():
    # EN 1991-1-4 Fig. E.2: subcritical plateau, log-Re transition, transcritical plateau
    assert lateral_force_coefficient_base(1.0e4) == pytest.approx(0.7)
    assert lateral_force_coefficient_base(1.0e7) == pytest.approx(0.35)
    # log-midpoint between 2e5 and 1e6 -> halfway between 0.7 and 0.35
    mid_re = math.sqrt(2.0e5 * 1.0e6)
    assert lateral_force_coefficient_base(mid_re) == pytest.approx(0.525)


def test_lateral_force_coefficient_table_e3():
    c0 = 0.7
    # lock-in (critical wind ~ mean wind at the correlation centre)
    assert lateral_force_coefficient(1.0, c0) == pytest.approx(0.75 * c0)
    # well below resonance: linear ramp [3/2 - 4/3 * ratio] * c0
    assert lateral_force_coefficient(0.5, c0) == pytest.approx((1.5 - 4 / 3 * 0.5) * c0)
    # critical wind above 1.25 * mean wind: effect need not be investigated
    assert lateral_force_coefficient(1.25, c0) == 0.0
    assert lateral_force_coefficient(2.0, c0) == 0.0


def test_crosswind_amplitude_e7_closed_form():
    # y_max/b = (1/St^2)(1/Sc) c_lat K_w K, EN 1991-1-4 E.7
    b = 0.030
    st = 0.18
    sc = 10.0
    c_lat = 0.525
    k_w = 0.5
    k = 0.13
    y = crosswind_amplitude(b, sc, c_lat, k_w, k, strouhal=st)
    expected = b * (1 / st**2) * (1 / sc) * c_lat * k_w * k
    assert y == pytest.approx(expected, rel=1e-9)
    # dimensionless check: y/b = (1/St^2)(1/Sc) c_lat K_w K
    assert y / b == pytest.approx((1 / st**2) * (1 / sc) * c_lat * k_w * k, rel=1e-9)


def test_correlation_length_factor_cantilever_golden():
    # K_w = int_{tip-L}^{tip} phi ds / int_0^h phi ds for phi=(s/h)^2 -> 1-(1-L/h)^3
    h = 1.0
    b = 0.030
    L = 6.0 * b
    kw = correlation_length_factor_cantilever(h, b)
    assert kw == pytest.approx(1 - (1 - L / h) ** 3, rel=1e-12)
    # matches direct numerical integration of the mode shape
    n = 100000
    s = [i * h / n for i in range(n + 1)]
    tip = [i * h / n for i in range(n + 1) if i * h / n >= h - L]
    num = (sum((x / h) ** 2 for x in tip) - (tip[0] / h) ** 2) * (h / n)
    den = (sum((x / h) ** 2 for x in s) - 0.0) * (h / n)
    assert kw == pytest.approx(num / den, rel=1e-3)
    # capped at 0.6 per E.8
    assert correlation_length_factor_cantilever(h, b, correlation_length=h) == pytest.approx(0.6)


def test_lock_in_window():
    # Table E.3 effective range: 0.83 < V_crit/V_m < 1.25 -> V_m in (V_crit/1.25, V_crit/0.83)
    v = 5.0
    lo, hi = lock_in_window(v)
    assert lo == pytest.approx(v / 1.25)
    assert hi == pytest.approx(v / 0.83)
