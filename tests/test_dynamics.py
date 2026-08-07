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
    assert f1 == pytest.approx(0.2 * 10.0 / 0.030, rel=1e-9)


def test_critical_wind_speed_lockin():
    # V_crit = f_i * D / St; at V_crit shedding frequency equals the natural frequency
    f = 5.0
    D = 0.030
    v = critical_wind_speed(f, D)
    assert v == pytest.approx(f * D / 0.2, rel=1e-9)
    assert shedding_frequency(v, D) == pytest.approx(f, rel=1e-9)


def test_scruton_number():
    # Sc = 4*pi*m*zeta/(rho*D^2); small Sc -> strong lock-in response
    m = 0.5
    zeta = 0.01
    D = 0.030
    sc = scruton_number(m, zeta, D)
    assert sc == pytest.approx(4 * math.pi * m * zeta / (1.225 * D**2), rel=1e-9)
