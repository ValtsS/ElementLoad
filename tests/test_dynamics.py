import math

import pytest

from elementload.solver import (
    Node,
    Element,
    Support,
    DistributedLoad,
    assemble_stiffness,
    solve,
    internal_moment,
)
from elementload.dynamics import (
    assemble_mass,
    natural_frequencies,
    modal_data,
    equivalent_mass,
    vortex_inertia_load,
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


def test_modal_data_normalizes_mode():
    # mode shapes normalized so max transverse displacement = 1 (E.6), tip DOF peaks
    L = 2.0
    m = 1.0
    I = 1.0e-8
    n = 12
    nodes = [Node(x) for x in (i * L / n for i in range(n + 1))]
    elements = [Element(i, i + 1, E, I) for i in range(n)]
    supports = [Support(0, "fixed")]
    freqs, modes = modal_data(nodes, elements, supports, [m] * n, n_modes=3)
    f = natural_frequencies(nodes, elements, supports, [m] * n, n_modes=3)
    assert list(freqs) == pytest.approx(list(f), rel=1e-12)
    for i in range(3):
        w = modes[i][0::2]
        assert abs(w[0]) == pytest.approx(0.0, abs=1e-12)  # clamped
        assert max(abs(v) for v in w) == pytest.approx(1.0, rel=1e-9)
        assert abs(w[-1]) == pytest.approx(1.0, rel=1e-3)  # tip near max displacement


def test_equivalent_mass_analytic_modes():
    # F.14 m_e = sum(m_i int phi^2 ds)/sum(int phi^2 ds)
    h = 1.0
    n = 1000
    L = h / n
    lengths = [L] * n
    # rigid translation mode phi=1 -> m_e = m
    mode_rigid = [1.0] * (n + 1)
    assert equivalent_mass([2.0] * n, lengths, mode_rigid) == pytest.approx(2.0, rel=1e-9)
    # uniform mass, mode phi=(s/h)^2 -> m_e = m (integral of phi^2 cancels)
    xs = [i * L for i in range(n + 1)]
    mode_sq = [(x / h) ** 2 for x in xs]
    assert equivalent_mass([2.0] * n, lengths, mode_sq) == pytest.approx(2.0, rel=1e-6)
    # non-uniform m(s) = m0*(1+s/h), phi=(s/h)^2:
    #   m_e = m0 * int(1+u) u^4 du / int u^4 du = m0*(11/30)/(1/5) = m0*11/6
    m0 = 2.0
    masses = [m0 * (1 + xs[i] / h) for i in range(n)]
    assert equivalent_mass(masses, lengths, mode_sq) == pytest.approx(m0 * 11 / 6, rel=1e-3)


def test_vortex_inertia_load_e6_formula():
    # F_w = m (2 pi f)^2 phi y_max, phi averaged over the element
    m = 1.5
    f = 4.0
    y_max = 0.01
    mode_w = [0.0, 0.25, 1.0]  # two elements, mid phi = 0.125 and 0.625
    q = vortex_inertia_load([m, m], f, mode_w, y_max)
    omega = 2 * math.pi * f
    assert q[0] == pytest.approx(m * omega**2 * 0.125 * y_max, rel=1e-12)
    assert q[1] == pytest.approx(m * omega**2 * 0.625 * y_max, rel=1e-12)


def test_vortex_response_base_moment_closed_form():
    # E.6-equivalent static load q(s)=m (2 pi f)^2 (s/h)^2 y_max on a uniform
    # cantilever -> base moment M = m w^2 y_max h^2/4 (exact integral)
    h = 1.0
    m = 1.0
    I = 1.0e-8
    y_max = 0.01
    n = 60
    xs = [i * h / n for i in range(n + 1)]
    nodes = [Node(x) for x in xs]
    elements = [Element(i, i + 1, E, I) for i in range(n)]
    supports = [Support(0, "fixed")]
    freqs, _ = modal_data(nodes, elements, supports, [m] * n, n_modes=1)
    w = 2 * math.pi * freqs[0]
    mode_w = [(x / h) ** 2 for x in xs]  # analytic mode, normalized to 1 at tip
    loads = vortex_inertia_load([m] * n, freqs[0], mode_w, y_max)
    dloads = [DistributedLoad(i, q) for i, q in enumerate(loads)]
    u = solve(nodes, elements, supports, dloads)
    m_base = abs(internal_moment(elements, 0, nodes, u, dloads, 0.0))
    closed = m * w**2 * y_max * h**2 / 4.0
    assert m_base == pytest.approx(closed, rel=2e-3)
    # per-element hand-sum: M = sum(q_k * L * x_bar_k)
    hand = sum(q * (h / n) * (xs[i] + h / (2 * n)) for i, q in enumerate(loads))
    assert m_base == pytest.approx(hand, rel=1e-9)
