import pytest

from elementload.solver import (
    Node,
    Element,
    Support,
    PointLoad,
    DistributedLoad,
    solve,
    element_end_forces,
    internal_moment,
)

E = 69.0e9
I = 1.732902508e-8


def test_cantilever_tip_load():
    # w(L) = P*L^3 / (3*E*I), theta(L) = P*L^2 / (2*E*I)
    L = 1.0
    P = 100.0
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [PointLoad(1, force=-P)]
    u = solve(model, elements, supports, loads)
    w_tip = u[1][0]
    theta_tip = u[1][1]
    assert w_tip == pytest.approx(-P * L**3 / (3 * E * I), rel=1e-9)
    assert theta_tip == pytest.approx(-P * L**2 / (2 * E * I), rel=1e-9)


def test_cantilever_uniform_load():
    # w(L) = q*L^4 / (8*E*I)
    L = 2.0
    q = -20.0  # N/m downward
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [DistributedLoad(0, q)]
    u = solve(model, elements, supports, loads)
    assert u[1][0] == pytest.approx(q * L**4 / (8 * E * I), rel=1e-9)


def test_simply_supported_center_point_load():
    # w(L/2) = P*L^3 / (48*E*I)
    L = 2.0
    P = -200.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "pin"), Support(2, "roller")]
    loads = [PointLoad(1, force=P)]
    u = solve(model, elements, supports, loads)
    assert u[1][0] == pytest.approx(P * L**3 / (48 * E * I), rel=1e-9)


def test_simply_supported_center_uniform_load():
    # w(L/2) = 5*q*L^4 / (384*E*I)
    L = 2.0
    q = -30.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "pin"), Support(2, "roller")]
    loads = [DistributedLoad(0, q), DistributedLoad(1, q)]
    u = solve(model, elements, supports, loads)
    assert u[1][0] == pytest.approx(5 * q * L**4 / (384 * E * I), rel=1e-9)


def test_cantilever_tip_load_moments():
    # M at fixed end = P*L, shear = P along the element
    L = 1.0
    P = 100.0
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [PointLoad(1, force=-P)]
    u = solve(model, elements, supports, loads)
    v_i, m_i, v_j, m_j = element_end_forces(elements, 0, model, u, loads)
    assert v_i == pytest.approx(P, rel=1e-9)  # upward reaction
    assert m_i == pytest.approx(P * L, rel=1e-9)
    assert v_j == pytest.approx(-P, rel=1e-9)


def test_cantilever_uniform_load_moments():
    # M at fixed end = q*L^2 / 2 (with q downward, sign flips)
    L = 2.0
    q = -20.0
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [DistributedLoad(0, q)]
    u = solve(model, elements, supports, loads)
    v_i, m_i, v_j, m_j = element_end_forces(elements, 0, model, u, loads)
    assert m_i == pytest.approx(-q * L**2 / 2, rel=1e-9)
    assert v_i == pytest.approx(-q * L, rel=1e-9)  # upward reaction = -total load


def test_simply_supported_center_point_load_moment():
    # M at center = P*L / 4
    L = 2.0
    P = -200.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "pin"), Support(2, "roller")]
    loads = [PointLoad(1, force=P)]
    u = solve(model, elements, supports, loads)
    _, m_center, _, _ = element_end_forces(elements, 1, model, u, loads)
    assert m_center == pytest.approx(P * L / 4, rel=1e-9)


def test_internal_moment_cantilever_uniform():
    # M(s) = m_i - v_i*s - q*s^2/2 = -q*(L-s)^2/2 for a uniform cantilever
    L = 2.0
    q = -20.0
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [DistributedLoad(0, q)]
    u = solve(model, elements, supports, loads)
    for s in (0.0, L / 4, L / 2, 3 * L / 4, L):
        expected = -q * (L - s) ** 2 / 2
        assert internal_moment(elements, 0, model, u, loads, s) == pytest.approx(expected, rel=1e-9)


def test_internal_moment_cantilever_tip_load():
    # M(s) = -P_applied*(L - s)
    L = 1.0
    P = -100.0
    model = [Node(0.0), Node(L)]
    elements = [Element(0, 1, E, I)]
    supports = [Support(0, "fixed")]
    loads = [PointLoad(1, force=P)]
    u = solve(model, elements, supports, loads)
    for s in (0.0, L / 3, 2 * L / 3, L):
        expected = -P * (L - s)
        assert internal_moment(elements, 0, model, u, loads, s) == pytest.approx(expected, rel=1e-9)
