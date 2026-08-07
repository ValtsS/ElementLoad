import pytest

from elementload.solver import (
    Node,
    Element,
    Support,
    PointLoad,
    DistributedLoad,
    solve,
    element_end_forces,
)

E = 69.0e9
I = 1.732902508e-8


def test_fixed_ends_uniform_load():
    # Fully fixed-fixed beam with uniform load q:
    #   w(L/2) = q*L^4 / (384*E*I), M(end) = -q*L^2/12, M(mid) = q*L^2/24
    L = 2.0
    q = -30.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "fixed"), Support(2, "fixed")]
    loads = [DistributedLoad(0, q), DistributedLoad(1, q)]
    u = solve(model, elements, supports, loads)
    assert u[1][0] == pytest.approx(q * L**4 / (384 * E * I), rel=1e-9)
    v_i, m_i, _, _ = element_end_forces(elements, 0, model, u, loads)
    assert m_i == pytest.approx(-q * L**2 / 12, rel=1e-9)
    _, m_mid, _, _ = element_end_forces(elements, 1, model, u, loads)
    assert m_mid == pytest.approx(q * L**2 / 24, rel=1e-9)


def test_propped_cantilever_udl_reaction():
    # Propped cantilever: fixed at node 0, roller at node 2 (span L), UDL q.
    # Roller reaction R = 3/8 * q*L (for q downward, R upward).
    L = 2.0
    q = -40.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "fixed"), Support(2, "roller")]
    loads = [DistributedLoad(0, q), DistributedLoad(1, q)]
    u = solve(model, elements, supports, loads)
    # Reaction at roller = end shear at node 2 of element 1
    _, _, v_j, _ = element_end_forces(elements, 1, model, u, loads)
    assert v_j == pytest.approx(-3 / 8 * q * L, rel=1e-9)


def test_reactions_balance_loads():
    # Fixed-roller propped cantilever, UDL: sum of reactions = -total load
    L = 2.0
    q = -40.0
    model = [Node(0.0), Node(L / 2), Node(L)]
    elements = [Element(0, 1, E, I), Element(1, 2, E, I)]
    supports = [Support(0, "fixed"), Support(2, "roller")]
    loads = [DistributedLoad(0, q), DistributedLoad(1, q)]
    u = solve(model, elements, supports, loads)
    _, _, v_r, _ = element_end_forces(elements, 1, model, u, loads)
    v_i, m_i, _, _ = element_end_forces(elements, 0, model, u, loads)
    total = q * L
    assert v_i + v_r == pytest.approx(-total, rel=1e-9)
    assert m_i == pytest.approx(-q * L**2 / 8, rel=1e-9)
