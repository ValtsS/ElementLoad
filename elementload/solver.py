from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve as dense_solve


@dataclass(frozen=True)
class Node:
    x: float


@dataclass(frozen=True)
class Element:
    i: int
    j: int
    e: float
    inertia: float


@dataclass(frozen=True)
class Support:
    node: int
    kind: str  # "fixed" | "pin" | "roller"


@dataclass(frozen=True)
class PointLoad:
    node: int
    force: float
    moment: float = 0.0


@dataclass(frozen=True)
class DistributedLoad:
    element: int
    q: float


def _elem_stiff_matrix(el, L):
    k = np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ]
    )
    return el / L**3 * k


def _elem_consistent_load(q, L):
    return np.array([q * L / 2, q * L**2 / 12, q * L / 2, -q * L**2 / 12])


def _elem_mass_matrix(m, L):
    # consistent mass matrix for a 2-node Euler-Bernoulli element
    return m * L / 420 * np.array(
        [
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L**2, 13 * L, -3 * L**2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
        ]
    )


def _element_dofs(e):
    return [2 * e.i, 2 * e.i + 1, 2 * e.j, 2 * e.j + 1]


def assemble_stiffness(nodes, elements):
    ndof = 2 * len(nodes)
    k_global = np.zeros((ndof, ndof))
    for e in elements:
        L = nodes[e.j].x - nodes[e.i].x
        k = _elem_stiff_matrix(e.e * e.inertia, L)
        dofs = _element_dofs(e)
        k_global[np.ix_(dofs, dofs)] += k
    return k_global


def constraint_dofs(supports, ndof):
    constrained = np.zeros(ndof, dtype=bool)
    for s in supports:
        constrained[2 * s.node] = True
        if s.kind == "fixed":
            constrained[2 * s.node + 1] = True
    return constrained


def solve(nodes, elements, supports, loads):
    n = len(nodes)
    ndof = 2 * n
    k_global = assemble_stiffness(nodes, elements)
    f_global = np.zeros(ndof)

    length = {id(e): nodes[e.j].x - nodes[e.i].x for e in elements}

    for load in loads:
        if isinstance(load, PointLoad):
            f_global[2 * load.node] += load.force
            f_global[2 * load.node + 1] += load.moment
        elif isinstance(load, DistributedLoad):
            e = elements[load.element]
            L = length[id(e)]
            f = _elem_consistent_load(load.q, L)
            dofs = _element_dofs(e)
            f_global[dofs] += f
        else:
            raise TypeError(f"unsupported load type: {type(load)}")

    constrained = constraint_dofs(supports, ndof)

    free = ~constrained
    k_red = k_global[np.ix_(free, free)]
    f_red = f_global[free]
    u_red = dense_solve(k_red, f_red)
    u = np.zeros(ndof)
    u[free] = u_red
    return [(u[2 * i], u[2 * i + 1]) for i in range(n)]


def element_end_forces(elements, index, nodes, u, loads):
    e = elements[index]
    L = nodes[e.j].x - nodes[e.i].x
    k = _elem_stiff_matrix(e.e * e.inertia, L)
    dofs = [2 * e.i, 2 * e.i + 1, 2 * e.j, 2 * e.j + 1]
    ue = np.array([u[dof // 2][dof % 2] for dof in dofs])
    f = k @ ue
    for load in loads:
        if isinstance(load, DistributedLoad) and load.element == index:
            f = f - _elem_consistent_load(load.q, L)
    return tuple(f)


def internal_moment(elements, index, nodes, u, loads, s):
    e = elements[index]
    v_i, m_i, _, _ = element_end_forces(elements, index, nodes, u, loads)
    q = sum(load.q for load in loads
            if isinstance(load, DistributedLoad) and load.element == index)
    return m_i - v_i * s - q * s**2 / 2
