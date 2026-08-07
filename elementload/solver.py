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


def solve(nodes, elements, supports, loads):
    n = len(nodes)
    ndof = 2 * n
    k_global = np.zeros((ndof, ndof))
    f_global = np.zeros(ndof)

    length = {id(e): nodes[e.j].x - nodes[e.i].x for e in elements}

    for e in elements:
        L = length[id(e)]
        k = _elem_stiff_matrix(e.e * e.inertia, L)
        dofs = [2 * e.i, 2 * e.i + 1, 2 * e.j, 2 * e.j + 1]
        k_global[np.ix_(dofs, dofs)] += k

    for load in loads:
        if isinstance(load, PointLoad):
            f_global[2 * load.node] += load.force
            f_global[2 * load.node + 1] += load.moment
        elif isinstance(load, DistributedLoad):
            e = elements[load.element]
            L = length[id(e)]
            f = _elem_consistent_load(load.q, L)
            dofs = [2 * e.i, 2 * e.i + 1, 2 * e.j, 2 * e.j + 1]
            f_global[dofs] += f
        else:
            raise TypeError(f"unsupported load type: {type(load)}")

    constrained = np.zeros(ndof, dtype=bool)
    for s in supports:
        constrained[2 * s.node] = True
        if s.kind == "fixed":
            constrained[2 * s.node + 1] = True

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
