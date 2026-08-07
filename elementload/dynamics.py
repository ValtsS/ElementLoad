import math

import numpy as np
from scipy.linalg import eigh

from elementload.geometry import tube_area
from elementload.solver import _elem_mass_matrix, _element_dofs, assemble_stiffness, constraint_dofs

RHO_AIR = 1.225
STROUHAL = 0.18  # EN 1991-1-4 E.1.3.2, smooth circular cylinder, all Re


def mass_per_length(outer_d, wall, density):
    return tube_area(outer_d, wall) * density


def assemble_mass(nodes, elements, masses):
    ndof = 2 * len(nodes)
    m_global = np.zeros((ndof, ndof))
    for e, m in zip(elements, masses):
        L = nodes[e.j].x - nodes[e.i].x
        me = _elem_mass_matrix(m, L)
        dofs = _element_dofs(e)
        m_global[np.ix_(dofs, dofs)] += me
    return m_global


def natural_frequencies(nodes, elements, supports, masses, n_modes=3):
    ndof = 2 * len(nodes)
    k_global = assemble_stiffness(nodes, elements)
    m_global = assemble_mass(nodes, elements, masses)
    constrained = constraint_dofs(supports, ndof)
    free = ~constrained
    k_red = k_global[np.ix_(free, free)]
    m_red = m_global[np.ix_(free, free)]
    w2, _ = eigh(k_red, m_red)
    return np.sqrt(w2[:n_modes]) / (2 * np.pi)


def shedding_frequency(wind_speed, outer_d):
    return STROUHAL * wind_speed / outer_d


def critical_wind_speed(frequency, outer_d):
    return frequency * outer_d / STROUHAL


def scruton_number(mass_per_length, damping_ratio, outer_d, rho_air=RHO_AIR):
    return 4 * np.pi * mass_per_length * damping_ratio / (rho_air * outer_d**2)

