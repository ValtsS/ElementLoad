import math

import numpy as np
from scipy.linalg import eigh

from elementload.geometry import tube_area
from elementload.solver import _elem_mass_matrix, _element_dofs, assemble_stiffness, constraint_dofs

RHO_AIR = 1.225
STROUHAL = 0.18  # EN 1991-1-4 E.1.3.2, smooth circular cylinder, all Re

# EN 1991-1-4 Fig. E.2 basic lateral force coefficient, smooth circular cylinder
C_LAT_0_SUBCRITICAL = 0.7
C_LAT_0_TRANSCRITICAL = 0.35
RE_CRIT_ONSET = 2.0e5
RE_CRIT_FULL = 1.0e6

MODE_SHAPE_FACTOR_CANTILEVER = 0.13  # EN 1991-1-4 Table E.5, cantilever fundamental mode


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


def modal_data(nodes, elements, supports, masses, n_modes=3):
    """Natural frequencies (Hz) and mass-normalized-free mode shapes.
    Each mode shape is a full 2*N vector with the transverse displacement
    normalized to 1 at the point of maximum displacement (EN 1991-1-4 E.6)."""
    ndof = 2 * len(nodes)
    k_global = assemble_stiffness(nodes, elements)
    m_global = assemble_mass(nodes, elements, masses)
    constrained = constraint_dofs(supports, ndof)
    free = ~constrained
    k_red = k_global[np.ix_(free, free)]
    m_red = m_global[np.ix_(free, free)]
    w2, vecs = eigh(k_red, m_red)
    w2 = w2[:n_modes]
    vecs = vecs[:, :n_modes]
    freqs = np.sqrt(w2) / (2 * np.pi)
    modes = np.zeros((n_modes, ndof))
    for i in range(n_modes):
        vec = np.zeros(ndof)
        vec[free] = vecs[:, i]
        peak = np.max(np.abs(vec[0::2]))
        if peak > 0.0:
            vec = vec / peak
        modes[i] = vec
    return freqs, modes


def natural_frequencies(nodes, elements, supports, masses, n_modes=3):
    freqs, _ = modal_data(nodes, elements, supports, masses, n_modes=n_modes)
    return freqs


def equivalent_mass(masses, lengths, mode_w):
    """EN 1991-1-4 F.14: m_e = sum(m_i * int phi^2 ds) / sum(int phi^2 ds)."""
    num = 0.0
    den = 0.0
    for m, L, w_a, w_b in zip(masses, lengths, mode_w[:-1], mode_w[1:]):
        phi2 = (w_a**2 + w_b**2) / 2.0
        num += m * phi2 * L
        den += phi2 * L
    return num / den


def vortex_inertia_load(masses, frequency, mode_w, y_max):
    """EN 1991-1-4 E.6: peak transverse load per unit length on each element,
    F = m * (2*pi*f)^2 * phi * y_max, phi averaged over the element."""
    omega = 2 * np.pi * frequency
    loads = []
    for m, w_a, w_b in zip(masses, mode_w[:-1], mode_w[1:]):
        phi = (w_a + w_b) / 2.0
        loads.append(m * omega**2 * phi * y_max)
    return loads


def shedding_frequency(wind_speed, outer_d):
    return STROUHAL * wind_speed / outer_d


def critical_wind_speed(frequency, outer_d):
    return frequency * outer_d / STROUHAL


def scruton_number(mass_per_length, damping_ratio, outer_d, rho_air=RHO_AIR):
    return 4 * np.pi * mass_per_length * damping_ratio / (rho_air * outer_d**2)


def lateral_force_coefficient_base(reynolds):
    """c_lat,0 for a smooth circular cylinder, EN 1991-1-4 Fig. E.2 (graph approximation)."""
    if reynolds <= RE_CRIT_ONSET:
        return C_LAT_0_SUBCRITICAL
    if reynolds >= RE_CRIT_FULL:
        return C_LAT_0_TRANSCRITICAL
    t = (math.log10(reynolds) - math.log10(RE_CRIT_ONSET)) / (
        math.log10(RE_CRIT_FULL) - math.log10(RE_CRIT_ONSET)
    )
    return C_LAT_0_SUBCRITICAL + t * (C_LAT_0_TRANSCRITICAL - C_LAT_0_SUBCRITICAL)


def lateral_force_coefficient(vcrit_over_vm, c_lat_0):
    """EN 1991-1-4 Table E.3: c_lat from the critical/mean wind velocity ratio."""
    if vcrit_over_vm < 0.83:
        return (1.5 - 4.0 / 3.0 * vcrit_over_vm) * c_lat_0
    if vcrit_over_vm < 1.25:
        return 0.75 * c_lat_0
    return 0.0  # EN 1991-1-4 E.1.2(2): vortex shedding need not be investigated


def crosswind_amplitude(outer_d, scruton, c_lat, k_w, k, strouhal=STROUHAL):
    """EN 1991-1-4 E.7 (approach 1): y_max/b = (1/St^2)(1/Sc) c_lat K_w K."""
    return outer_d / strouhal**2 / scruton * c_lat * k_w * k


def correlation_length_factor_cantilever(half_length, outer_d, mode_exponent=2.0,
                                         correlation_length=None):
    """EN 1991-1-4 E.8 for a cantilever, mode phi=(s/h)^zeta, correlation region at the tip.
    Correlation length L_j from Table E.4 (small amplitude): L_j = 6*b. Capped at 0.6."""
    if correlation_length is None:
        correlation_length = 6.0 * outer_d
    lam = min(correlation_length, half_length) / half_length
    k_w = 1.0 - (1.0 - lam) ** (mode_exponent + 1.0)
    return min(k_w, 0.6)


def lock_in_window(critical_wind):
    """Mean-wind range with effective vortex resonance, Table E.3 ratios 0.83..1.25."""
    return (critical_wind / 1.25, critical_wind / 0.83)

