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


# --- Along-wind dynamic amplification (EN 1991-1-4 6.3.1 / Annex B) ---

def turbulence_intensity(z, z_0, c_0=1.0, k_I=1.0):
    """EN 1991-1-4 (4.7): I_v(z) = k_I / (c_0 ln(z/z_0))."""
    return k_I / (c_0 * math.log(z / z_0))


def turbulence_length_scale(z, L_t=300.0, z_t=200.0):
    """EN 1991-1-4 B.1(1): L(z) = L_t (z/z_t)^0.67 for z >= z_min."""
    return L_t * (z / z_t) ** 0.67


def reduced_frequency(frequency, length_scale, mean_wind):
    """EN 1991-1-4 B.1(2): f_L(z,n) = n L(z) / v_m(z)."""
    return frequency * length_scale / mean_wind


def wind_spectrum(f_L):
    """EN 1991-1-4 B.1(2): non-dimensional power spectral density S_L."""
    return 6.8 * f_L / (1.0 + 10.2 * f_L) ** (5.0 / 3.0)


def aerodynamic_admittance(eta):
    """EN 1991-1-4 B.2(6): R_h/R_b; R(0) = 1."""
    if eta <= 0.0:
        return 1.0
    return 1.0 / eta - (1.0 - math.exp(-2.0 * eta)) / (2.0 * eta**2)


def background_factor(width, height, length_scale):
    """EN 1991-1-4 B.2(2): B^2 = 1/(1 + 0.9 ((b+h)/L(z_s))^0.63)."""
    return 1.0 / (1.0 + 0.9 * ((width + height) / length_scale) ** 0.63)


def resonance_response_factor(width, height, frequency, mean_wind, length_scale, delta):
    """EN 1991-1-4 B.2(5): R^2 = (pi^2/(2 delta)) S_L R_h R_b."""
    f_L = reduced_frequency(frequency, length_scale, mean_wind)
    s_l = wind_spectrum(f_L)
    eta_h = 4.6 * height / length_scale * f_L
    eta_b = 4.6 * width / length_scale * f_L
    r_h = aerodynamic_admittance(eta_h)
    r_b = aerodynamic_admittance(eta_b)
    return (math.pi**2 / (2.0 * delta)) * s_l * r_h * r_b


def peak_factor(frequency, T=600.0):
    """EN 1991-1-4 B.2(3): k_p = sqrt(2 ln(nu T)) + 0.6/sqrt(2 ln(nu T)), not below 3."""
    x = 2.0 * math.log(frequency * T)
    if x <= 0.0:
        return 3.0
    return max(math.sqrt(x) + 0.6 / math.sqrt(x), 3.0)


def structural_factor(z, width, height, frequency, mean_wind, z_0, delta,
                      c_0=1.0, k_I=1.0, T=600.0):
    """EN 1991-1-4 6.3.1(1) eq (6.1) with Annex B procedure 1.
    Returns c_s*c_d and its components (c_s, c_d, B^2, R^2, k_p, I_v, L)."""
    i_v = turbulence_intensity(z, z_0, c_0, k_I)
    l_s = turbulence_length_scale(z)
    b2 = background_factor(width, height, l_s)
    r2 = resonance_response_factor(width, height, frequency, mean_wind, l_s, delta)
    k_p = peak_factor(frequency, T)
    c_s_cd = (1.0 + 2.0 * k_p * i_v * math.sqrt(b2 + r2)) / (1.0 + 7.0 * i_v)
    c_s = (1.0 + 7.0 * i_v * b2) / (1.0 + 7.0 * i_v)
    return dict(c_s_cd=c_s_cd, c_s=c_s, c_d=c_s_cd / c_s, background=b2, resonance=r2,
                peak_factor=k_p, turbulence_intensity=i_v, length_scale=l_s)


def log_decrement_structural(damping_ratio):
    """EN 1991-1-4 F.5(2): logarithmic decrement of structural damping, delta_s = 2 pi zeta."""
    return 2.0 * math.pi * damping_ratio


def aerodynamic_log_decrement(force_coefficient, rho_air, width, mean_wind, frequency,
                              mass_per_length):
    """EN 1991-1-4 F.18: delta_a = c_f rho b v_m / (2 n_1 m_e)."""
    return (force_coefficient * rho_air * width * mean_wind
            / (2.0 * frequency * mass_per_length))

