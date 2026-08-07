import math
from dataclasses import dataclass

from elementload.assembly import Tube, build_telescoping
from elementload.failure import allowables, local_buckling_stress
from elementload.geometry import tube_area, tube_section_modulus
from elementload.solver import (
    Node,
    Element,
    Support,
    DistributedLoad,
    solve,
    element_end_forces,
)
from elementload.wind import drag_coefficient, drag_per_unit_length, reynolds_number

RHO_AIR = 1.225
G = 9.81
AL_DENSITY = 2700.0


@dataclass
class Model:
    tube_specs: list
    overlap: float
    joint: str
    support_specs: list
    material: str
    wind_speed: float
    nodes: list
    elements: list
    supports: list
    loads: list
    segments: list
    pin_diameter: float = 0.004
    n_pins: int = 2


def build_model(tube_specs, overlap, joint, supports, wind_speed, material="al6061-t6",
                pin_diameter=0.004, n_pins=2):
    tubes = [Tube(*s) for s in tube_specs]
    nodes_x, segments = build_telescoping(tubes, overlap, joint)
    nodes = [Node(x) for x in nodes_x]
    index = {x: k for k, x in enumerate(nodes_x)}
    elements = [Element(index[s.x0], index[s.x1], s.e_modulus, s.inertia) for s in segments]
    loads = [DistributedLoad(i, wind_and_weight_load(seg, wind_speed, material))
             for i, seg in enumerate(segments)]
    supports_obj = []
    for pos, kind in supports:
        x = 0.0 if pos == "center" else pos
        idx = min(range(len(nodes_x)), key=lambda k: abs(nodes_x[k] - x))
        supports_obj.append(Support(idx, kind))
    return Model(tube_specs, overlap, joint, supports, material, wind_speed,
                 nodes, elements, supports_obj, loads, segments, pin_diameter, n_pins)


def wind_load(wind_speed, outer_d):
    cd = drag_coefficient(reynolds_number(wind_speed, outer_d))
    return drag_per_unit_length(wind_speed, cd, outer_d, RHO_AIR)


def self_weight_load(seg):
    area = tube_area(seg.outer_d, seg.wall)
    return area * AL_DENSITY * G


def wind_and_weight_load(seg, wind_speed, material):
    return wind_load(wind_speed, seg.outer_d) + self_weight_load(seg)


@dataclass
class Analysis:
    displacements: list
    max_moment: float
    max_moment_location: float
    max_stress: float
    max_stress_location: float
    max_deflection: float
    max_sag: float
    utilization: dict


def self_weight_loads(model):
    return [DistributedLoad(i, self_weight_load(seg))
            for i, seg in enumerate(model.segments)]


def analyze(model):
    u = solve(model.nodes, model.elements, model.supports, model.loads)
    max_moment = 0.0
    max_moment_location = 0.0
    max_stress = 0.0
    max_stress_location = 0.0
    yield_stress = allowables(model.material)["yield"]
    shear_strength = allowables(model.material)["shear"]
    pin_area = model.n_pins * math.pi * model.pin_diameter**2 / 4

    max_buckling_factor = 0.0
    buckling_location = 0.0
    max_pin_factor = 0.0
    pin_location = 0.0

    for i, seg in enumerate(model.segments):
        v_i, m_i, v_j, m_j = element_end_forces(model.elements, i, model.nodes, u, model.loads)
        z = tube_section_modulus(seg.outer_d, seg.wall)
        stress = max(abs(m_i), abs(m_j)) / z
        for m, x in ((m_i, seg.x0), (m_j, seg.x1)):
            if abs(m) > max_moment:
                max_moment = abs(m)
                max_moment_location = x
            if abs(m) / z > max_stress:
                max_stress = abs(m) / z
                max_stress_location = x
        sigma_cr = local_buckling_stress(seg.outer_d, seg.wall, seg.e_modulus)
        if stress / sigma_cr > max_buckling_factor:
            max_buckling_factor = stress / sigma_cr
            buckling_location = (seg.x0 if abs(m_i) >= abs(m_j) else seg.x1)
        if seg.is_overlap:
            shear = max(abs(v_i), abs(v_j))
            pin_factor = (shear / pin_area) / shear_strength
            if pin_factor > max_pin_factor:
                max_pin_factor = pin_factor
                pin_location = (seg.x0 if abs(v_i) >= abs(v_j) else seg.x1)

    max_deflection = max(abs(w) for w, _ in u)
    u_sag = solve(model.nodes, model.elements, model.supports, self_weight_loads(model))
    max_sag = max(abs(w) for w, _ in u_sag)
    utilization = {
        "yield": (max_stress / yield_stress, max_stress_location),
        "buckling": (max_buckling_factor, buckling_location),
        "pin_shear": (max_pin_factor, pin_location),
    }
    return Analysis(u, max_moment, max_moment_location, max_stress,
                    max_stress_location, max_deflection, max_sag, utilization)


def survival_wind_speed(model, mode="yield"):
    if mode not in ("yield", "buckling", "pin_shear"):
        raise ValueError(f"unknown failure mode: {mode}")
    low, high = 1.0, 500.0
    for _ in range(60):
        mid = (low + high) / 2
        m = build_model(model.tube_specs, model.overlap, model.joint,
                        model.support_specs, mid, model.material,
                        model.pin_diameter, model.n_pins)
        factor, _ = analyze(m).utilization[mode]
        if factor >= 1.0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def natural_frequencies(model, n_modes=3):
    from elementload.dynamics import natural_frequencies as _nf
    masses = [seg.mass_per_length for seg in model.segments]
    return _nf(model.nodes, model.elements, model.supports, masses, n_modes)


def _half_span(model):
    x0 = model.nodes[model.supports[0].node].x
    return max(abs(n.x - x0) for n in model.nodes)


def vortex_crosswind_response(model, damping=0.01, n_modes=3):
    """EN 1991-1-4 Annex E approach 1: crosswind response of each mode at lock-in.
    Returns per-mode dicts with amplitude y_max and cyclic bending stress range."""
    from elementload.dynamics import (
        MODE_SHAPE_FACTOR_CANTILEVER,
        correlation_length_factor_cantilever,
        critical_wind_speed,
        crosswind_amplitude,
        equivalent_mass,
        lateral_force_coefficient,
        lateral_force_coefficient_base,
        lock_in_window,
        modal_data,
        scruton_number,
        vortex_inertia_load,
    )
    from elementload.geometry import tube_section_modulus

    masses = [seg.mass_per_length for seg in model.segments]
    lengths = [model.nodes[e.j].x - model.nodes[e.i].x for e in model.elements]
    freqs, modes = modal_data(model.nodes, model.elements, model.supports, masses, n_modes)
    b = model.tube_specs[0][0]
    h = _half_span(model)
    k = MODE_SHAPE_FACTOR_CANTILEVER
    k_w = correlation_length_factor_cantilever(h, b)

    responses = []
    for i in range(n_modes):
        f = freqs[i]
        v_crit = critical_wind_speed(f, b)
        re = reynolds_number(v_crit, b)
        c_lat = lateral_force_coefficient(1.0, lateral_force_coefficient_base(re))
        mode_w = modes[i][0::2]
        m_e = equivalent_mass(masses, lengths, mode_w)
        sc = scruton_number(m_e, damping, b)
        y_max = crosswind_amplitude(b, sc, c_lat, k_w, k)
        loads = vortex_inertia_load(masses, f, mode_w, y_max)
        dloads = [DistributedLoad(j, q) for j, q in enumerate(loads)]
        u = solve(model.nodes, model.elements, model.supports, dloads)
        peak_stress = 0.0
        for j, seg in enumerate(model.segments):
            _, m_i, _, m_j = element_end_forces(model.elements, j, model.nodes, u, dloads)
            z = tube_section_modulus(seg.outer_d, seg.wall)
            peak_stress = max(peak_stress, abs(m_i) / z, abs(m_j) / z)
        responses.append(dict(
            frequency=f,
            v_crit=v_crit,
            lock_in=lock_in_window(v_crit),
            reynolds=re,
            c_lat=c_lat,
            mass_per_length=m_e,
            scruton=sc,
            y_max=y_max,
            stress_range=2.0 * peak_stress,
        ))
    return responses


def along_wind_dynamic_factor(model, damping=0.01, height=10.0, z_0=0.05, wind_speed=None):
    """EN 1991-1-4 6.3.1/Annex B: along-wind structural factor c_s c_d of the
    fundamental mode, including the aerodynamic damping delta_a (F.18)."""
    from elementload.dynamics import (
        aerodynamic_log_decrement,
        equivalent_mass,
        log_decrement_structural,
        modal_data,
        structural_factor,
    )
    v_m = wind_speed if wind_speed is not None else model.wind_speed
    masses = [seg.mass_per_length for seg in model.segments]
    lengths = [model.nodes[e.j].x - model.nodes[e.i].x for e in model.elements]
    freqs, modes = modal_data(model.nodes, model.elements, model.supports, masses, n_modes=1)
    f1 = freqs[0]
    b = model.tube_specs[0][0]
    h = max(n.x for n in model.nodes) - min(n.x for n in model.nodes)
    m_e = equivalent_mass(masses, lengths, modes[0][0::2])
    delta_s = log_decrement_structural(damping)
    c_f = drag_coefficient(reynolds_number(v_m, b))
    delta_a = aerodynamic_log_decrement(c_f, RHO_AIR, b, v_m, f1, m_e)
    delta = delta_s + delta_a
    r = structural_factor(height, b, h, f1, v_m, z_0, delta)
    r.update(frequency=f1, mass_per_length=m_e, delta_s=delta_s, delta_a=delta_a, delta=delta)
    return r


def max_wind_stress(model):
    """Peak bending stress (Pa) from the wind-only load at the design wind speed."""
    loads = [DistributedLoad(i, wind_load(model.wind_speed, seg.outer_d))
             for i, seg in enumerate(model.segments)]
    u = solve(model.nodes, model.elements, model.supports, loads)
    peak = 0.0
    for i, seg in enumerate(model.segments):
        _, m_i, _, m_j = element_end_forces(model.elements, i, model.nodes, u, loads)
        z = tube_section_modulus(seg.outer_d, seg.wall)
        peak = max(peak, abs(m_i) / z, abs(m_j) / z)
    return peak


def fatigue_report(model, damping=0.01, height=10.0, z_0=0.05):
    """Cyclic stress sources (VIV lock-in of the fundamental mode + gust buffeting
    resonance) and combined Miner fatigue life in hours. Each source is assumed to
    act continuously; VIV life is therefore the life under sustained lock-in.
    VIV is limited to the fundamental mode: the E.7 amplitude factors (K from Table
    E.5, K_w from the (s/h)^2 shape) are only defined for the fundamental mode."""
    from elementload.fatigue import damage_per_hour, miner_life_hours
    sources = []
    r = vortex_crosswind_response(model, damping, n_modes=1)[0]
    if r["stress_range"] > 0.0:
        sources.append(dict(
            name="VIV mode 1 lock-in",
            stress_range=r["stress_range"],
            frequency=r["frequency"],
            damage_per_hour=damage_per_hour(r["stress_range"], r["frequency"], model.material),
        ))
    daf = along_wind_dynamic_factor(model, damping, height, z_0)
    iv = daf["turbulence_intensity"]
    k_p = daf["peak_factor"]
    b2, r2 = daf["background"], daf["resonance"]
    sig_mean = max_wind_stress(model)
    resonant_amp = sig_mean * 2.0 * k_p * iv * (math.sqrt(b2 + r2) - math.sqrt(b2)) / (1.0 + 7.0 * iv)
    gust_range = 2.0 * resonant_amp
    sources.append(dict(
        name="gust buffeting",
        stress_range=gust_range,
        frequency=daf["frequency"],
        damage_per_hour=damage_per_hour(gust_range, daf["frequency"], model.material),
    ))
    total = sum(s["damage_per_hour"] for s in sources)
    cases = [(s["stress_range"], s["frequency"]) for s in sources]
    return dict(sources=sources, damage_per_hour=total,
                life_hours=miner_life_hours(cases, model.material))


def _model_mass(model):
    # length-weighted average mass per length (representative of the element)
    total_length = sum(s.x1 - s.x0 for s in model.segments)
    return sum(s.mass_per_length * (s.x1 - s.x0) for s in model.segments) / total_length
