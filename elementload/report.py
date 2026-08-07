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
    utilization: dict


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
    utilization = {
        "yield": (max_stress / yield_stress, max_stress_location),
        "buckling": (max_buckling_factor, buckling_location),
        "pin_shear": (max_pin_factor, pin_location),
    }
    return Analysis(u, max_moment, max_moment_location, max_stress,
                    max_stress_location, max_deflection, utilization)


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
