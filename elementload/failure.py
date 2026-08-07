import math

_MATERIALS = {
    "al6061-t6": {"yield": 276.0e6, "shear": 207.0e6},
    "al6063-t6": {"yield": 170.0e6, "shear": 128.0e6},
}


def allowables(material):
    return _MATERIALS[material]


def yield_factor(moment, section_modulus, yield_stress):
    stress = moment / section_modulus
    return stress / yield_stress


def local_buckling_stress(outer_d, wall, e_modulus):
    mean_radius = (outer_d - wall) / 2
    return 0.3 * e_modulus * wall / mean_radius


def pin_shear_factor(axial_force, pin_diameter, n_pins, shear_strength):
    area = n_pins * math.pi * pin_diameter**2 / 4
    stress = axial_force / area
    return stress / shear_strength
