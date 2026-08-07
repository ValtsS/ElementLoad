import math


def tube_area(outer_d, wall):
    inner_d = outer_d - 2 * wall
    return math.pi / 4 * (outer_d**2 - inner_d**2)


def tube_inertia(outer_d, wall):
    inner_d = outer_d - 2 * wall
    return math.pi / 64 * (outer_d**4 - inner_d**4)


def tube_section_modulus(outer_d, wall):
    return tube_inertia(outer_d, wall) / (outer_d / 2)


def tube_mass_per_length(outer_d, wall, density):
    return tube_area(outer_d, wall) * density
