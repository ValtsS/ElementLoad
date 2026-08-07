def drag_per_unit_length(velocity, cd, diameter, rho_air=1.225):
    return 0.5 * rho_air * velocity**2 * cd * diameter


def reynolds_number(velocity, diameter, nu_air=1.5e-5):
    return velocity * diameter / nu_air


# EN 1991-1-4, Fig. 7.28, smooth cylinder: (Re, Cd) chart anchor points.
_CD_CHART = [
    (1.0e4, 1.2),
    (1.0e5, 1.1),
    (2.0e5, 1.0),
    (4.0e5, 0.4),
    (5.0e5, 0.35),
    (1.0e6, 0.6),
    (3.0e6, 0.8),
    (1.0e7, 0.9),
]


def drag_coefficient(re):
    if re <= _CD_CHART[0][0]:
        return _CD_CHART[0][1]
    if re >= _CD_CHART[-1][0]:
        return _CD_CHART[-1][1]
    for (r0, c0), (r1, c1) in zip(_CD_CHART, _CD_CHART[1:]):
        if r0 <= re <= r1:
            t = (re - r0) / (r1 - r0)
            return c0 + t * (c1 - c0)
    raise AssertionError("unreachable")


def drag_coefficient_at(velocity, diameter, nu_air=1.5e-5):
    return drag_coefficient(reynolds_number(velocity, diameter, nu_air))
