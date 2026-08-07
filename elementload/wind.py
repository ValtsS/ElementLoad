def drag_per_unit_length(velocity, cd, diameter, rho_air=1.225):
    return 0.5 * rho_air * velocity**2 * cd * diameter


def reynolds_number(velocity, diameter, nu_air=1.5e-5):
    return velocity * diameter / nu_air


# EN 1991-1-4, Fig. 7.28, smooth circular cylinder (k/b ~ 0): (Re, Cf,0).
# Subcritical plateau ~1.2 up to Re~2e5 (Wieselsberger data, per Schewe 1983),
# drag crisis 2e5..6e5 down to ~0.4, recovery toward ~0.8 transcritical.
# Cross-checked against Dlubal Eurocode verification example VE0308:
# Cf,0 = 0.55 at Re = 2e6.
_CD_CHART = [
    (1.0e4, 1.2),
    (1.0e5, 1.2),
    (2.0e5, 1.2),
    (3.0e5, 0.9),
    (4.0e5, 0.6),
    (5.0e5, 0.4),
    (6.0e5, 0.4),
    (1.0e6, 0.45),
    (2.0e6, 0.55),
    (5.0e6, 0.7),
    (1.0e7, 0.8),
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
