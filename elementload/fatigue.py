"""Fatigue life estimates (stress-life, Miner's rule) for aluminum Yagi elements.

Stress amplitude S_a (fully reversed, R = -1) vs cycles N follows the closed-form
fit to 6061-T6 fatigue data of Yahr (1997), "Fatigue Design Curves for 6061-T6
Aluminum", ASME J. Pressure Vessel Technology:

    S_a = A / sqrt(N) + S_e          (S_a, S_e in Pa; N in cycles)

with A = 14 479 MPa and S_e = 96.5 MPa (14 ksi) for 6061-T6.  Aluminum has no
true endurance limit; S_e is the practical long-life fatigue strength.  Below S_e
the curve is treated as infinite life (constant-amplitude assumption).

6063-T6 is anchored to its published fatigue strength (ASM Handbook): 69 MPa at
the same long-life condition; A is scaled by 69/96.5 to keep the curve shape.
"""

import math

SN_DATA = {
    "al6061-t6": {"A": 14479.0e6, "endurance": 96.5e6},
    "al6063-t6": {"A": 10352.0e6, "endurance": 69.0e6},
}


def stress_amplitude_at(cycles, material="al6061-t6"):
    """S_a(N): fully-reversed stress amplitude (Pa) that fails at N cycles."""
    data = SN_DATA[material]
    return data["A"] / math.sqrt(cycles) + data["endurance"]


def cycles_to_failure(stress_amplitude, material="al6061-t6"):
    """N(S_a): cycles to failure; infinity below the endurance stress amplitude."""
    data = SN_DATA[material]
    if stress_amplitude <= data["endurance"]:
        return float("inf")
    return (data["A"] / (stress_amplitude - data["endurance"])) ** 2


def stress_range_to_failure(stress_range, material="al6061-t6"):
    """N(Delta sigma): cycles to failure for a fully-reversed stress range."""
    return cycles_to_failure(stress_range / 2.0, material)


def damage_fraction(stress_amplitude, cycles, material="al6061-t6"):
    """n/N: damage from `cycles` applied at `stress_amplitude` (Miner)."""
    n_fail = cycles_to_failure(stress_amplitude, material)
    if math.isinf(n_fail):
        return 0.0
    return cycles / n_fail


def damage_per_hour(stress_range, frequency, material="al6061-t6"):
    """Miner damage accumulated per hour of continuous cycling at `frequency` Hz."""
    n_fail = stress_range_to_failure(stress_range, material)
    if math.isinf(n_fail):
        return 0.0
    return 3600.0 * frequency / n_fail


def miner_life_hours(cases, material="al6061-t6"):
    """Hours of operation until cumulative damage D = 1 (Miner's rule).
    `cases` is a list of (stress_range, frequency) pairs, each assumed to act
    continuously for its share of the load spectrum."""
    damage = 0.0
    for stress_range, frequency in cases:
        damage += damage_per_hour(stress_range, frequency, material)
    if damage <= 0.0:
        return float("inf")
    return 1.0 / damage
