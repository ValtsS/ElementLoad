import math

import pytest

from elementload.fatigue import (
    SN_DATA,
    stress_amplitude_at,
    cycles_to_failure,
    stress_range_to_failure,
    damage_fraction,
    damage_per_hour,
    miner_life_hours,
)


def test_sn_6061_t6_literal_anchors():
    # Yahr (1997) fit S_a = 14479/sqrt(N) + 96.5 MPa, fully reversed 6061-T6
    assert SN_DATA["al6061-t6"]["A"] == pytest.approx(14479.0e6, rel=1e-9)
    assert SN_DATA["al6061-t6"]["endurance"] == pytest.approx(96.5e6, rel=1e-9)
    # hand-computed curve points
    assert stress_amplitude_at(1e4) == pytest.approx(241.29e6, rel=1e-4)
    assert stress_amplitude_at(1e5) == pytest.approx(142.287e6, rel=1e-4)
    assert stress_amplitude_at(1e6) == pytest.approx(110.979e6, rel=1e-4)
    assert stress_amplitude_at(1e7) == pytest.approx(101.079e6, rel=1e-4)
    # long-life asymptote ~96.5 MPa (published 14 ksi fatigue strength)
    assert stress_amplitude_at(5e8) == pytest.approx(97.15e6, rel=1e-3)


def test_sn_6063_t6_endurance():
    # ASM fatigue strength 6063-T6 = 69 MPa, 6061-T6 = 96 MPa
    assert SN_DATA["al6063-t6"]["endurance"] == pytest.approx(69.0e6, rel=1e-9)
    assert stress_amplitude_at(5e8, "al6063-t6") == pytest.approx(69.46e6, rel=1e-3)


def test_cycles_to_failure_inverts_curve():
    # N = (A/(S_a - S_e))^2; the curve inverts back to the anchor points
    n_fail = cycles_to_failure(110.979e6)
    assert n_fail == pytest.approx(1e6, rel=1e-3)
    n_fail = cycles_to_failure(241.29e6)
    assert n_fail == pytest.approx(1e4, rel=1e-3)
    # doubling the excess amplitude above the endurance limit quarters the life
    excess = 110.979e6 - 96.5e6
    s2 = 96.5e6 + 2.0 * excess
    assert cycles_to_failure(s2) == pytest.approx(1e6 / 4.0, rel=1e-9)


def test_cycles_to_failure_below_endurance_infinite():
    assert math.isinf(cycles_to_failure(96.5e6))
    assert math.isinf(cycles_to_failure(50.0e6))
    # a finite life just above the endurance amplitude
    assert math.isfinite(cycles_to_failure(100.0e6))


def test_stress_range_to_failure_uses_half_amplitude():
    # fully-reversed range Delta_sigma = 2*S_a
    amplitude = 110.979e6
    assert stress_range_to_failure(2.0 * amplitude) == pytest.approx(1e6, rel=1e-3)
    assert math.isinf(stress_range_to_failure(2.0 * 96.5e6))


def test_damage_fraction_miner():
    # n cycles at amplitude that fails at N -> damage n/N, linear accumulation
    n_fail = 1e6
    assert damage_fraction(110.979e6, 500_000) == pytest.approx(0.5, rel=1e-3)
    assert damage_fraction(110.979e6, 2_000_000) == pytest.approx(2.0, rel=1e-3)
    assert damage_fraction(50.0e6, 1e9) == 0.0  # below endurance -> no damage


def test_damage_per_hour_and_life():
    # at f = 1 Hz one hour is 3600 cycles; life = N_fail/(3600 f)
    f = 1.0
    sr = 2.0 * 110.979e6
    n_fail = stress_range_to_failure(sr)
    assert damage_per_hour(sr, f) == pytest.approx(3600.0 / n_fail, rel=1e-9)
    assert miner_life_hours([(sr, f)]) == pytest.approx(n_fail / 3600.0, rel=1e-9)
    # below endurance -> infinite life
    assert miner_life_hours([(2.0 * 50.0e6, 10.0)]) == float("inf")


def test_miner_multiple_sources_add_damage_rates():
    # two independent cyclic sources: total damage rate = sum of individual rates
    case_a = (2.0 * 110.979e6, 2.0)   # N=1e6 at 2 Hz -> 7200 cycles/h -> D/h=7.2e-3
    case_b = (2.0 * 241.29e6, 8.0)    # N=1e4 at 8 Hz -> 28800 cycles/h -> D/h=2.88
    d_a = damage_per_hour(*case_a)
    d_b = damage_per_hour(*case_b)
    assert d_a == pytest.approx(7200.0 / 1e6, rel=1e-9)
    assert d_b == pytest.approx(28800.0 / 1e4, rel=1e-9)
    assert miner_life_hours([case_a, case_b]) == pytest.approx(1.0 / (d_a + d_b), rel=1e-9)
