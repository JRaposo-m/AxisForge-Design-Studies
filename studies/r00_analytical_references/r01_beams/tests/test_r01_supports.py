"""r01 extension (SW-03): beams on three or more supports, springs, settlements, clamps.

Judges: Harris2007_vol2 Eqs 9.42-9.44 (three-bearing shaft, point load in span 1, PDF p. 285) and the
slope-deflection Eqs 9.24-9.27 (PDF p. 281); Clapeyron's three-moment equation and textbook hand cases derived in
the docstrings (no handbook held). The existing tests of ``test_r01.py`` are unchanged.
"""
import math

import numpy as np
import pytest

from studies.r00_analytical_references.r01_beams import (DistributedLoad, Segment, Support, solve_stepped,
                                                         solve_supported)

EI, KGA = 8.3498e9, 5.0605e7                    # [N.mm^2], [N]: the steel d 30 mm shaft of test_r01
L1, L2, A, P = 100.0, 150.0, 40.0, 1000.0       # spans [mm], load position in span 1 [mm], load [N]
W = 5.0                                         # uniform load [N/mm]


def _three(kGA=None):
    """Rigid supports at 0, L1, L1 + L2, a downward force P at A."""
    return solve_supported([(A, -P)], [], [Support(0.0), Support(L1), Support(L1 + L2)],
                           [Segment(0.0, L1 + L2, EI, kGA)], L1 + L2)


def test_harris_three_bearing_reactions_eq_9_42_to_9_44():
    """Harris2007_vol2 Eqs 9.42-9.44 (PDF p. 285), F upward, P downward in span 1 at a (rigid bearings, uniform EI,
    no moment loads, equal bearing deflections: the assumptions of PDF p. 284). The inputs are this case's own:
    Harris prints no numbers for Fig. 9.9."""
    l1, l2, a = L1, L2, A
    F1 = P * (l1 - a) * (2 * l1 * (l1 + l2) - a * (l1 + a)) / (2 * l1**2 * (l1 + l2))
    F2 = P * a * ((l1 + l2) ** 2 - a**2 - l2**2) / (2 * l1**2 * l2)
    F3 = -P * a * (l1**2 - a**2) / (2 * l1 * l2 * (l1 + l2))
    assert _three().R == pytest.approx([F1, F2, F3], rel=1e-12)


def test_clapeyron_moment_over_the_middle_support():
    """Three-moment equation, ends pinned: 2 M_B (l1 + l2) = -P a b (l1 + a) / l1, b = l1 - a (hogging, M < 0)."""
    b = L1 - A
    assert float(_three().M(L1)) == pytest.approx(-P * A * b * (L1 + A) / (2 * L1 * (L1 + L2)), rel=1e-10)


def test_two_equal_spans_uniform_load_3_8_and_5_4():
    """Two equal spans L under q: end reactions 3/8 q L, middle 5/4 q L, M over the middle -q L^2 / 8; v = 0 there."""
    L = 100.0
    sol = solve_supported([], [DistributedLoad(0.0, 2 * L, lambda x: -W)], [Support(0.0), Support(L), Support(2 * L)],
                          [Segment(0.0, 2 * L, EI)], 2 * L)
    assert sol.R == pytest.approx([3 / 8 * W * L, 5 / 4 * W * L, 3 / 8 * W * L], rel=1e-12)
    assert float(sol.M(L)) == pytest.approx(-W * L**2 / 8, rel=1e-9)
    assert sol.v([0.0, L, 2 * L]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)


def test_settlement_of_the_middle_support():
    """Two equal spans, no load, middle support settled by s: the beam is a pin-pin span 2L pulled by R at mid-span,
    s = R (2L)^3 / (48 EI) so R = 6 EI s / L^3, ends -R/2 each; v(L) = s."""
    L, s = 100.0, -0.01
    sol = solve_supported([], [], [Support(0.0), Support(L, settlement=s), Support(2 * L)], [Segment(0.0, 2 * L, EI)],
                          2 * L)
    R = 6 * EI * s / L**3
    assert sol.R == pytest.approx([-R / 2, R, -R / 2], rel=1e-12)
    assert float(sol.v(L)) == pytest.approx(s, rel=1e-12)


def test_stiff_spring_equals_rigid_and_soft_spring_equals_two_supports():
    """k -> inf: the rigid three-support result; k -> 0: the pin-pin beam of solve_stepped (the middle support gone)."""
    x = np.linspace(0.0, L1 + L2, 26)
    rigid = _three()
    stiff = solve_supported([(A, -P)], [], [Support(0.0), Support(L1, k_r=1e15), Support(L1 + L2)],
                            [Segment(0.0, L1 + L2, EI)], L1 + L2)
    assert stiff.v(x) == pytest.approx(rigid.v(x), abs=1e-9 * np.max(np.abs(rigid.v(x))))
    soft = solve_supported([(A, -P)], [], [Support(0.0), Support(L1, k_r=1e-9), Support(L1 + L2)],
                           [Segment(0.0, L1 + L2, EI)], L1 + L2)
    pins = solve_stepped([(A, -P)], [], 0.0, L1 + L2, [Segment(0.0, L1 + L2, EI)], L1 + L2)
    assert soft.v(x) == pytest.approx(pins.v(x), rel=1e-9)


def test_elastic_end_supports_move_the_beam_rigidly_when_determinate():
    """Two supports with radial springs k: the moment is unchanged, each end sinks by R/k."""
    k = 2e4
    sol = solve_supported([(A, -P)], [], [Support(0.0, k_r=k), Support(L1, k_r=k)], [Segment(0.0, L1, EI)], L1)
    pins = solve_stepped([(A, -P)], [], 0.0, L1, [Segment(0.0, L1, EI)], L1)
    assert sol.R == pytest.approx([pins.R_A, pins.R_B], rel=1e-12)
    assert sol.v([0.0, L1]) == pytest.approx([-pins.R_A / k, -pins.R_B / k], rel=1e-12)
    assert sol.M(np.linspace(0, L1, 11)) == pytest.approx(pins.M(np.linspace(0, L1, 11)), abs=1e-6)


def test_clamp_and_pin_under_uniform_load():
    """Propped cantilever (clamp at 0 by k_theta = inf, pin at L), downward q: R_B = 3/8 q L, R_A = 5/8 q L,
    |M(0)| = q L^2 / 8, theta(0) = 0."""
    L = 200.0
    sol = solve_supported([], [DistributedLoad(0.0, L, lambda x: -W)], [Support(0.0, k_theta=math.inf), Support(L)],
                          [Segment(0.0, L, EI)], L)
    assert sol.R == pytest.approx([5 / 8 * W * L, 3 / 8 * W * L], rel=1e-12)
    assert abs(sol.C[0]) == pytest.approx(W * L**2 / 8, rel=1e-12)
    assert float(sol.theta(0.0)) == pytest.approx(0.0, abs=1e-15)


def test_slope_deflection_of_a_clamped_span_eq_9_24_to_9_27():
    """Harris Eqs 9.24-9.27 (PDF p. 281), SIGNED: a clamped-clamped span l (Euler-Bernoulli) with a downward P at a,
    both ends settled and both ends rotated. Harris's senses (Eqs 9.6-9.10, Fig. 9.4, PDF p. 278-279): y and delta_r
    positive DOWN (along P), theta = dy/dx, F upward on the shaft, M_1 counter-clockwise, M_2 clockwise. r01 has +v up
    and couples counter-clockwise, so delta = -settlement, theta_H = -rotation, F = R, M_1 = C[0], M_2 = -C[1].
    Load terms: the fixed-end forces P b^2 (l + 2a)/l^3, P a^2 (l + 2b)/l^3 and couples P a b^2/l^2, P a^2 b/l^2
    (no applied couple T: its sense is not checked). The 2 (delta_1 - delta_2)/l is that of Eq. 9.24 (Eq. 9.30 prints 3).
    """
    L, a, s1, s2, phi1, phi2 = 120.0, 45.0, 0.02, -0.005, 1e-3, -4e-4
    b = L - a
    sup = [Support(0.0, k_theta=math.inf, settlement=s1, rotation=phi1),
           Support(L, k_theta=math.inf, settlement=s2, rotation=phi2)]
    sol = solve_supported([(a, -P)], [], sup, [Segment(0.0, L, EI)], L)
    d1, d2, t1, t2 = -s1, -s2, -phi1, -phi2                      # Harris's senses (down positive)
    F1 = P * b**2 * (L + 2 * a) / L**3 - 6 * EI / L**2 * (t1 + t2 + 2 * (d1 - d2) / L)        # Eq. 9.24
    M1 = P * a * b**2 / L**2 - 2 * EI / L * (2 * t1 + t2 + 3 * (d1 - d2) / L)                 # Eq. 9.25
    F2 = P * a**2 * (L + 2 * b) / L**3 + 6 * EI / L**2 * (t1 + t2 + 2 * (d1 - d2) / L)        # Eq. 9.26
    M2 = P * a**2 * b / L**2 + 2 * EI / L * (t1 + 2 * t2 + 3 * (d1 - d2) / L)                 # Eq. 9.27
    assert sol.R == pytest.approx([F1, F2], rel=1e-12)
    assert [sol.C[0], -sol.C[1]] == pytest.approx([M1, M2], rel=1e-12)
    # the prescribed end state is met, and equilibrium (forces; moments about x = 0, couples counter-clockwise)
    assert sol.v([0.0, L]) == pytest.approx([s1, s2], rel=1e-12)
    assert sol.theta([0.0, L]) == pytest.approx([phi1, phi2], rel=1e-12)
    assert sol.R.sum() == pytest.approx(P, rel=1e-12)
    assert L * sol.R[1] + sol.C.sum() == pytest.approx(a * P, rel=1e-12)


def test_per_span_ei_and_timoshenko_limit():
    """A change of EI at the middle support (per-span EI) keeps equilibrium; Timoshenko with kGA -> inf equals EB."""
    seg = [Segment(0.0, L1, EI), Segment(L1, L1 + L2, 3 * EI)]
    sol = solve_supported([(A, -P)], [], [Support(0.0), Support(L1), Support(L1 + L2)], seg, L1 + L2)
    assert sol.R.sum() == pytest.approx(P, rel=1e-12)
    assert sol.R[1] * L1 + sol.R[2] * (L1 + L2) == pytest.approx(P * A, rel=1e-12)
    x = np.linspace(0.0, L1 + L2, 26)
    assert _three(kGA=KGA * 1e9).v(x) == pytest.approx(_three().v(x), rel=1e-7, abs=1e-12)
    assert abs(float(_three(kGA=KGA).v(A))) > abs(float(_three().v(A)))      # shear makes the beam softer


@pytest.mark.parametrize("supports", [[Support(0.0)], [Support(0.0), Support(0.0)], [Support(0.0), Support(300.0)],
                                      [Support(0.0, k_r=0.0), Support(100.0)], [Support(0.0, k_theta=-1.0), Support(1.0)]])
def test_bad_supports_raise(supports):
    with pytest.raises(ValueError):
        solve_supported([], [], supports, [Segment(0.0, 250.0, EI)], 250.0)


def test_clamp_with_a_vanishing_spring_is_a_cantilever():
    """Clamp at 0 and a radial spring k -> 0 at L: the cantilever of r01 C11 (the clamp takes P and P a)."""
    sol = solve_supported([(A, -P)], [], [Support(0.0, k_theta=math.inf), Support(L1, k_r=1e-12)],
                          [Segment(0.0, L1, EI)], L1)
    assert sol.R[0] == pytest.approx(P, rel=1e-9) and abs(sol.C[0]) == pytest.approx(P * A, rel=1e-9)
