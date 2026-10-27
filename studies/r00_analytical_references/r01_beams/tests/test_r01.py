"""r01_beams: closed forms, limits and identities (no published table held: README section 5)."""
import math

import numpy as np
import pytest

from studies.r00_analytical_references.r01_beams import (DistributedLoad, Segment, kappa_cowper, kappa_hutchinson,
                                                         section_stiffness, solve_stepped, solve_uniform)

E, NU, D = 210e3, 0.3, 30.0                      # [MPa], [-], [mm]: a solid steel shaft
EI, KGA = section_stiffness(E, NU, D)            # Cowper
L, F, Q = 200.0, 1000.0, 5.0                     # span [mm], point load [N], distributed load [N/mm]
X = np.linspace(0.0, L, 41)


def test_section_stiffness_from_the_design_inputs():
    """EI = E pi d^4 / 64, kGA = kappa G pi d^2 / 4 with G = E / (2 (1 + nu)); hollow EB: I of the annulus."""
    assert EI == pytest.approx(E * math.pi * D**4 / 64.0, rel=1e-15)
    assert KGA == pytest.approx(kappa_cowper(NU) * E / (2.0 * (1.0 + NU)) * math.pi * D**2 / 4.0, rel=1e-15)
    assert section_stiffness(E, NU, D, kappa="hutchinson")[1] / KGA == pytest.approx(
        kappa_hutchinson(NU) / kappa_cowper(NU), rel=1e-15)
    assert section_stiffness(E, NU, D, 20.0, kappa=None) == (pytest.approx(E * math.pi * (D**4 - 20.0**4) / 64.0,
                                                                           rel=1e-15), None)


def test_kappa_values_and_their_common_limit():
    """Printed doctest values at nu 0.3; both factors are 6/7 at nu = 0."""
    assert (round(kappa_cowper(0.3), 6), round(kappa_hutchinson(0.3), 6)) == (0.886364, 0.925182)
    assert kappa_cowper(0.0) == kappa_hutchinson(0.0) == pytest.approx(6.0 / 7.0, rel=1e-15)


def test_hollow_kappa_is_not_implemented():
    with pytest.raises(NotImplementedError, match="MG-02"):
        section_stiffness(E, NU, D, 20.0)


@pytest.mark.parametrize("args", [(0.0, NU, D, 0.0), (E, 0.5, D, 0.0), (E, NU, D, D), (E, NU, D, -1.0),
                                  (E, NU, D, 0.0, "timoshenko")])
def test_bad_section_raises(args):
    with pytest.raises(ValueError):
        section_stiffness(*args)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_simply_supported_central_load(kGA):
    """v(L/2) = F L^3 / (48 EI) + F L / (4 kGA); reactions F/2 against the load."""
    sol = solve_uniform([(L / 2.0, F)], 0.0, L, EI, kGA)
    shear = 0.0 if kGA is None else F * L / (4.0 * kGA)
    assert float(sol.v(L / 2.0)) == pytest.approx(F * L**3 / (48.0 * EI) + shear, rel=1e-12)
    assert (sol.R_A, sol.R_B) == (pytest.approx(-F / 2.0, rel=1e-15), pytest.approx(-F / 2.0, rel=1e-15))
    assert float(sol.M(L / 2.0)) == pytest.approx(-F * L / 4.0, rel=1e-15)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_overhang_tip_load(kGA):
    """Tip load F at a beyond the pin B: v_tip = F a^2 (L + a) / (3 EI) + F a (L + a) / (kGA L). The
    a^3 part of the bending term is the cantilever F a^3 / (3 EI); the rest is the pin rotation times a."""
    a = 60.0
    sol = solve_uniform([(L + a, F)], 0.0, L, EI, kGA)
    shear = 0.0 if kGA is None else F * a * (L + a) / (kGA * L)
    assert float(sol.v(L + a)) == pytest.approx(F * a**2 * (L + a) / (3.0 * EI) + shear, rel=1e-12)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_simply_supported_uniform_load(kGA):
    """Stepped solver, one segment, q over the span: v(L/2) = 5 q L^4 / (384 EI) + q L^2 / (8 kGA)."""
    sol = solve_stepped([], [DistributedLoad(0.0, L, lambda x: Q)], 0.0, L, [Segment(0.0, L, EI, kGA)], L)
    shear = 0.0 if kGA is None else Q * L**2 / (8.0 * kGA)
    assert float(sol.v(L / 2.0)) == pytest.approx(5.0 * Q * L**4 / (384.0 * EI) + shear, rel=1e-12)


def test_euler_bernoulli_is_timoshenko_with_infinite_kGA():
    loads = [(50.0, F), (140.0, -0.4 * F)]
    eb, stiff = solve_uniform(loads, 0.0, L, EI, None), solve_uniform(loads, 0.0, L, EI, 1e30)
    np.testing.assert_allclose(stiff.v(X), eb.v(X), rtol=1e-12, atol=1e-18)
    timo = solve_uniform(loads, 0.0, L, EI, KGA)
    np.testing.assert_allclose(timo.v_bending(X), eb.v(X), rtol=1e-15, atol=0.0)
    assert float(timo.v(50.0)) > float(eb.v(50.0)) > 0.0             # shear makes the beam softer


@pytest.mark.parametrize("kGA", [None, KGA])
def test_uniform_equals_stepped_with_equal_steps(kGA):
    """Three equal segments with overhangs at both ends: the stepped solver gives the Macaulay beam."""
    length, x_A, x_B = 260.0, 20.0, 220.0
    loads = [(0.0, 0.3 * F), (90.0, F), (150.0, -0.6 * F), (250.0, 0.2 * F)]   # V is right-continuous: no load at x = length
    segments = [Segment(0.0, 80.0, EI, kGA), Segment(80.0, 170.0, EI, kGA), Segment(170.0, length, EI, kGA)]
    uni, step = solve_uniform(loads, x_A, x_B, EI, kGA), solve_stepped(loads, [], x_A, x_B, segments, length)
    x = np.linspace(0.0, length, 53)
    scale = float(np.max(np.abs(uni.v(x))))
    np.testing.assert_allclose(step.v(x), uni.v(x), rtol=0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(step.M(x), uni.M(x), rtol=0.0, atol=1e-10 * F * length)
    np.testing.assert_allclose(step.V(x), uni.V(x), rtol=0.0, atol=1e-10 * F)
    assert (step.R_A, step.R_B) == (pytest.approx(uni.R_A, rel=1e-13), pytest.approx(uni.R_B, rel=1e-13))


def test_reactions_balance_the_loads():
    sol = solve_stepped([(70.0, F)], [DistributedLoad(100.0, 180.0, lambda x: Q * (x - 100.0) / 80.0)], 0.0, L,
                        [Segment(0.0, L, EI, KGA)], L)
    total = F + Q * 80.0 / 2.0
    moment_A = F * 70.0 + Q * 80.0 / 2.0 * (100.0 + 2.0 / 3.0 * 80.0)
    assert sol.R_A + sol.R_B == pytest.approx(-total, rel=1e-13)
    assert sol.R_B * L == pytest.approx(-moment_A, rel=1e-13)


@pytest.mark.parametrize("solve", [lambda: solve_uniform([], L, 0.0, EI, None),
                                   lambda: solve_stepped([], [], L, 0.0, [Segment(0.0, L, EI)], L)])
def test_supports_out_of_order_raise(solve):
    with pytest.raises(ValueError):
        solve()
