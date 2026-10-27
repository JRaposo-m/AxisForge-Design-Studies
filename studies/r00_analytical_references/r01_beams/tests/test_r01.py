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


# ---- point couples and the fixed end (CW-02) ------------------------------------------------------------------------
# Closed forms derived here by Macaulay integration of EI v'' = M (README section 2: no beam handbook held). A couple
# is positive from +x toward +v; it lowers M by C across its position and adds no shear force.
C0, A = 5.0e4, 60.0                              # couple [N.mm], its position on the span [mm]


def _uniform_and_stepped(loads, x_A, x_B, kGA, couples, length=L):
    """The same beam through both solvers (stepped: three equal segments)."""
    cuts = np.linspace(0.0, length, 4)
    segments = [Segment(lo, hi, EI, kGA) for lo, hi in zip(cuts[:-1], cuts[1:])]
    return (solve_uniform(loads, x_A, x_B, EI, kGA, couples=couples),
            solve_stepped(loads, [], x_A, x_B, segments, length, couples=couples))


@pytest.mark.parametrize("kGA", [None, KGA])
def test_couple_on_a_simply_supported_span(kGA):
    """Couple C0 at a on pins at 0 and L: R_A = C0 / L = -R_B; M = C0 x / L before a, C0 x / L - C0 after (jump -C0);
    v(x <= a) = C0 x (x^2 + 3 (L - a)^2 - L^2) / (6 L EI), v(a) = C0 a (L - a) (L - 2a) / (3 L EI). V = -C0 / L is
    constant on the span, so its shear deflection is a rigid rotation the pins take back: v is the same with shear,
    theta(0) is not (theta = v' - V / kGA)."""
    for sol in _uniform_and_stepped([], 0.0, L, kGA, [(A, C0)]):
        assert (sol.R_A, sol.R_B) == (pytest.approx(C0 / L, rel=1e-12), pytest.approx(-C0 / L, rel=1e-12))
        x = np.array([20.0, A - 1e-9, A, 150.0])
        np.testing.assert_allclose(sol.M(x), C0 * x / L - C0 * (x >= A), rtol=0.0, atol=1e-9 * C0)
        before = X[X <= A]
        np.testing.assert_allclose(sol.v(before), C0 * before * (before**2 + 3.0 * (L - A)**2 - L**2) / (6.0 * L * EI),
                                   rtol=0.0, atol=1e-12 * float(np.max(np.abs(sol.v(X)))))
        assert float(sol.v(A)) == pytest.approx(C0 * A * (L - A) * (L - 2.0 * A) / (3.0 * L * EI), rel=1e-11)
        shear = 0.0 if kGA is None else C0 / (L * kGA)
        assert float(sol.theta(0.0)) == pytest.approx(C0 * (3.0 * (L - A)**2 - L**2) / (6.0 * L * EI) + shear,
                                                      rel=1e-11)
        np.testing.assert_allclose(sol.V(X[:-1]), -C0 / L, rtol=0.0, atol=1e-10 * F)   # V(L) takes R_B


@pytest.mark.parametrize("kGA", [None, KGA])
def test_cantilever_tip_force(kGA):
    """Fixed at 0, P at L: v(L) = P L^3 / (3 EI) + P L / kGA, theta(L) = P L^2 / (2 EI) (the section rotation: shear
    does not turn it); R_A = -P, C_A = -P L, M = P (L - x)."""
    for sol in _uniform_and_stepped([(L, F)], 0.0, None, kGA, []):
        shear = 0.0 if kGA is None else F * L / kGA
        assert float(sol.v(L)) == pytest.approx(F * L**3 / (3.0 * EI) + shear, rel=1e-12)
        assert float(sol.theta(L)) == pytest.approx(F * L**2 / (2.0 * EI), rel=1e-12)
        assert (sol.R_A, sol.R_B, sol.C_A) == (pytest.approx(-F, rel=1e-15), None, pytest.approx(-F * L, rel=1e-15))
        np.testing.assert_allclose(sol.M(X[:-1]), F * (L - X[:-1]), rtol=1e-12, atol=1e-9 * F * L)
        assert float(sol.v(0.0)) == pytest.approx(0.0, abs=1e-15)
        assert float(sol.theta(0.0)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_cantilever_tip_couple(kGA):
    """Fixed at 0, couple C0 at L: v(L) = C0 L^2 / (2 EI), theta(L) = C0 L / EI, with or without shear (V = 0);
    R_A = 0, C_A = -C0, M = C0 on the span."""
    for sol in _uniform_and_stepped([], 0.0, None, kGA, [(L, C0)]):
        assert float(sol.v(L)) == pytest.approx(C0 * L**2 / (2.0 * EI), rel=1e-12)
        assert float(sol.theta(L)) == pytest.approx(C0 * L / EI, rel=1e-12)
        assert (sol.R_A, sol.C_A) == (0.0, pytest.approx(-C0, rel=1e-15))
        np.testing.assert_allclose(sol.M(X[:-1]), C0, rtol=1e-12)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_cantilever_fixed_at_the_right_end(kGA):
    """The mirror of the tip-force case: fixed at L, P at 0: v(0) = P L^3 / (3 EI) + P L / kGA,
    theta(0) = -P L^2 / (2 EI)."""
    for sol in _uniform_and_stepped([(0.0, F)], L, None, kGA, []):
        shear = 0.0 if kGA is None else F * L / kGA
        assert float(sol.v(0.0)) == pytest.approx(F * L**3 / (3.0 * EI) + shear, rel=1e-12)
        assert float(sol.theta(0.0)) == pytest.approx(-F * L**2 / (2.0 * EI), rel=1e-12)


@pytest.mark.parametrize("kGA", [None, KGA])
def test_fixed_inside_the_beam(kGA):
    """Fixed at L/2 with P at 0 and P at L: two cantilevers of length L/2,
    v(0) = v(L) = P (L/2)^3 / (3 EI) + P (L/2) / kGA,
    R_A = -2 P, C_A = 0 by symmetry."""
    h = L / 2.0
    for sol in _uniform_and_stepped([(0.0, F), (L, F)], h, None, kGA, []):
        tip = F * h**3 / (3.0 * EI) + (0.0 if kGA is None else F * h / kGA)
        assert (float(sol.v(0.0)), float(sol.v(L))) == (pytest.approx(tip, rel=1e-12), pytest.approx(tip, rel=1e-12))
        assert (sol.R_A, sol.C_A) == (pytest.approx(-2.0 * F, rel=1e-15), pytest.approx(0.0, abs=1e-9 * F * L))


@pytest.mark.parametrize("kGA", [None, KGA])
def test_cantilever_uniform_load_stepped(kGA):
    """Stepped solver, fixed at 0, q over the length: v(L) = q L^4 / (8 EI) + q L^2 / (2 kGA)."""
    sol = solve_stepped([], [DistributedLoad(0.0, L, lambda x: Q)], 0.0, None, [Segment(0.0, L, EI, kGA)], L)
    shear = 0.0 if kGA is None else Q * L**2 / (2.0 * kGA)
    assert float(sol.v(L)) == pytest.approx(Q * L**4 / (8.0 * EI) + shear, rel=1e-12)


MIXED = [(30.0, F), (120.0, -0.5 * F)], [(80.0, C0), (170.0, -2.0 * C0)]   # forces, couples


@pytest.mark.parametrize("x_B", [L, None], ids=["pins", "fixed"])
@pytest.mark.parametrize("kGA", [None, KGA])
def test_equilibrium_with_forces_and_couples(x_B, kGA):
    """Forces and moments about x = 0 sum to zero (a force F at x gives x F); M and V vanish past the last load."""
    loads, couples = MIXED
    uni, step = _uniform_and_stepped(loads, 0.0, x_B, kGA, couples)
    for sol in (uni, step):
        R_B = sol.R_B or 0.0
        assert sum(F_ for _, F_ in loads) + sol.R_A + R_B == pytest.approx(0.0, abs=1e-12 * F)
        moment = sum(x * F_ for x, F_ in loads) + sum(C for _, C in couples) + (x_B or 0.0) * R_B + sol.C_A
        assert moment == pytest.approx(0.0, abs=1e-12 * F * L)
        assert float(sol.M(L)) == pytest.approx(0.0, abs=1e-9 * F * L)
    assert float(uni.M(L + 10.0)) == pytest.approx(0.0, abs=1e-9 * F * L)       # Macaulay: also past the beam end
    assert float(uni.V(L + 10.0)) == pytest.approx(0.0, abs=1e-9 * F)


@pytest.mark.parametrize("x_B", [L, None], ids=["pins", "fixed"])
def test_couples_euler_bernoulli_is_timoshenko_with_infinite_kGA(x_B):
    loads, couples = MIXED
    eb = solve_uniform(loads, 0.0, x_B, EI, None, couples=couples)
    stiff = solve_uniform(loads, 0.0, x_B, EI, 1e30, couples=couples)
    np.testing.assert_allclose(stiff.v(X), eb.v(X), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(stiff.theta(X), eb.theta(X), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("x_B", [L, None], ids=["pins", "fixed"])
@pytest.mark.parametrize("kGA", [None, KGA])
def test_couples_uniform_equals_stepped(x_B, kGA):
    """Forces and couples, pins or a fixed end: the stepped solver with equal steps gives the Macaulay beam."""
    uni, step = _uniform_and_stepped(*MIXED[:1], 0.0, x_B, kGA, MIXED[1])
    scale = float(np.max(np.abs(uni.v(X))))
    np.testing.assert_allclose(step.v(X), uni.v(X), rtol=0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(step.theta(X), uni.theta(X), rtol=0.0, atol=1e-12 * scale / L)
    np.testing.assert_allclose(step.M(X), uni.M(X), rtol=0.0, atol=1e-10 * F * L)
    np.testing.assert_allclose(step.V(X[:-1]), uni.V(X[:-1]), rtol=0.0, atol=1e-10 * F)   # at L: left / right limit
    assert step.C_A == pytest.approx(uni.C_A, rel=1e-13, abs=1e-12 * F * L)
