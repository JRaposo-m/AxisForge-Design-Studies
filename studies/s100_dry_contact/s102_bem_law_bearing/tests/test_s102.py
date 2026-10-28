"""Fast tests of s102: the limit case (c_p law = AxisForge), the series law, the Newton record."""
import numpy as np
import pytest

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import deep_groove, load_node, solve_bearing
from studies.s100_dry_contact.s101_bem_ball_raceway_law import BemLaw
from studies.s100_dry_contact.s102_bem_law_bearing import power_law, series_law, solve_distribution


@pytest.mark.parametrize("s_mm, Fa", [(0.0, 0.0), (0.02, 0.0), (0.02, 2500.0)])
def test_cp_law_reproduces_axisforge(s_mm, Fa):
    """The limit case: with Q = c_p delta^1.5 this solver gives AxisForge's ISO/TS 16281 result."""
    b = deep_groove(s_mm)
    summ, res, el = solve_distribution(b, power_law(b.cp), 5000.0, Fa, label="t")
    ld = solve_bearing(b, load_node(b, 5000.0, Fa), postprocess=False)[0].load_distribution
    assert res.converged and summ["delta_r_mm"] == pytest.approx(ld.delta_r, rel=1e-6)
    Q = np.array([e["Q_N"] for e in el])
    assert np.max(np.abs(Q - ld.rows[0].Q_j)) / Q.max() < 1e-6
    assert res.records()[0]["k"] == 0 and res.records()[-1]["residual_norm"] <= 1e-10


def test_series_of_two_laws():
    """Two Hertz laws (c = 0) in series give K_s = (K_i^(-2/3) + K_o^(-2/3))^(-3/2) and c = 0; two softened laws give
    a softened ball whose delta is the sum of the contact deltas at every load."""
    Ki, Ko = 3.0e10, 3.4e10
    d = np.geomspace(1e-7, 3e-5, 8)

    def law(K, c):
        return BemLaw(name="x", delta_m=tuple(d), Q_N=tuple(K * d**1.5 * (1 + c * d)), K_hertz_N_per_m1_5=K,
                      c_per_m=c, fit_rms=0.0)
    s = series_law("s", law(Ki, 0.0), law(Ko, 0.0))
    Ks = (Ki ** (-2 / 3) + Ko ** (-2 / 3)) ** -1.5
    assert (s.K_hertz_N_per_m1_5, s.c_per_m) == pytest.approx((Ks, 0.0), abs=1e-6, rel=1e-12)
    soft = series_law("s", law(Ki, -800.0), law(Ko, -700.0))
    Q = np.array([100.0, 2000.0])
    assert soft.c_per_m < 0.0
    assert soft.delta(Q) == pytest.approx(law(Ki, -800.0).delta(Q) + law(Ko, -700.0).delta(Q), rel=1e-3)
    # two equal contacts share the approach half and half: K_s = K / 2^1.5 and, to first order, c_s = c / 2
    same = series_law("s", law(Ki, -800.0), law(Ki, -800.0))
    assert same.K_hertz_N_per_m1_5 == pytest.approx(Ki / 2**1.5, rel=1e-12)
    assert same.c_per_m == pytest.approx(-400.0, rel=1e-2)


def test_stribeck_invariance():
    """s = 0, Fa = 0: with Q = K delta^1.5 the load sharing Q_j / Fr does not depend on K, and delta goes as K^(-2/3)."""
    b = deep_groove(0.0)
    (s1, _, e1), (s2, _, e2) = (solve_distribution(b, power_law(K), 5000.0) for K in (b.cp, 2.0 * b.cp))
    assert np.array([e["Q_N"] for e in e2]) == pytest.approx(np.array([e["Q_N"] for e in e1]), abs=1e-7 * 5000.0)
    assert s2["delta_max_mm"] / s1["delta_max_mm"] == pytest.approx(2.0 ** (-2.0 / 3.0), rel=1e-8)
    assert s1["Kt_N_per_mm"] == pytest.approx(1.5 * 5000.0 / s1["delta_r_mm"], rel=1e-6)


def test_no_load_no_tension():
    assert power_law(1e5)([-1e-3, 0.0])[0] == 0.0
