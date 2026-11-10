"""Fast tests of s101: the contacts, the power-law fit, the law object, one small BEM solve."""
import math

import numpy as np
import pytest
from slippy_bridge.contact import hertz

from studies.s100_dry_contact.s101_bem_ball_raceway_law import (BemLaw, contacts, fit_law, fit_power_law,
                                                               solve_at_approach)


@pytest.fixture(scope="module")
def built():
    return contacts()


def test_contacts_are_the_most_loaded_balls(built):
    assert set(built) == {"6210 inner", "6210 outer", "7210 a40 inner", "7210 a40 outer"}
    _, _, c = built["6210 inner"]
    assert c.load == pytest.approx(2436.2, abs=0.1) and c.alpha_rad == pytest.approx(0.0, abs=1e-7)
    _, _, c = built["7210 a40 outer"]
    assert math.degrees(c.alpha_rad) == pytest.approx(38.04, abs=0.01)


def test_fit_recovers_a_hertz_law():
    d = np.geomspace(1e-6, 3e-5, 7)
    assert fit_power_law(d, 2.0e10 * d**1.5) == pytest.approx((2.0e10, 1.5), rel=1e-10)


def test_law_is_hertz_times_a_linear_ratio_and_inverts():
    """Q = K_H d^1.5 (1 + c d): Hertz as d -> 0, no tension, smooth; delta(Q) is its inverse."""
    law = BemLaw(name="t", delta_m=(1e-6, 1e-5), Q_N=(1.0, 1.0), K_hertz_N_per_m1_5=3e10, c_per_m=-800.0, fit_rms=0.0)
    d = np.array([1e-6, 1e-5, 2.5e-5])
    assert law.Q(d) == pytest.approx(3e10 * d**1.5 * (1 - 800.0 * d), rel=1e-14)
    assert law.Q([0.0, -1e-6]).tolist() == [0.0, 0.0]
    assert law.delta(law.Q(d)) == pytest.approx(d, rel=1e-12)


def test_fit_law_recovers_the_ratio_slope():
    d = np.geomspace(1e-6, 3e-5, 9)
    c, rms = fit_law(d, 3e10 * d**1.5 * (1 - 700.0 * d), 3e10)
    assert (c, rms) == pytest.approx((-700.0, 0.0), abs=1e-9)


def test_small_approach_meets_hertz_on_a_coarse_grid(built, tmp_path):
    """delta = 0.05 delta_H(Q_ref), a/r_g about 0.07, b/h asked 8 (fast): the load within 1 % of Hertz."""
    _, _, c = built["6210 outer"]
    d = 0.05 * hertz(c).delta
    r, hz, _ = solve_at_approach(c, d, tmp_path, b_over_h=8.0)
    assert r.converged and r.edge_clear
    assert r.load_found / r.contact.load == pytest.approx(1.0, abs=0.01)
