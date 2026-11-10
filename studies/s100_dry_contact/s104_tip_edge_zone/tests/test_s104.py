"""Fast tests of s104: the measures on a Hertz field, the points near A and E, one coarse grid series at C."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from slippy_bridge.contact import hertz

from studies.s100_dry_contact.s104_tip_edge_zone import (contact_at, contacts_near_end, edge_of, fzg_c40, grid_series,
                                                         hertz_measures, measures)


@pytest.fixture(scope="module")
def system():
    return fzg_c40()


def _hertz_field(b, h, p0=1e9):
    x = np.arange(-4 * b, 4 * b, h) + h / 2
    p = p0 * np.sqrt(np.clip(1.0 - (x / b) ** 2, 0.0, None))
    grid = SimpleNamespace(X=np.repeat(x[:, None], 4, axis=1), h=h)
    return SimpleNamespace(grid=grid, p=np.repeat(p[:, None], 4, axis=1), hertz=SimpleNamespace(p0=p0))


def test_measures_of_a_hertz_field_are_the_hertz_values():
    """A semi-ellipse sampled finely: b_eq = b, x_c = 0, edge share as hertz_measures (the closed form)."""
    b = 1e-4
    r = _hertz_field(b, b / 2000)
    for x_edge in (-0.5 * b, -1.5 * b, 0.25 * b):
        m, ref = measures(r, x_edge, b), hertz_measures(x_edge, b)
        assert m["b_eq_over_b"] == pytest.approx(1.0, abs=1e-5)
        assert m["x_c_over_b"] == pytest.approx(0.0, abs=1e-6)
        assert m["edge_share"] == pytest.approx(ref["edge_share"], abs=1e-3)
    assert hertz_measures(-3.0 * b, b)["edge_share"] == 0.0 and hertz_measures(0.0, b)["edge_share"] == 1.0


def test_measures_on_a_flat_punch_field():
    """A singular field (Johnson1985 Eq. (2.64), PDF p. 47: p ~ 1/sqrt(a^2 - x^2)): x_c = 0, b_eq = sqrt(2) a.
    The node sums integrate the r^(-1/2) ends as s104 does next to a tip (midpoint nodes, slow O(sqrt(h)) error)."""
    a, h = 1e-4, 1e-4 / 20000
    x = np.arange(-2 * a, 2 * a, h) + h / 2
    p = np.where(np.abs(x) < a, 1.0 / np.sqrt(np.clip(a * a - x * x, 1e-300, None)), 0.0)
    r = SimpleNamespace(grid=SimpleNamespace(X=x[:, None], h=h), p=p[:, None], hertz=SimpleNamespace(p0=1.0))
    m = measures(r, -2 * a, a)
    assert m["x_c_over_b"] == pytest.approx(0.0, abs=1e-6)
    assert m["b_eq_over_b"] == pytest.approx(math.sqrt(2.0), rel=5e-3)


def test_points_near_the_ends(system):
    """d_end/b of each point is its target; the edge near A is the tip of gear 2 (-x), near E the tip of gear 1 (+x)."""
    for end, sign in (("A", -1.0), ("E", 1.0)):
        for target, _, c in contacts_near_end(system, end, targets=(0.0, 0.5, 2.0)):
            e = edge_of(c, end)
            assert e["d_end_m"] / hertz(c).b == pytest.approx(target, abs=1e-6)
            assert math.copysign(1.0, e["x_edge_m"]) == sign


def test_coarse_grid_series_at_c_is_hertz(system, tmp_path):
    """C (no edge within the domain): the integral measures equal Hertz to 0.01 already on coarse grids."""
    c = contact_at(system, float(system["path"]["C"]), "C_test")
    rows, summary, _ = grid_series(c, edge_of(c, "C")["x_edge_m"], tmp_path, b_over_h=(8.0, 16.0, 32.0), span_x=6.0, label="C_test")
    assert len(rows) == 3 and all(r["converged"] for r in rows)
    assert rows[-1]["b_eq_over_b"] == pytest.approx(1.0, abs=0.01)
    assert abs(rows[-1]["x_c_over_b"]) < 0.01 and not rows[-1]["singular"]
    assert set(summary) == {"x_c_over_b", "b_eq_over_b", "edge_share"}
