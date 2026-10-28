"""Fast tests of v102: the systems, the half-unit rule, one elliptical solve through check."""
import math

from slippy_bridge.contact import Grid, hertz, solve

from studies.v100_dry_contact.v102_hertz_ellipses import check, crossed_cylinders, literature, table5
from studies.v100_dry_contact.v102_hertz_ellipses.check import half_unit


def test_crossed_cylinders_give_the_asked_alpha():
    for alpha in (1.0, 5.0, 100.0):
        c = crossed_cylinders(alpha)
        assert math.isclose(c.kappa_max / c.kappa_min, alpha, rel_tol=1e-12)


def test_table5_systems_from_the_print():
    t5 = table5()
    assert set(t5) == {"wheel on rail", "ball on plane", "ball-outer-ring contact"}
    ring = t5["ball-outer-ring contact"]
    assert all(map(math.isclose, ring.body2.r, (-38.9e-3, -6.6e-3))) and math.isclose(ring.load, 222.4111)
    assert math.isclose(ring.kappa_min / ring.kappa_max, 1 / 22.0905, rel_tol=1e-4)    # printed alpha


def test_half_unit_of_the_printed_digit():
    assert math.isclose(half_unit("0.0250"), 0.00005 / 0.025)
    assert math.isclose(half_unit("3.56e-04"), 0.005e-4 / 3.56e-4)


def test_check_of_one_ellipse_passes(tmp_path):
    c = crossed_cylinders(10.0)
    hz = hertz(c)
    grid = Grid.around(c, hz, h=hz.b / 16.0, span_x=2.0, span_y=2.0, angle="principal")
    r = solve(c, grid, tmp_path, hertz_result=hz, use_frequency_domain=False)
    rows = {row["id"] + row["quantity"]: row for row in check(r)}
    for key in ("C1delta", "C3p0", "C4b", "E1k = a/b", "E2orientation of a"):
        assert rows[key]["verdict"] == "PASS", rows[key]


def test_literature_cases_have_pages():
    cases = literature.cases()
    assert len(cases) == 10 and all(row["pdf_page"] for case in cases for row in case.published)
