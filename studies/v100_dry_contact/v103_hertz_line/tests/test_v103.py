"""Fast tests of v103: the system, one line solve through check, the literature rows."""
import math

from slippy_bridge.contact import Grid, hertz, solve

from studies.v100_dry_contact.v103_hertz_line import check, literature, rollers


def test_the_system_is_a_line_unsolved():
    c = rollers()
    assert c.line and c.kappa_min == 0.0 and c.psi == 0.0 and c.load == 1e5


def test_check_of_a_line_on_a_wide_domain(tmp_path):
    c = rollers()
    hz = hertz(c)
    grid = Grid.around(c, hz, h=hz.b / 16.0, span_x=8.0, n_y_line=8)
    rows = {r["id"]: r for r in check(solve(c, grid, tmp_path, hertz_result=hz, use_frequency_domain=True))}
    assert rows["C3"]["verdict"] == "PASS" and rows["C4"]["verdict"] == "PASS"
    assert math.isclose(rows["C6"]["measured"], 1e5, rel_tol=1e-3)
    assert rows["L1"]["verdict"] == "RECORD" and math.isclose(rows["L1"]["reference"], 4 / math.pi, rel_tol=1e-7)


def test_literature_rows_have_pages():
    (case,) = literature.cases()
    assert case.build().label == "rollers" and all(r["pdf_page"] for r in case.published)
