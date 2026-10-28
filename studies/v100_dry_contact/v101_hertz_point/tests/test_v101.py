"""Fast tests of v101: the system, the comparator on one small solve, the literature."""
import math

import pytest
from slippy_bridge.contact import Grid, hertz, solve

from studies.v100_dry_contact.v101_hertz_point import ball_on_flat, check, literature, reference
from studies.v100_dry_contact.v101_hertz_point.study import KERNELS


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Johnson's system at a/h >= 16 (spatial kernel, interference control) and at a/h >= 8."""
    contact = ball_on_flat()
    hz = hertz(contact)
    delta = reference(contact, contact.load).delta_m
    _, span = KERNELS["spatial"]
    out = {}
    for aoh in (8.0, 16.0):
        grid = Grid.around(contact, hz, h=hz.a / aoh, span_x=span, span_y=span)
        out[aoh] = solve(contact, grid, tmp_path_factory.mktemp("slippy"), hertz_result=hz,
                         use_frequency_domain=False, interference=delta)
    return out


def test_the_system_is_returned_unsolved():
    c = ball_on_flat()
    assert (c.load, c.body1.r, c.body2.r, c.line) == (500.0, (6.35e-3, 6.35e-3), (math.inf, math.inf), False)


def test_check_passes_at_the_judged_grid(solved):
    rows = check(solved[16.0])
    assert [r["id"] for r in rows] == ["C1", "C2", "C3", "C4", "C4", "C5", "L1", "L2"]
    judged = [r for r in rows if r["verdict"] != "RECORD"]
    assert [r["id"] for r in judged] == ["C1", "C3", "C4", "C4"]
    assert all(r["verdict"] == "PASS" for r in judged), judged
    assert all(r["b_over_h"] >= 16.0 and r["kernel"] == "spatial" for r in rows)


def test_check_only_records_below_the_judged_grid(solved):
    assert {r["verdict"] for r in check(solved[8.0])} == {"RECORD"}


def test_literature_is_johnsons_sphere_with_its_pages():
    (case,) = literature.cases()
    assert case.build().label == "ball_on_flat"
    assert all(row["pdf_page"] and row["source_key"] == "Johnson1985" for row in case.published)
    assert [row["value"] for row in case.published if row["symbol"] == "p0/p_m"] == [1.5]
