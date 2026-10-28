"""Fast tests of v105: the transfer AxisForge -> bridge (C0-C4), the RECORD split, the series, Hamrock."""
import math

import pytest

from studies.v000_axisforge.v003_iso16281_load_zone import deep_groove_6210
from studies.v100_dry_contact.v105_ball_contacts import check, load_cases, series_records
from studies.v100_dry_contact.v105_ball_contacts.study import contacts_of, literature_series


def _case(name):
    return next(c for c in load_cases() if c["name"] == name)


def test_transfer_is_judged_at_alpha_0_whatever_the_size():
    """Zero clearance: every element at alpha_0 = 0, so C0-C4 are judged at light and heavy load alike."""
    for name in ("6210 s 0 Fr 100 N", "6210 s 0 Fr 5000 N"):
        bearing, result, contacts = contacts_of(_case(name))
        rows = check(contacts, bearing, result)
        judged = [r for r in rows if r["id"] in ("C0", "C1", "C2", "C3", "C4")]
        assert judged and all(r["verdict"] == "PASS" for r in judged), [r for r in judged if r["verdict"] != "PASS"]


def test_elements_away_from_alpha_0_are_records():
    bearing, result, contacts = contacts_of(_case("6210 s 20 um Fr 5000 N"))
    rows = [r for r in check(contacts, bearing, result) if r["id"] in ("C1", "C2", "C3", "C4")]
    assert {r["verdict"] for r in rows} == {"RECORD"} and "SB-48" in rows[0]["note"]


def test_series_of_the_bridge_contacts():
    bearing, _, contacts = contacts_of(_case("6210 s 0 Fr 2000 N"))
    rows = series_records(contacts, bearing)
    assert {r["verdict"] for r in rows if r["id"] == "S1"} == {"PASS"}
    assert all(abs(r["rel_err"]) < 5e-3 for r in rows if r["id"] == "S2")                  # v003 C7 tolerance


def test_bridge_validity_is_looser_than_per_body():
    """Characterisation test (SB-07, OPEN R5-88): it pins today's bridge behaviour, a contact r02 puts at a/r1 ~ 0.32
    is valid for the bridge. It is EXPECTED to fail when SB-07 is ruled and the bridge reports the per-body ratio."""
    bearing, result, contacts = contacts_of(_case("6210 s 0 Fr 5000 N"))
    c5 = [r for r in check(contacts, bearing, result) if r["id"] == "C5" and r["j"] == 0]
    assert all(r["bridge_valid"] and r["size_ratio_r02"] > 0.3 and r["bridge_size_ratio"] < 0.1 for r in c5)
    assert {r["size_governing"] for r in c5} == {"a/r1"}                  # the ball governs, not the groove


def test_hamrock_radii():
    rows = {r["id"]: r for r in literature_series()}
    assert [rows[f"L{n}"]["verdict"] for n in (1, 2, 3, 4)] == ["PASS"] * 4
    assert all(rows[f"L{n}"]["verdict"] == "RECORD" for n in range(5, 13))
    assert math.isclose(rows["L9"]["reference"], 9.42)            # the printed k_i (FLAG, R5-09)
    assert rows["L9"]["measured"] == pytest.approx(9.285, abs=5e-4)   # exact k at alpha = 0 (physics review)


def test_r02_bodies_comes_from_v003():
    from studies.v000_axisforge.v003_iso16281_load_zone import r02_bodies
    body = r02_bodies(deep_groove_6210(0.0), "outer")
    assert body["r2_m"][0] < 0.0 and body["r2_m"][1] < 0.0 and body["r1_m"] == pytest.approx(6.35e-3)
