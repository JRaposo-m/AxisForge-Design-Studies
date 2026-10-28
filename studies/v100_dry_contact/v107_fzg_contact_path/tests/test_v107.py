"""Fast tests of v107: AxisForge against r04 along the path, and the Martins forces."""
from studies.v100_dry_contact.v107_fzg_contact_path import PAIRS, build_pair, check, literature


def test_axisforge_follows_r04_along_the_path():
    for name in PAIRS:
        rows = check(build_pair(name), 323.27, data=PAIRS[name])
        judged = [r for r in rows if r["verdict"] != "RECORD"]
        assert len(judged) == 30 and all(r["verdict"] == "PASS" for r in judged), [r for r in judged
                                                                                   if r["verdict"] != "PASS"]
        assert {r["point"] for r in rows if r["id"] == "P"} == set("ABCDE")


def test_spur_pressure_readings_are_recorded():
    rows = check(build_pair("C40"), 323.27, data=PAIRS["C40"])
    h = {r["point"]: r for r in rows if r["id"] == "H"}
    assert set(h) == {"A", "B", "C", "E"} and all(r["verdict"] == "RECORD" for r in h.values())
    assert h["A"]["quantity"].endswith("(share 0.5)") and h["C"]["quantity"].endswith("(share 1)")


def test_literature_cases_and_the_torques_of_construction():
    cases = literature.cases()
    assert len(cases) == 6 and {c.published["stage"] for c in cases} == {"K5", "K7", "K9"}
    from axisforge.outputs._format import read_records_csv

    from studies.v100_dry_contact.v107_fzg_contact_path.construction import WHEEL_TORQUE_NM
    from studies.v100_dry_contact.v107_fzg_contact_path.literature import PUBLISHED
    printed = {r["condition"].split(";")[0].removeprefix("stage="): r["value"] for r in read_records_csv(PUBLISHED)
               if r["quantity"] == "wheel torque"}
    assert printed == WHEEL_TORQUE_NM
