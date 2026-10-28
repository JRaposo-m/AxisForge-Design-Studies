"""Run of v107: C40 and H501 at K5, K7, K9 through AxisForge, against r04 + r02 and Martins 2015.

``python -m studies.v100_dry_contact.v107_fzg_contact_path.study`` writes ``results/v107_check.csv``
(git-ignored).
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import write_records_csv

from . import literature
from .check import check
from .construction import PAIRS, WHEEL_TORQUE_NM

RESULTS = Path(__file__).resolve().parent / "results"


def run(results=RESULTS) -> dict:
    """Every literature case; write and return the records."""
    rows = []
    for case in literature.cases():
        name, stage = case.published["pair"], case.published["stage"]
        rows += [{"stage": stage, **r} for r in check(case.build(), WHEEL_TORQUE_NM[stage], data=PAIRS[name],
                                                       published=case.published)]
    Path(results).mkdir(parents=True, exist_ok=True)
    write_records_csv(Path(results) / "v107_check.csv", rows)
    return {"check": rows}


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("pair", "pair", "s"), ("stage", "stage", "s"), ("id", "id", "s"), ("point", "pt", "s"),
                        ("quantity", "quantity", "s"), ("axisforge", "AxisForge", ".6g"), ("reference", "ref", ".6g"),
                        ("err", "err", "+.2e"), ("verdict", "verdict", "s")), out["check"]))
