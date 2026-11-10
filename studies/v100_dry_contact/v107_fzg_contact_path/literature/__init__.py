"""The published systems v107 knows: the FZG C40 and H501 pairs of Martins 2015 (Tables 3-4, PDF p. 4).

Each case is one pair at one load stage (K5, K7, K9: the wheel torques of Table 4) with its printed radial and
axial forces and Hertzian pressure, values kept as printed strings. Rows: ``published/martins2015_t4.csv``
(vault extraction of 2026-10-16 through the tag ``studies-pre-sweep``). The torque and force rows repeat
r04's published table on purpose: a case keeps its own data (amendment item 2); the test ties the torques of
``construction.WHEEL_TORQUE_NM`` to these rows. The geometry rows are r04's.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import read_records_csv

from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import WHEEL_TORQUE_NM, build_pair

__all__ = ["cases"]

PUBLISHED = Path(__file__).with_name("published") / "martins2015_t4.csv"
_LABEL = {"C40": "Spur C40", "H501": "Helical H501/951"}


def cases() -> list[LiteratureCase]:
    """C40 and H501 at K5, K7, K9: ``published`` = {"stage", "Fr", "Fa", "p_H"} (printed strings)."""
    rows = read_records_csv(PUBLISHED, text_fields=("value",))
    out = []
    for name, label in _LABEL.items():
        for stage in WHEEL_TORQUE_NM:
            mine = {r["symbol"]: r["value"] for r in rows if f"stage={stage};gear={label}" in r["condition"]}
            out.append(LiteratureCase(f"martins2015_{name}_{stage}", lambda n=name: build_pair(n),
                                      {"pair": name, "stage": stage, "Fr": mine["Fr"], "Fa": mine["Fa"], "p_H": mine["p_H"]},
                                      "Martins2015 Table 4, PDF p. 4"))
    return out
