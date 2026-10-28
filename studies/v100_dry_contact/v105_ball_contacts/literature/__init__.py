"""The published system v105 knows (D-0036 amendment item 4): Hamrock (1983) sec. 14.2, the two contacts.

Hamrock's deep groove ball bearing (NASA RP-1105 sec. 14.2, PDF p. 44-46; 9 balls of 12.7 mm, pitch diameter
65 mm, grooves 0.52 d, clearance 15 um, steel 2e11 Pa) prints the equivalent radii of the inner and outer
contacts (Eqs (23)-(26)), their curvature sums and radius ratios, the Hamrock-Brewe ellipticity fits (Eq. (37))
and the compressions of the most loaded ball at F_max = 4513 N. The bearing is v003's literature system
(``hamrock_dgbb``, imported: OPEN R5-80); here its bridge contacts are compared with the printed contact values.
Hamrock's printed k_i (9.42; 32.35^(2/pi) = 9.15) and his inner radius ratio (32.35; 0.165/0.00511 = 32.29) do not
follow from his own numbers (OPEN R5-09): those rows are RECORD; the radii (three printed digits) are judged.

HamrockBrewe1983 Table 5 (ball on an outer ring, PDF p. 6) is v102's literature system and judged there (its
T5 rows); it is not repeated here.

The rows, with their pages, are in ``published/hamrock1983_sec14_2_contacts.csv`` (read from the page images,
2026-10-28). PDF page = printed page + 5.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import read_records_csv

# OPEN R5-10 / R2-02: LiteratureCase is defined in v101
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase
# OPEN R5-80: Hamrock's bearing as v003 builds it
from studies.v000_axisforge.v003_iso16281_load_zone.literature import hamrock_dgbb

__all__ = ["cases", "published", "PUBLISHED"]

PUBLISHED = Path(__file__).with_name("published") / "hamrock1983_sec14_2_contacts.csv"


def published() -> dict[str, dict]:
    """The printed rows by symbol; ``value`` kept as the printed string (its digits set the tolerance)."""
    return {r["symbol"]: r for r in read_records_csv(PUBLISHED, text_fields=("value", "note", "pdf_page"))}


def cases() -> list[LiteratureCase]:
    """The published system of v105: Hamrock 1983 sec. 14.2, the inner and outer contacts."""
    return [LiteratureCase("hamrock1983_sec14_2_contacts", hamrock_dgbb, list(published().values()),
                           "Hamrock1983 sec. 14.2, Eqs (19), (23)-(26), (37), PDF p. 44-46")]
