"""The published grid studies v104 knows: Polonsky and Keer (1999) and Muser (2014).

Neither prints a table to digitise: Polonsky and Keer state bounds on their error measures for a
Hertz contact on square grids of M = 64-1024 rows slightly larger than the contact diameter (PDF
p. 10, Fig. 1 on p. 9, plotted only); Muser states a resolution rule for a periodic GFMD cell (PDF
p. 7). ``check.judge`` judges those statements on v104's check records. The rows, with pages, are
in ``published/polonsky_keer_1999_muser_2014.csv`` (vault extraction of 2026-10-16, via the tag
``studies-pre-sweep``).
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import read_records_csv

from studies.v100_dry_contact.v101_hertz_point import ball_on_flat
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

__all__ = ["cases", "MUSER_B_OVER_H", "MUSER_SPAN"]

PUBLISHED = Path(__file__).with_name("published") / "polonsky_keer_1999_muser_2014.csv"
MUSER_B_OVER_H = 32.0    # Delta_a / a_c = 1/32 (h rounded down to 2**k: b/h in [32, 64))
# a_c / L = 1/8 with L Muser's periodic cell: SlipPY's frequency kernel repeats the loads every 2 n h, twice the
# grid, so half-length 2a gives the period 8a; the spatial kernel has no cell (half-length 4a, the grid L = 8a: an
# analogy, RECORD)
MUSER_SPAN = {"frequency": 2.0, "spatial": 4.0}


def cases() -> list[LiteratureCase]:
    """Both grid studies use a Hertz sphere: v101's ball on a flat is their system here."""
    rows = read_records_csv(PUBLISHED)
    return [LiteratureCase("polonsky_keer_1999_hertz_grid", ball_on_flat,
                           [r for r in rows if r["source_key"] == "PolonskyKeer1999"],
                           "PolonskyKeer1999 Sec. 4.1, PDF p. 9-10"),
            LiteratureCase("muser_2014_grid_rule", ball_on_flat, [r for r in rows if r["source_key"] == "Muser2014"],
                           "Muser2014 Numerical analysis, PDF p. 7")]
