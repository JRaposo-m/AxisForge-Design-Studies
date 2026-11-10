"""The published system v103 knows: Johnson's two cylinders in plane-strain contact (section 4.2(c)).

Johnson prints the closed forms (4.43)-(4.45) (PDF p. 111) and the constant ``p0 = (4/pi) p_m``, which
``check`` evaluates on the BEM field (L1). Rows with pages: ``published/johnson1985_line.csv`` (vault
extraction of 2026-10-16 through the tag ``studies-pre-sweep``; the 4/pi row added from Eq. (4.45)).
"""
from __future__ import annotations

from axisforge.outputs._format import read_records_csv

from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import PUBLISHED, rollers

__all__ = ["cases"]


def cases() -> list[LiteratureCase]:
    """One case: two steel rollers (10 and 20 mm) at 1e5 N/m."""
    return [LiteratureCase("johnson1985_two_cylinders", rollers, read_records_csv(PUBLISHED / "johnson1985_line.csv"),
                           "Johnson1985 section 4.2(c), Eqs (4.43)-(4.45), PDF p. 110-111")]
