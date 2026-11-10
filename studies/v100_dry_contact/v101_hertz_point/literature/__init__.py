"""The published systems v101 knows (D-0036 amendment item 4): Johnson's sphere on a flat.

Johnson (1985) prints no worked number for a sphere on a flat: what he prints are the closed forms
(4.19)-(4.24) and their constants, ``p0 = (3/2) p_m`` (text after Eq. (4.21), PDF p. 103) and
``delta = a^2 / R`` (Eq. (4.23), PDF p. 103), which ``check`` evaluates on the BEM field (L1, L2).
The rows, with their pages, are in ``published/johnson1985_ch4.csv`` (vault extraction of
2026-10-16, through the tag ``studies-pre-sweep``; checked against the page images by the
librarian, 2026-10-28).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from axisforge.outputs._format import read_records_csv

from ..construction import PUBLISHED, ball_on_flat

__all__ = ["LiteratureCase", "cases"]


# OPEN R2-02: defined here, imported by the other v cases of level 100 until a common place is ruled
class LiteratureCase(NamedTuple):
    """One published system: its name, its builder (unsolved), the published records, the source."""

    name: str
    build: Callable
    published: list[dict]
    source: str


def cases() -> list[LiteratureCase]:
    """The published systems of v101 (one: Johnson 1985 section 4.2(a), the ball of construction on a flat)."""
    return [LiteratureCase("johnson1985_sphere_on_flat", ball_on_flat,
                           read_records_csv(PUBLISHED / "johnson1985_ch4.csv"),
                           "Johnson1985 section 4.2(a), Eqs (4.19)-(4.24), PDF p. 102-103")]
