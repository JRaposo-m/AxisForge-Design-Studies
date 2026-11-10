"""The published systems v102 knows: Hamrock and Brewe (1983) Tables 1 and 5.

* Table 1 (PDF p. 4): the exact ellipticity ``k`` (Hamrock-Anderson 1973 numerics) against
  ``alpha = R_y/R_x``; v102's crossed cylinders are built at Table 1 alphas and ``check`` compares the BEM
  ``a/b`` with the printed ``k`` (T1; r02's tests already hold Tables 1-2 against hertz_full).
* Table 5 (PDF p. 6): three worked contacts (wheel on rail, ball on plane, ball on the outer ring) with
  printed contact diameters, approach and maximum stress.

The rows, values kept as printed strings, are in ``published/hamrock_brewe_1983_t1_t5.csv`` (vault
extraction of 2026-10-16 through the tag ``studies-pre-sweep``; Table 1 as in r02's
``published/hamrock_brewe_1983_t1_t2.csv``; the note of the two ball load rows corrected: 222.4111 N is
printed, librarian 2026-10-28). OPEN R2-07: the wheel-on-rail column of Table 5 is inconsistent in
the paper: its printed D_x, D_y, delta and sigma_max are one Hertz contact of the printed F, E' and k but
of R = 31.41 cm, while its printed radii give R = 18.78 cm (1/R = 1/R_x + 1/R_y); a source error
(librarian and physics-reviewer, page image, 2026-10-28). Judged as printed; the miss is recorded.
"""
from __future__ import annotations

from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import ALPHAS, crossed_cylinders, published_rows, table5

__all__ = ["cases"]


def cases() -> list[LiteratureCase]:
    """Table 1 at the sweep alphas (one case per alpha) and the three Table 5 contacts."""
    rows = published_rows()
    out = []
    for alpha in ALPHAS:
        mine = [r for r in rows if r["location"] == "Table 1" and float(r["condition"].split(";")[0].removeprefix("alpha=")) == alpha]
        out.append(LiteratureCase(f"hamrock_brewe_t1_alpha{alpha:g}", lambda a=alpha: crossed_cylinders(a), mine,
                                  "HamrockBrewe1983 Table 1, PDF p. 4"))
    for app, contact in table5().items():
        mine = [r for r in rows if r["location"] == "Table 5" and r["condition"].startswith(f"application={app};")]
        out.append(LiteratureCase(f"hamrock_brewe_t5_{contact.label}", lambda c=contact: c, mine,
                                  "HamrockBrewe1983 Table 5, PDF p. 6"))
    return out
