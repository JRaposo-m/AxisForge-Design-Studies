"""The systems of v102, returned NOT solved: elliptical contacts of two bodies at an angle, and the
three worked contacts of Hamrock and Brewe (1983) Table 5.

* :func:`crossed_cylinders` - two equal steel cylinders of radius ``R`` whose axes cross at
  ``theta``: the relative curvatures are ``(1 +- cos theta)/R``, so ``alpha = R_y/R_x =
  (1 + cos theta)/(1 - cos theta)`` sets ``theta`` (Johnson 1985, section 4.1, Eqs (4.4)-(4.5), PDF p. 96).
  Each body is given with its own radii and the angle between them, as r02 takes them; the
  ellipse lies at ``psi`` from bridge x (neither body's axis).
* :func:`table5` - wheel on rail, ball on plane, ball on the outer ring (concave groove,
  ``R_x = -38.9 mm``, ``R_y = -6.6 mm``) of Table 5 (PDF p. 6), from ``literature/published``.

SI units (m, Pa, N). Hamrock and Brewe print cm, N and ``E' = 2.197e7 N/cm^2`` for both bodies;
converted here, in this one place: lengths x 1e-2, stresses x 1e4, and ``E = E' (1 - nu^2)`` with
``nu = 0.3`` (only ``E'`` enters Hertz; ``E' = 2 E*`` for two equal bodies).
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.outputs._format import read_records_csv
from slippy_bridge.contact import SweptCircleBody

from studies.v100_dry_contact.v101_hertz_point import PointContact
from studies.v100_dry_contact.v101_hertz_point.construction import E_STEEL, NU_STEEL

__all__ = ["crossed_cylinders", "table5", "published_rows", "ALPHAS", "R_CYLINDER", "LOAD_CYLINDERS", "PUBLISHED",
           "CM", "N_PER_CM2"]

PUBLISHED = Path(__file__).with_name("literature") / "published" / "hamrock_brewe_1983_t1_t5.csv"
ALPHAS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)   # alpha = R_y/R_x of the sweep: rows of Table 1 [-]
R_CYLINDER = 10e-3       # [m]
LOAD_CYLINDERS = 50.0    # [N]: the largest semi-axis stays below 0.1 R up to alpha 100 (a/R 0.088)
CM, N_PER_CM2 = 1e-2, 1e4                            # cm -> m, N/cm^2 -> Pa (the one place)
_NU_HB = 0.3             # Poisson's ratio taken for Table 5 (only E' printed)


def crossed_cylinders(alpha: float, *, R: float = R_CYLINDER, load: float = LOAD_CYLINDERS) -> PointContact:
    """Two steel cylinders of radius ``R`` [m] crossed so that ``R_y/R_x = alpha``, unsolved.

    Body 1 is curved along bridge x (axis along y); body 2 the same cylinder turned by
    ``theta = arccos((alpha - 1)/(alpha + 1))`` (90 deg at alpha 1: a sphere of radius R on a flat).
    """
    theta = math.acos((alpha - 1.0) / (alpha + 1.0))
    body = {"r": (R, math.inf), "E": E_STEEL, "nu": NU_STEEL}
    return PointContact(body1=SweptCircleBody(**body), body2=SweptCircleBody(**body), load=load, angle=theta,
                        label=f"crossed_cylinders_alpha{alpha:g}")


def published_rows() -> list[dict]:
    """Rows of ``hamrock_brewe_1983_t1_t5.csv``, ``value`` kept as the printed string (its digits set tolerances)."""
    return read_records_csv(PUBLISHED, text_fields=("value",))


def table5() -> dict[str, PointContact]:
    """The three Table 5 contacts by application name, unsolved (an empty radius is printed infinity)."""
    rows = [r for r in published_rows() if r["location"] == "Table 5"]
    E_prime = float(next(r["value"] for r in rows if r["symbol"] == "E'")) * N_PER_CM2
    E = E_prime * (1.0 - _NU_HB**2)
    out = {}
    for app in dict.fromkeys(r["condition"].split(";")[0].removeprefix("application=") for r in rows
                             if r["condition"].startswith("application=")):
        mine = {r["symbol"]: r["value"] for r in rows if r["condition"].startswith(f"application={app};")}

        def radius(symbol):
            return math.inf if mine[symbol] == "" else float(mine[symbol]) * CM

        out[app] = PointContact(body1=SweptCircleBody(r=(radius("r_ax"), radius("r_ay")), E=E, nu=_NU_HB),
                                body2=SweptCircleBody(r=(radius("r_bx"), radius("r_by")), E=E, nu=_NU_HB),
                                load=float(mine["F"]), label=app.replace(" ", "_").replace("-", "_"))
    return out
