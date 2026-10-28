"""Comparator of v103: a solved BEM line contact (periodic strip) against r02 Hertz line (README 6).

The reference is r02 ``hertz(..., line=True)`` of the two bodies as given, at the load per unit length
the BEM carries (v101 ``reference``, ``bem_load``). The approach of a line contact is not defined by
the local contact (Johnson 1985, section 4.2(c), PDF p. 110): r02 gives none and it is not compared.
"""
from __future__ import annotations

import math

from axisforge.outputs._format import read_records_csv

from studies.v100_dry_contact.v101_hertz_point import bem_load, pressure_l1_error, reference

from .construction import PUBLISHED

__all__ = ["check", "B_OVER_H_JUDGED", "SPAN_X_JUDGED"]

B_OVER_H_JUDGED = 16.0     # [-] half-width over spacing from which the criteria hold (v104 rule)
SPAN_X_JUDGED = 4.0        # [-] domain half-length in half-widths: the frequency kernel images (SPAN_X_0)
_P0_OVER_PM = float(next(r["value"] for r in read_records_csv(PUBLISHED / "johnson1985_line.csv")
                         if r["symbol"] == "p0/p_m"))       # 4/pi, Johnson 1985 Eq. (4.45), PDF p. 111


def check(result, *, case: str = "v103") -> list[dict]:
    """Comparison records of one solved line contact against r02 Hertz line.

    * C3 ``p0`` within ``M^(-3/2)``, C4 ``b`` (half-width) within ``1/M``, ``M = 2 b_H/h`` (as v101);
    * C5 Polonsky-Keer ``eps_p`` over the loaded nodes against ``p_H(x)`` (Eq. (4.44)), RECORD;
    * C6 load per unit length carried (``sum p h^2 / period``) against the nominal, RECORD (a balance);
    * L1 ``p0/p_m`` against Johnson's printed ``4/pi`` (Eq. (4.45)), ``p_m = w/(2b)`` from the BEM, RECORD.

    Judged on a line solved with the frequency kernel (the only one the bridge allows for a line) at
    ``b_H/h >= 16`` and a domain half-length ``>= 4 b`` (its images along x), converged, edge clear,
    load balanced, Hertz reference in range; otherwise RECORD.
    """
    c, g = result.contact, result.grid
    if not c.line:
        raise ValueError("v103.check takes a line contact; a point contact is v101 / v102")
    ref = reference(c, bem_load(result))
    h = g.h
    M = 2.0 * ref.b_m / h
    span_x, _ = g.spans(result.hertz)
    judged = (ref.b_m / h >= B_OVER_H_JUDGED and span_x >= SPAN_X_JUDGED and result.converged and result.edge_clear
              and result.load_ok and ref.validity.in_range)
    common = {"case": case, "label": c.label, "kernel": "frequency" if result.use_frequency_domain else "spatial",
              "load_control": result.load_control, "h_m": h, "b_over_h": ref.b_m / h, "M": M, "span_x": span_x,
              "n_x": g.X.shape[0], "n_y": g.X.shape[1], "period_m": result.period, "converged": result.converged,
              "edge_clear": result.edge_clear, "hertz_in_range": ref.validity.in_range}

    def row(cid, quantity, unit, measured, reference_, tol, source="r02 hertz (line)"):
        err = measured / reference_ - 1.0 if reference_ else measured
        verdict = "RECORD" if not judged or tol is None else ("PASS" if abs(err) <= tol else "MISS")
        return {**common, "id": cid, "quantity": quantity, "unit": unit, "measured": measured,
                "reference": reference_, "rel_err": err, "tol": math.nan if tol is None else tol,
                "verdict": verdict, "source": source}

    w = result.load / result.period                                   # [N/m] carried by the BEM
    return [row("C3", "p0", "Pa", result.p0, ref.p0_Pa, M ** -1.5),
            row("C4", "b", "m", result.b, ref.b_m, 1.0 / M),
            row("C5", "eps_p", "-", pressure_l1_error(result, ref), 0.0, None, "PolonskyKeer1999 PDF p. 10"),
            row("C6", "w", "N/m", w, c.load, None),
            row("L1", "p0/p_m", "-", result.p0 / (w / (2.0 * result.b)), _P0_OVER_PM, None,
                "Johnson1985 Eq. (4.45), PDF p. 111")]
