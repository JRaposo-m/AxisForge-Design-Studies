"""Comparator of v102: a solved elliptical BEM contact against r02 Hertz and Hamrock-Brewe (README 6).

:func:`check` adds to v101's comparison (approach, p0, semi-axes, against r02 at the BEM load) the
quantities of an ellipse: its ellipticity ``k = a/b``, its orientation measured on the BEM field,
the per-body size ratio of r02's validity (a / smallest radius of each body, recorded: MG-08), and,
when the published records of the system are given, Hamrock and Brewe's printed values.
"""
from __future__ import annotations

import math

import numpy as np

from studies.v100_dry_contact.v101_hertz_point import bem_load, reference
from studies.v100_dry_contact.v101_hertz_point import check as check_point

from .construction import CM, N_PER_CM2

__all__ = ["check", "orientation", "half_unit"]



def orientation(result) -> float:
    """Direction of the major axis of the BEM pressure field from bridge x [rad], in [-pi/2, pi/2).

    The principal direction of the largest second moment ``sum p (x, y)(x, y)^T`` over the grid
    (node coordinates on the grid axes, turned back by the grid angle); for a Hertz ellipse it is
    the direction of the major semi-axis ``a``.
    """
    g, p = result.grid, np.asarray(result.p)
    Sxx, Syy, Sxy = (float((p * u * v).sum()) for u, v in ((g.X, g.X), (g.Y, g.Y), (g.X, g.Y)))
    theta = g.angle + 0.5 * math.atan2(2.0 * Sxy, Sxx - Syy)
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def half_unit(printed: str) -> float:
    """Half a unit of the last printed digit of ``printed``, relative to its value [-] ("0.0250" -> 0.02)."""
    mantissa = printed.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(printed.lower().split("e")[1]) if "e" in printed.lower() else 0
    return 0.5 * 10.0 ** (exponent - decimals) / abs(float(printed))


def check(result, *, published: list[dict] | None = None, case: str = "v102") -> list[dict]:
    """Comparison records of one solved elliptical point contact.

    v101's rows (C1 approach 0.5 %, C2-C5, judged on the spatial kernel at ``b_H/h >= 16``), then:

    * E1 ``k = a/b`` (BEM measured) against r02: within ``2/M`` (C4 on both semi-axes);
    * E2 orientation of the major axis (:func:`orientation`) against r02 ``alpha_rad`` (mod pi):
      within ``h/a_H`` rad (one node at the end of the major semi-axis; the author's choice), judged
      as C1, RECORD for a circle (no direction);
    * V1 r02 per-body size ratio (largest semi-axis / smallest finite radius of each body) and its
      governing body: RECORD (the soft limit 0.1 of r02 sets ``hertz_in_range``, MG-08);
    * T1 when ``published`` holds a Table 1 ``k`` (values as printed strings): BEM ``a/b`` against it,
      ``2/M`` plus half a unit of the printed digit;
    * T5 rows when ``published`` holds a Table 5 column:
      ``2a``, ``2b`` against the larger / smaller of ``D_x``, ``D_y``, ``delta``, ``sigma_max`` against
      p0, ``k`` against a/b, the BEM values taken to the printed load by the Hertz exponents; tolerance the v101 tolerance of the quantity (0.5 %, ``1/M``,
      ``M^-3/2``; ``2/M`` for k) plus half a unit of the printed digit; and the same values from r02
      (RECORD: closed form vs print).
    """
    rows = check_point(result, case=case)
    c, g = result.contact, result.grid
    ref = reference(c, bem_load(result))
    judged = all(r["verdict"] != "RECORD" for r in rows if r["id"] == "C1")
    common = {k: rows[0][k] for k in ("case", "label", "kernel", "load_control", "h_m", "a_over_h", "b_over_h", "M",
                                      "span_x", "span_y", "n_x", "n_y", "converged", "edge_clear", "hertz_in_range")}
    M = common["M"]
    circle = math.isclose(c.kappa_max, c.kappa_min, rel_tol=1e-9)

    def row(cid, quantity, unit, measured, reference_, tol, rel=True, source="r02 hertz", force_record=False):
        err = (measured / reference_ - 1.0) if rel else measured - reference_
        ok = tol is not None and judged and not force_record
        return {**common, "id": cid, "quantity": quantity, "unit": unit, "measured": measured,
                "reference": reference_, "rel_err": err, "tol": math.nan if tol is None else tol,
                "verdict": ("PASS" if abs(err) <= tol else "MISS") if ok else "RECORD", "source": source}

    theta = orientation(result)
    d_theta = (theta - ref.alpha_rad + math.pi / 2) % math.pi - math.pi / 2
    rows += [row("E1", "k = a/b", "-", result.a / result.b, ref.a_m / ref.b_m, 2.0 / M),
             {**row("E2", "orientation of a", "rad", d_theta, 0.0, g.h / ref.a_m, rel=False, force_record=circle),
              "measured": theta, "reference": ref.alpha_rad},
             {**row("V1", f"size ratio, governing {ref.validity.governing}", "-", ref.validity.ratio, 0.1, None),
              "rel_err": math.nan}]
    p = {r["symbol"]: r["value"] for r in published or ()}
    if "k" in p and "D_x" not in p:                                       # a Table 1 row: printed k only
        rows.append(row("T1", "k = a/b BEM", "-", result.a / result.b, float(p["k"]), 2.0 / M + half_unit(p["k"]),
                        source="HamrockBrewe1983 Table 1, PDF p. 4"))
    if "D_x" in p:                                                        # a Table 5 column
        # the BEM values taken to the printed load by the Hertz exponents (sizes and p0 ~ P^(1/3), approach
        # ~ P^(2/3)); the BEM approach is the one that carries the load found (under interference control
        # the approach is an input, the load the output); r02 at the printed load
        F = float(p["F"])
        s = F / result.load_found
        delta = result.interference if result.load_control != "normal_load" else result.delta
        ref_F = reference(c, F)
        (D0, s0), (D1, s1) = sorted((float(p[k]), p[k]) for k in ("D_x", "D_y"))
        src = "HamrockBrewe1983 Table 5, PDF p. 6"
        for cid, quantity, unit, bem, r02, printed, scale, tol in (
                ("T5", "2b", "m", 2 * result.b * s ** (1 / 3), 2 * ref_F.b_m, D0, CM, 1.0 / M + half_unit(s0)),
                ("T5", "2a", "m", 2 * result.a * s ** (1 / 3), 2 * ref_F.a_m, D1, CM, 1.0 / M + half_unit(s1)),
                ("T5", "delta", "m", delta * s ** (2 / 3), ref_F.delta_m, float(p["delta"]), CM,
                 0.005 + half_unit(p["delta"])),
                ("T5", "sigma_max = p0", "Pa", result.p0 * s ** (1 / 3), ref_F.p0_Pa, float(p["sigma_max"]), N_PER_CM2,
                 M ** -1.5 + half_unit(p["sigma_max"])),
                ("T5", "k = a/b", "-", result.a / result.b, ref_F.a_m / ref_F.b_m, float(p["k"]), 1.0,
                 2.0 / M + half_unit(p["k"]))):
            k_published = printed if quantity != "k = a/b" else max(printed, 1.0 / printed)   # HB's k = D_y/D_x
            rows += [row(cid, f"{quantity} BEM", unit, bem, k_published * scale, tol, source=src),
                     row(cid, f"{quantity} r02", unit, r02, k_published * scale, None, source=src)]
    return rows
