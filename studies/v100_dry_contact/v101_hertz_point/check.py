"""Comparator of v101: a solved BEM point contact against r02 Hertz and Johnson's constants (README 6).

:func:`check` takes any solved point contact (a ``slippy_bridge.contact.StaticResult``: the system
after ``construction -> solve``, from this case's literature or from a study) and returns one
comparison record per quantity. The reference is r02 :func:`hertz` of the same two bodies, as
given (radii, E, nu, angle), at the load the BEM carries: the nominal load under load control, the
load found under interference control. r02 and the bridge's own Hertz both run SlipPY
``hertz_full``: this is the BEM against SlipPY's Hertz (verified by José, rule 13).
"""
from __future__ import annotations

import math

import numpy as np
from axisforge.outputs._format import read_records_csv

from studies.r00_analytical_references.r02_hertz_contact import hertz

from .construction import PUBLISHED

__all__ = ["check", "reference", "bem_load", "pressure_cuts", "pressure_l1_error", "TOL_DELTA", "B_OVER_H_JUDGED"]

TOL_DELTA = 0.005          # [-] approach, spatial kernel, b/h >= 16 (v_prerequisites row 1)
B_OVER_H_JUDGED = 16.0     # [-] minor semi-axis over spacing from which the criteria hold
_P0_OVER_PM = float(next(r["value"] for r in read_records_csv(PUBLISHED / "johnson1985_ch4.csv")
                         if r["symbol"] == "p0/p_m"))     # 3/2, Johnson 1985 text after Eq. (4.21), PDF p. 103


def reference(contact, load: float):
    """r02 Hertz of the two bodies of ``contact`` as given, at ``load`` [N] (a line: [N/m])."""
    c = contact
    return hertz(c.body1.r, c.body2.r, (c.body1.E, c.body2.E), (c.body1.nu, c.body2.nu), load, angle_rad=c.angle,
                 line=c.line)


# OPEN R2-04: the bridge applies the same rule for StaticResult.hertz (solvers/static.py:160-166); a bridge
# property (StaticResult.reference_load) would keep it in one place
def bem_load(result) -> float:
    """The load the BEM carries [N] (a line: [N/m]): ``contact.load`` under load control, the load
    found under interference (or mean-gap) control (a line: over the period)."""
    c = result.contact
    if result.load_control == "normal_load":
        return c.load
    return result.load_found / result.period if c.line else result.load_found


def check(result, *, case: str = "v101") -> list[dict]:
    """Comparison records of one solved BEM point contact against r02 Hertz.

    Quantities and tolerances (README section 6, fixed before the run; ``M = 2 b_H / h``):

    * C1 ``delta`` (approach, SlipPY ``interference``) [m]: within ``TOL_DELTA`` = 0.5 %. Under
      interference control the given approach against Hertz at the load found;
    * C2 ``load`` [N] (interference control): the load found against the Hertz load at the given
      approach, ``P_H = P_found (delta / delta_H)^(3/2)`` (Johnson Eq. (4.23)); RECORD: the same
      measurement as C1 read the other way;
    * C3 ``p0`` [Pa]: within ``M^(-3/2)`` (the author's choice after Polonsky and Keer's eps_p
      scaling, PDF p. 10, applied to the peak with unit constant);
    * C4 ``a``, ``b`` [m] (major, minor semi-axes): within ``1/M`` (Polonsky and Keer's eps_r bound
      for a node count, applied to the sub-node edge of the bridge: loose);
    * C5 ``eps_p`` [-]: Polonsky and Keer's L1 pressure error over the loaded nodes
      (:func:`pressure_l1_error`), RECORD;
    * L1 ``p0/p_m`` against Johnson's printed 3/2 (``p_m = P/(pi a b)``), L2 ``delta R/(a b)``
      against 1 (Eq. (4.23)); circular contacts only, RECORD (implied by C1-C4).

    A quantity is judged (``PASS`` / ``MISS``) only on the spatial kernel at ``b_H/h >= 16``, with a
    converged solve, no pressure on the domain edge, the load balanced under load control and the
    Hertz reference in its validity range; otherwise ``RECORD``.

    Parameters
    ----------
    result : slippy_bridge.contact.StaticResult
        A solved point contact (``contact.line`` False).
    case : str, optional
        Case id written in the records (an s passes its own).

    Returns
    -------
    list of dict
        One record per quantity: ``case``, ``label``, ``id``, ``quantity``, ``unit``, ``measured``,
        ``reference``, ``rel_err``, ``tol``, ``verdict``, ``source``, and the grid and solver of the
        solve (``kernel``, ``load_control``, ``h_m``, ``a_over_h``, ``b_over_h``, ``M``, ``span_x``,
        ``span_y``, ``n_x``, ``n_y``, ``converged``, ``edge_clear``, ``hertz_in_range``).
    """
    c, g = result.contact, result.grid
    if c.line:
        raise ValueError("v101.check takes a point contact; a line contact is v103's")
    ref = reference(c, bem_load(result))
    h = g.h
    M = 2.0 * ref.b_m / h
    judged = (not result.use_frequency_domain and ref.b_m / h >= B_OVER_H_JUDGED and result.converged
              and result.edge_clear and result.load_ok and ref.validity.in_range)
    span_x, span_y = g.spans(result.hertz)
    common = {"case": case, "label": c.label, "kernel": "frequency" if result.use_frequency_domain else "spatial",
              "load_control": result.load_control, "h_m": h, "a_over_h": ref.a_m / h, "b_over_h": ref.b_m / h,
              "M": M, "span_x": span_x, "span_y": span_y, "n_x": g.X.shape[0], "n_y": g.X.shape[1],
              "converged": result.converged, "edge_clear": result.edge_clear,
              "hertz_in_range": ref.validity.in_range}

    def row(cid, quantity, unit, measured, reference_, tol, source="r02 hertz"):
        err = measured / reference_ - 1.0 if reference_ else measured
        verdict = "RECORD" if not judged or tol is None else ("PASS" if abs(err) <= tol else "MISS")
        return {**common, "id": cid, "quantity": quantity, "unit": unit, "measured": measured,
                "reference": reference_, "rel_err": err, "tol": math.nan if tol is None else tol,
                "verdict": verdict, "source": source}

    delta = result.delta if result.load_control == "normal_load" else result.interference
    rows = [row("C1", "delta", "m", delta, ref.delta_m, TOL_DELTA)]
    if result.load_control != "normal_load":
        rows.append(row("C2", "load", "N", result.load_found, result.load_found * (delta / ref.delta_m) ** 1.5, None))
    rows += [row("C3", "p0", "Pa", result.p0, ref.p0_Pa, M ** -1.5),
             row("C4", "a", "m", result.a, ref.a_m, 1.0 / M),
             row("C4", "b", "m", result.b, ref.b_m, 1.0 / M),
             row("C5", "eps_p", "-", pressure_l1_error(result, ref), 0.0, None, "PolonskyKeer1999 PDF p. 10")]
    if math.isclose(c.kappa_max, c.kappa_min, rel_tol=1e-9):          # Johnson section 4.2(a): circles
        R = 1.0 / c.kappa_max                                            # 1/R = 1/R1 + 1/R2 [m]
        p_m = result.load_found / (math.pi * result.a * result.b)
        rows += [row("L1", "p0/p_m", "-", result.p0 / p_m, _P0_OVER_PM, None,
                     "Johnson1985 text after Eq. (4.21), PDF p. 103"),
                 row("L2", "delta R/(a b)", "-", delta * R / (result.a * result.b), 1.0, None,
                     "Johnson1985 Eq. (4.23), PDF p. 103")]
    return rows


def pressure_l1_error(result, ref=None) -> float:
    """Polonsky and Keer's pressure error ``sum |p - p_H| / sum p_H`` over the loaded nodes [-].

    ``p_H = p0_H sqrt(1 - x^2/b_H^2 - y^2/a_H^2)`` at each node (Polonsky and Keer 1999, Sec. 4.1,
    PDF p. 10, written for the ellipse); loaded = above the bridge's numerical zero, 1e-6 p0. Needs
    the ellipse axes on the grid lines (as :func:`pressure_cuts`).
    """
    c, g = result.contact, result.grid
    if not math.isclose(c.psi, g.angle, abs_tol=1e-12):
        raise ValueError(f"needs the ellipse axes on the grid (psi {c.psi} = grid angle {g.angle})")
    ref = reference(c, bem_load(result)) if ref is None else ref
    p = np.asarray(result.p)
    loaded = p > 1e-6 * result.p0
    X, Y = g.X[loaded], g.Y[loaded]
    p_H = ref.p0_Pa * np.sqrt(np.clip(1.0 - (X / ref.b_m) ** 2 - (Y / ref.a_m) ** 2, 0.0, None))
    return float(np.abs(p[loaded] - p_H).sum() / p_H.sum())


def pressure_cuts(result, ref=None) -> list[dict]:
    """BEM pressure on the two principal cuts through the grid centre node, scaled by Hertz.

    Needs the ellipse axes on the grid lines (``contact.psi == grid.angle``: any contact on a
    ``Grid.around(..., angle="principal")`` grid, or ``psi = 0`` on the bridge axes): the minor axis
    ``b`` along grid x (array axis 0), the major ``a`` along grid y. A line contact gives the cut
    across its line only.

    Returns
    -------
    list of dict
        ``axis`` ("b (minor)" or "a (major)"), ``xi`` = coordinate / Hertz semi-axis of the cut [-],
        ``p_over_p0_ref`` = p / p0_H [-] (records of ``outputs...plots.contact.pressure_cut``).
    """
    c, g = result.contact, result.grid
    if not math.isclose(c.psi, g.angle, abs_tol=1e-12):
        raise ValueError(f"pressure_cuts needs the ellipse axes on the grid (psi {c.psi} = grid angle {g.angle})")
    ref = reference(c, bem_load(result)) if ref is None else ref
    i0, j0 = g.X.shape[0] // 2, g.X.shape[1] // 2
    p = np.asarray(result.p) / ref.p0_Pa
    out = [{"axis": "b (minor)", "xi": float(x) / ref.b_m, "p_over_p0_ref": float(v)}
           for x, v in zip(g.X[:, j0], p[:, j0])]
    if not c.line:
        out += [{"axis": "a (major)", "xi": float(y) / ref.a_m, "p_over_p0_ref": float(v)}
                for y, v in zip(g.Y[i0, :], p[i0, :])]
    return [r for r in out if abs(r["xi"]) <= 1.3]
