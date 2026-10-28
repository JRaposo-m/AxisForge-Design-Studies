"""The model of s103: what the split of a fixed profile-shift sum does along the path of contact.

For one split x1 (x2 = x_sum - x1) :func:`split_point` gathers, from AxisForge and the bridge only (an s never calls
an r; the r04 / r02 values come through v107 ``check`` in ``study``):

* the path: T1A..T1E [mm] on the line of action (``GearPair`` attributes, ISO_21771_2007 5.4.5, Eqs (77)-(87), PDF
  p. 36-40), the transverse contact ratio eps_alpha (Eq. (90), PDF p. 41-44);
* specific sliding at the two roots, zeta_f1 at A and zeta_f2 at E (``GearPair.zeta_f1``, ``zeta_f2``, 5.6.3 Eqs
  (116)-(117), PDF p. 45-48), and at every point A..E (``pair_values_at`` through the bridge contacts);
* the transverse radii of curvature rho1_t, rho2_t and the Hertz line pressure p0 at A..E: bridge
  ``CylindricalGearContact.from_pair`` with the L0 load of D-0035, ``F_bn / l_sum`` from AxisForge ``load_sharing``
  (l_sum = the summed contact-line length at that instant, 5.4.8, PDF p. 41-44). Spur: share 1/2 in the
  double-contact zones, also AT B and D; so B and D are added with the full load (the single-contact side of those
  points, where the largest p0 of L0 sits by construction). Helical: the share follows the contact-line lengths and
  changes with the split (0.67 at B / D, 0.35-0.97 at C in ``s103_points.csv``), set by the fractional parts of
  eps_alpha and eps_beta, not by the curvature. The flank-end flags of ``hertz`` are kept;
* the tooth thickness at each tip, s_at (``gear_values_at(gear, d_a)["s_yt"]``, 4.7.1 Eq. (38), PDF p. 29; 7.9 says
  Eq. (38) "may be used to calculate the tooth thickness at the tip", PDF p. 63), as s_an / m_n;
* the pair as cut (``construction.as_cut``: AxisForge manufactured family, zero allowances, counterpart rack of the
  ISO 53 basic rack, ``# OPEN R5-132``): undercut of each gear (x < x_Emin, 7.7 Eq. (135)), active root d_Nf vs root
  form d_Ff (7.6 Eq. (128), PDF p. 61-62), and whether AxisForge's ``validate_pair`` of the cut pair is empty.

:func:`bounds` gives two ranges. The sweep range (ideal flanks): the path between T1 and T2 (T1A >= 0, T2E >= 0:
AxisForge ``validate_pair``, involute interference) and no pointed tip (s_at > 0; the ideal family accepts a pointed
tip, R5-133). The range as cut, inside it: also no undercut (x >= x_Emin) and the mate's tip above the fillet
(d_Nf >= d_Ff, i.e. T1A >= T1Ff1 and T2E >= T2Ff2, 5.4.1 Eq. (66), (67)). Splits between the two ranges are kept and
tagged (``made_valid`` False): ideal flanks a rack cutter would not make. Below x_Emin Eq. (128) does not hold (the
form circle is the intersection of involute and trochoid, 7.6 last sentence, PDF p. 62, not computed): the active
root of an undercut gear is not checked there. Neither 7.9 nor this case fixes a minimum tip thickness: reported,
not a bound (``# OPEN R5-134``).

Spur is judged; for the helical pair the contact is the mid-face slice (bridge, R-027) and its p0 is a RECORD: the
contact-line length along the path (R2-10) is s106's. The helical couple F_a r_w (J-16) does not enter: there is no
shaft here.

Units: AxisForge mm, N, N.m; the bridge converts to SI inside ``from_pair``; p0 reported in MPa.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from slippy_bridge.contact import CylindricalGearContact, hertz

from .construction import DRIVER, E_MPA, FAMILY, NU, PAIRS, as_cut, split_pair, x_sum

__all__ = ["split_point", "bounds", "tip_thickness"]


def tip_thickness(gear) -> float:
    """Normal tooth thickness at the tip over the normal module, s_an / m_n [-] (s_at cos(beta_a) / m_n).

    s_at from AxisForge ``gear_values_at(gear, d_a)`` (ISO_21771_2007 4.7.1 Eq. (38), PDF p. 29); the normal value
    with the helix angle at the tip, beta_a (Eq. (8)). Zero: a pointed tooth.
    """
    v = FAMILY.gear_values_at(gear, gear.da)
    return float(v["s_yt"] * math.cos(v["beta_y"]) / gear.mn)


def _fillet_margin(cut, i: int) -> float:
    """d_Nf - d_Ff of gear i of a pair as cut [mm]; nan when d_Ff is not computed (undercut: Eq. (128) invalid)."""
    d_Ff = getattr(cut, f"gear{i}").d_Ffs                    # = d_Ffi: zero allowances
    return math.nan if d_Ff is None else getattr(cut, f"d_Nf{i}") - d_Ff


def bounds(name: str, *, addendum_reduction: bool = False) -> dict:
    """Range of x1 at fixed x_sum where the ideal pair exists (sweep) and where it can be cut (inside it).

    Each condition is a function of x1 for the pair assembled without validation; its roots are found by a sign scan
    on a wide grid and ``brentq`` (xtol 1e-12). A root below FEUP's x1 bounds from below, one above from above (FEUP's
    split satisfies every condition). Ideal: ``T1A = 0``, ``T2E = 0`` (involute interference), ``s_a1 = 0``,
    ``s_a2 = 0`` (pointed tip), ``h_a1 = 0``, ``h_a2 = 0`` (tip circle on the reference circle, AxisForge
    ``validate_gear``; binds only with the tip alteration k). As cut, in addition: ``d_Nf1 = d_Ff1`` (wheel tip on the pinion fillet),
    ``d_Nf2 = d_Ff2`` (pinion tip on the wheel fillet), ``x1 = x_Emin1`` and ``x2 = x_Emin2`` (undercut, AxisForge
    ``x_Emin``). eps_alpha >= 1 is not a bound: ``validate_pair`` checks it at every split of the sweep (assembled with
    ``check=True``).

    Returns
    -------
    dict
        ``x1_lo``, ``lo_by``, ``x1_hi``, ``hi_by`` (sweep, ideal), ``x1_lo_cut``, ``lo_cut_by``, ``x1_hi_cut``,
        ``hi_cut_by`` (as cut), ``x_Emin`` (pinion, wheel) [-], ``roots`` (condition -> list of x1).
    """
    xs = x_sum(name)

    def pair(x):
        return split_pair(name, x, addendum_reduction=addendum_reduction, check=False)

    ideal = {"T1A = 0": lambda x: pair(x).T1A,
             "T2E = 0": lambda x: pair(x).T2E,
             "s_a1 = 0": lambda x: tip_thickness(pair(x).gear1),
             "s_a2 = 0": lambda x: tip_thickness(pair(x).gear2),
             # AxisForge validate_gear: tip circle outside the reference circle (addendum (1 + x + k) m_n > 0)
             "h_a1 = 0": lambda x: pair(x).gear1.da - pair(x).gear1.d,
             "h_a2 = 0": lambda x: pair(x).gear2.da - pair(x).gear2.d}
    fillet = {"d_Nf1 = d_Ff1": lambda x: _fillet_margin(as_cut(pair(x)), 1),
              "d_Nf2 = d_Ff2": lambda x: _fillet_margin(as_cut(pair(x)), 2)}
    roots = {}
    grid = np.linspace(-1.2, 1.6, 57)                                 # wide bracket around the FZG sums
    for label, f in {**ideal, **fillet}.items():
        vals = [f(x) for x in grid]
        # nan (no d_Ff under undercut) never changes sign: those intervals are skipped
        roots[label] = [brentq(f, a, b, xtol=1e-12)
                        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]) if fa * fb < 0.0]
    cut = as_cut(split_pair(name, addendum_reduction=addendum_reduction))
    x_Emin = (cut.gear1.x_Emin, cut.gear2.x_Emin)
    roots["x1 = x_Emin1"], roots["x2 = x_Emin2"] = [x_Emin[0]], [xs - x_Emin[1]]
    x_feup = PAIRS[name]["x"][0]

    def ends(labels):
        found = [(x, k) for k in labels for x in roots[k]]
        return (max((r for r in found if r[0] < x_feup), default=(-math.inf, "none")),
                min((r for r in found if r[0] > x_feup), default=(math.inf, "none")))

    (lo, lo_by), (hi, hi_by) = ends(ideal)
    (lo_c, lo_c_by), (hi_c, hi_c_by) = ends(roots)
    return {"x1_lo": lo, "lo_by": lo_by, "x1_hi": hi, "hi_by": hi_by, "x1_lo_cut": lo_c, "lo_cut_by": lo_c_by,
            "x1_hi_cut": hi_c, "hi_cut_by": hi_c_by, "x_Emin": x_Emin, "roots": roots}


def split_point(pair, wheel_torque_Nm: float) -> tuple[dict, list[dict]]:
    """The quantities of one split: a summary record and one record per point A..E (plus B, D at full load).

    Parameters
    ----------
    pair : GearPair
        An FZG pair of :func:`construction.split_pair` (ideal family).
    wheel_torque_Nm : float
        Wheel torque [N.m] (R-124).

    Returns
    -------
    (dict, list of dict)
        Summary: ``x1``, ``x2``, ``k`` (tip alteration [-]), ``d_a1_mm``, ``d_a2_mm``, ``T1A_mm`` .. ``T1E_mm``,
        ``eps_alpha``, ``zeta_f1``, ``zeta_f2``, ``s_an1``, ``s_an2`` [m_n], ``F_bn_N``, ``p0_max_MPa`` (largest over
        the points where ``hertz`` is valid; nan if none), ``p0_max_at``, ``C_on_path`` (False: the path A..E lies on
        one side of the pitch point; C is then not evaluated), and as cut: ``d_Nf1_mm``, ``d_Ff1_mm``, ``d_Nf2_mm``,
        ``d_Ff2_mm`` (nan under undercut), ``undercut_pinion``, ``undercut_wheel``, ``fillet_pinion``,
        ``fillet_wheel`` (d_Nf < d_Ff), ``made_valid`` (AxisForge manufactured ``validate_pair`` empty). Points:
        ``point`` (A..E, "B full", "D full"), ``g_y_mm``, ``rho1_t_mm``, ``rho2_t_mm``, ``zeta1``, ``zeta2``,
        ``share`` (L0 share of ``load_sharing``; nan for the full-load B, D), ``w_N_per_mm``, ``p0_MPa``, ``b_H_mm``,
        ``hertz_valid``, ``hertz_reason``.
    """
    # normal tooth force of AxisForge mesh_forces for a WHEEL torque: gear 1 gets T2 z1 / z2 (R-124), pinion drives
    T1 = -wheel_torque_Nm * pair.gear1.z / pair.gear2.z
    F_bn = float(FAMILY.mesh_forces(pair, T1, -1 if T1 < 0 else 1).F_bn_N)
    E, nu = (E_MPA, E_MPA), (NU, NU)
    # the points A..E as the bridge's points="path" builds them (dy1 from T1C + g_y, ISO 21771 Eq. (17) inverted; copy
    # of slippy_bridge/contact/core/machine_elements/gears.py:533), less C when it lies off the path: a strongly
    # unequal split moves the whole path A..E to one side of C, and points="path" then raises (remove with R5-135)
    path = FAMILY.path_points(pair, DRIVER)
    c_on_path = bool(pair.T1A <= pair.T1C <= pair.T1E)
    dy1 = {k: 2.0 * math.hypot(pair.T1C + g, pair.gear1.rb) for k, g in path.items() if k != "C" or c_on_path}
    contacts = CylindricalGearContact.from_pair(pair, points=dy1, driver=DRIVER, F_bn_N=F_bn, E=E, nu=nu)
    if pair.epslon_beta == 0.0:
        # B and D on their single-contact side: the full F_bn on the one line of length b (L0 share 1)
        dy1 = {f"{k} full": dy1[k] for k in ("B", "D")}
        contacts += CylindricalGearContact.from_pair(pair, points=dy1, driver=DRIVER, load=F_bn / (pair.b * 1e-3),
                                                     E=E, nu=nu)
    points = []
    for c in contacts:
        h = hertz(c)
        points.append({"point": c.point, "g_y_mm": c.g_y_mm, "rho1_t_mm": c.body1.rho_t * 1e3,
                       "rho2_t_mm": c.body2.rho_t * 1e3, "zeta1": c.zeta1, "zeta2": c.zeta2, "share": c.share,
                       "w_N_per_mm": c.load * 1e-3, "p0_MPa": h.p0 * 1e-6, "b_H_mm": h.b * 1e3,
                       "hertz_valid": h.valid, "hertz_reason": h.reason})
    # largest p0 over the points where Hertz holds (A and E sit on a tip edge: s104's question, flagged by hertz)
    top = max((r for r in points if r["hertz_valid"]), key=lambda r: r["p0_MPa"],
              default={"p0_MPa": math.nan, "point": "none valid"})
    cut = as_cut(pair)
    d_Ff = [math.nan if g.d_Ffs is None else g.d_Ffs for g in (cut.gear1, cut.gear2)]
    summary = {"x1": pair.x1, "x2": pair.x2, "k": pair.gear1.data.k, "d_a1_mm": pair.gear1.da,
               "d_a2_mm": pair.gear2.da, **{f"T1{k}_mm": getattr(pair, f"T1{k}") for k in "ABCDE"},
               "eps_alpha": pair.epslon_alpha, "zeta_f1": pair.zeta_f1, "zeta_f2": pair.zeta_f2,
               "s_an1": tip_thickness(pair.gear1), "s_an2": tip_thickness(pair.gear2), "F_bn_N": F_bn,
               "p0_max_MPa": top["p0_MPa"], "p0_max_at": top["point"], "x_sum": x_sum(pair.label),
               "C_on_path": c_on_path, "d_Nf1_mm": cut.d_Nf1, "d_Ff1_mm": d_Ff[0], "d_Nf2_mm": cut.d_Nf2,
               "d_Ff2_mm": d_Ff[1], "undercut_pinion": bool(cut.x1 < cut.gear1.x_Emin),
               "undercut_wheel": bool(cut.x2 < cut.gear2.x_Emin),
               # nan < 0 is False: no fillet flag where d_Ff is not computed (undercut)
               "fillet_pinion": bool(_fillet_margin(cut, 1) < 0.0), "fillet_wheel": bool(_fillet_margin(cut, 2) < 0.0),
               "made_valid": not cut.validate()}
    return summary, points
