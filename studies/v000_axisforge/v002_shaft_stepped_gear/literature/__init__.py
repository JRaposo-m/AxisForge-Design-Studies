"""The published system v002 knows (D-0036 amendment item 4): Harris and Kotzalas (2007) vol. 2, section 9.2.2.

Harris2007_vol2 sec. 9.2.2, Fig. 9.3 (PDF p. 278): a uniform shaft of bending stiffness EI between two
bearings at x = 0 and x = l, transverse forces P^k and couples T^k at x = a^k, bearing deflections
delta_r1, delta_r2 and bearing slopes theta_1, theta_2. Integrating EI y'' = -M (Eq. 9.8) twice on
each side of the load and matching at x = a gives the bearing forces F_1, F_2 and moments M_1, M_2
(Eqs 9.20-9.23, PDF p. 280; superposition for n loads, Eqs 9.24-9.27, PDF p. 281). For pin supports
(M_1 = M_2 = 0, Harris's own reading, PDF p. 281) Eqs 9.25 and 9.27 fix theta_1, theta_2 and Eqs 9.24,
9.26 return the statically determinate reactions, whatever the settlements.

No worked numbers are printed in the section (p. 281 prints only "See Example 9.3."; the worked
examples are recorded as being on the book's CD-ROM, J-14, not checked on these pages): the judge is
the printed equations, evaluated with the FEM's own bearing slopes and the prescribed settlements
(L1-L5 of the README), at machine precision.

Flags (``published/harris2007_vol2.csv``):
* Eq. 9.14 (PDF p. 279) prints M_1 x^3 / 2 for M_1 x^2 / 2 (the integral of Eq. 9.13); not used here.
* Eqs 9.25 and 9.27 (PDF p. 281) print 1/l^3 on the P^k sum where the single-load Eqs 9.21, 9.23
  (PDF p. 280) print 1/l^2; 1/l^2 is the dimensionally right one (a moment, N.mm) and is used;
  the printed form is evaluated as a RECORD (L6) to show the misprint.

Sign mapping to AxisForge (plane XY, +v = +Y); THE CASE'S OWN READING, Harris states no direction for y
(Fig. 9.3 draws P down, F_1, F_2 up; librarian check): with EI y'' = -M (Eq. 9.8) and M = -F_1 x + M_1 left
of the load (Eq. 9.10), y is positive along P, so y = v and P^k is a force along +v; F_1, F_2 act against P (Fig. 9.3),
so the AxisForge reactions (force of the support on the shaft along +v) are ``R_A = -F_1``,
``R_B = -F_2``; theta = dy/dx = AxisForge ``theta_xy``; delta_r = settlement along +v. The couple T^k
enters Eq. 9.15 as -T for x > a, exactly as an r01 couple C enters M (r01 README): ``C = T^k``, from
+x toward +v. AxisForge receives it as an ``ExternalMoment`` with ``component(XY) = T^k``
(``convention-dependent (J-16)``: D-0010 rows 21-22).

Units: Harris prints no unit system for these equations (consistent units); the system here is in
AxisForge's mm, N, N.mm, MPa, so nothing is converted.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.core.loads import ExternalMoment, RadialLoad
from axisforge.outputs._format import read_records_csv

# OPEN R5-22 (as R2-02): LiteratureCase is defined in v101 and imported until a common place is ruled
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase
from studies.v000_axisforge.v001_shaft_uniform.construction import uniform_shaft

from studies.v000_axisforge.v001_shaft_uniform.check import section
from studies.v000_axisforge.v001_shaft_uniform.construction import THEORIES

from ..check import J16, TOL_NODAL

__all__ = ["cases", "harris_shaft", "harris_eqs_9_24_9_27", "harris_pin_slopes", "records", "LiteratureCase",
           "PUBLISHED", "SPAN_MM", "DIAMETER_MM", "LOADS", "SETTLEMENTS_MM", "N_ELEMENTS"]

PUBLISHED = Path(__file__).with_name("published") / "harris2007_vol2.csv"

# --- the system (chosen: Harris prints none). Uniform shaft = the body of construction (d 28 mm), the two
# pins at the shaft ends (Fig. 9.3 has no overhang) ---------------------------------------------------
SPAN_MM = 180.0                # l [mm]
DIAMETER_MM = 28.0             # [mm], S355 (v001 ensure_material)
# (a^k [mm], P^k [N] along +v, T^k [N.mm] from +x toward +v): two loads, superposition Eqs 9.24-9.27;
# T of the size of F_a r_w of construction's pinion (1.7 kN x 21 mm)
LOADS = ((60.0, 4000.0, 3.6e4), (130.0, -1500.0, -2.5e4))
SETTLEMENTS_MM = (0.02, -0.03)  # (delta_r1, delta_r2) along +v [mm]
N_ELEMENTS = 36                # uniform 5 mm elements: a^k on nodes


def harris_shaft(label: str = "harris2007_9_2_2"):
    """The uniform shaft of Fig. 9.3 with the ``LOADS`` in plane XY, unsolved (v001 ``uniform_shaft``).

    A force P along +v is a ``RadialLoad`` at theta 0 (+Y) for P > 0, 180 deg for P < 0; a couple T is
    an ``ExternalMoment`` of modulus abs(T) [N.mm] at theta 0 (component XY = +abs(T)) or 180 deg
    (J-16). Settlements are not part of the system: ``SETTLEMENTS_MM`` goes to the solve.
    """
    loads = []
    for a, P, T in LOADS:
        loads.append(RadialLoad(a, abs(P), theta_deg=0.0 if P >= 0.0 else 180.0, label=f"P@{a:g}"))
        loads.append(ExternalMoment(a, abs(T), theta_deg=0.0 if T >= 0.0 else 180.0, label=f"T@{a:g}"))
    return uniform_shaft(length_mm=SPAN_MM, diameter_mm=DIAMETER_MM, supports_mm=(0.0, SPAN_MM), loads=loads,
                         label=label)


def harris_eqs_9_24_9_27(EI: float, l: float, loads, theta_1: float, theta_2: float, delta_1: float, delta_2: float,
                         *, printed_l3: bool = False) -> dict:
    """Bearing forces and moments of Harris2007_vol2 Eqs 9.24-9.27 (PDF p. 281) [N, N.mm].

    Parameters
    ----------
    EI : float
        Bending stiffness [N.mm^2].
    l : float
        Bearing span [mm].
    loads : sequence of (a [mm], P [N], T [N.mm])
        Forces along y (down in Fig. 9.3) and couples at a, 0 < a < l.
    theta_1, theta_2 : float
        Shaft slopes dy/dx at the bearings [rad].
    delta_1, delta_2 : float
        Bearing deflections along y [mm].
    printed_l3 : bool
        True: the P^k sums of Eqs 9.25 and 9.27 divided by l^3 as printed (misprint); False: by l^2,
        as the single-load Eqs 9.21, 9.23 (PDF p. 280).

    Returns
    -------
    dict
        ``F_1``, ``F_2`` [N] (against P), ``M_1``, ``M_2`` [N.mm] (bearing moments of Fig. 9.3).
    """
    a = np.array([x for x, _, _ in loads], dtype=float)
    P = np.array([p for _, p, _ in loads], dtype=float)
    T = np.array([t for _, _, t in loads], dtype=float)
    lp = l**3 if printed_l3 else l**2
    s = theta_1 + theta_2 + 2.0 * (delta_1 - delta_2) / l
    F_1 = (np.sum(P * (l - a) ** 2 * (l + 2 * a)) / l**3 - 6.0 * np.sum(T * a * (l - a)) / l**3
           - 6.0 * EI / l**2 * s)                                                               # Eq. 9.24
    M_1 = (np.sum(P * a * (l - a) ** 2) / lp + np.sum(T * (l - a) * (l - 3 * a)) / l**2
           - 2.0 * EI / l * (2 * theta_1 + theta_2 + 3.0 * (delta_1 - delta_2) / l))           # Eq. 9.25
    F_2 = (np.sum(P * a**2 * (3 * l - 2 * a)) / l**3 + 6.0 * np.sum(T * a * (l - a)) / l**3
           + 6.0 * EI / l**2 * s)                                                               # Eq. 9.26
    M_2 = (np.sum(P * a**2 * (l - a)) / lp + np.sum(T * a * (2 * l - 3 * a)) / l**2
           + 2.0 * EI / l * (theta_1 + 2 * theta_2 + 3.0 * (delta_1 - delta_2) / l))           # Eq. 9.27
    return {"F_1": float(F_1), "F_2": float(F_2), "M_1": float(M_1), "M_2": float(M_2)}


def harris_pin_slopes(EI: float, l: float, loads, delta_1: float, delta_2: float) -> tuple[float, float]:
    """Bearing slopes of pin supports, from Eqs 9.25 and 9.27 with M_1 = M_2 = 0 (Harris, PDF p. 281) [rad].

    Both equations are linear in (theta_1, theta_2): solved as a 2 x 2 system.
    """
    m0 = harris_eqs_9_24_9_27(EI, l, loads, 0.0, 0.0, delta_1, delta_2)
    m1 = harris_eqs_9_24_9_27(EI, l, loads, 1.0, 0.0, delta_1, delta_2)
    m2 = harris_eqs_9_24_9_27(EI, l, loads, 0.0, 1.0, delta_1, delta_2)
    A = np.array([[m1["M_1"] - m0["M_1"], m2["M_1"] - m0["M_1"]], [m1["M_2"] - m0["M_2"], m2["M_2"] - m0["M_2"]]])
    theta = np.linalg.solve(A, -np.array([m0["M_1"], m0["M_2"]]))
    return float(theta[0]), float(theta[1])


def records(fem: dict, system) -> list[dict]:
    """L1-L6 of the README: the FEM of :func:`harris_shaft` against Harris's printed equations.

    Parameters
    ----------
    fem : dict
        From the FEM solve with ``SETTLEMENTS_MM``: ``theta_1``, ``theta_2`` [rad] (``theta_xy`` at
        x = 0 and x = l), ``R_A``, ``R_B`` [N] (``R_xy``).
    system : axisforge ShaftSystem
        :func:`harris_shaft` (its section gives EI).

    Returns
    -------
    list of dict
        ``case``, ``id``, ``quantity``, ``unit``, ``measured``, ``reference``, ``abs_diff``,
        ``ref_scale``, ``rel_err``, ``tol``, ``verdict``, ``source``, ``tag``. Scales: the largest
        static reaction (L1, L2), the largest fixed-end moment term ``abs(P a (l - a)^2 / l^2)`` +
        ``abs(T)`` (L3, L4, L6), the largest pin slope (L5).
    """
    EI, _ = section(system, THEORIES["euler_bernoulli"])     # [N.mm^2] (r01 section_stiffness)
    d1, d2 = SETTLEMENTS_MM
    h = harris_eqs_9_24_9_27(EI, SPAN_MM, LOADS, fem["theta_1"], fem["theta_2"], d1, d2)
    h_printed = harris_eqs_9_24_9_27(EI, SPAN_MM, LOADS, fem["theta_1"], fem["theta_2"], d1, d2, printed_l3=True)
    th = harris_pin_slopes(EI, SPAN_MM, LOADS, d1, d2)
    f_scale = max(abs(fem["R_A"]), abs(fem["R_B"]))
    m_scale = max(abs(P * a * (SPAN_MM - a) ** 2 / SPAN_MM**2) + abs(T) for a, P, T in LOADS)
    t_scale = max(abs(t) for t in th)
    src = "Harris2007_vol2 sec. 9.2.2"
    out = []

    def rec(lid, quantity, unit, measured, reference_, scale, eq, judged=True):
        diff = measured - reference_
        err = abs(diff) / scale
        verdict = ("PASS" if err <= TOL_NODAL else "MISS") if judged else "RECORD"
        out.append({"case": "v002", "id": lid, "quantity": quantity, "unit": unit, "measured": measured,
                    "reference": reference_, "abs_diff": diff, "ref_scale": scale, "rel_err": err,
                    "tol": TOL_NODAL if judged else math.nan, "verdict": verdict, "source": f"{src}, {eq}",
                    "tag": J16})

    # AxisForge reactions along +v; Harris's F act against P (along -y): R = -F
    rec("L1", "F_1 = -R_A", "N", -fem["R_A"], h["F_1"], f_scale, "Eq. 9.24, PDF p. 281")
    rec("L2", "F_2 = -R_B", "N", -fem["R_B"], h["F_2"], f_scale, "Eq. 9.26, PDF p. 281")
    rec("L3", "M_1 at the FEM slopes (pin: 0)", "N.mm", 0.0, h["M_1"], m_scale, "Eq. 9.25 (1/l^2), PDF p. 281")
    rec("L4", "M_2 at the FEM slopes (pin: 0)", "N.mm", 0.0, h["M_2"], m_scale, "Eq. 9.27 (1/l^2), PDF p. 281")
    rec("L5", "theta_1 (pin)", "rad", fem["theta_1"], th[0], t_scale, "Eqs 9.25, 9.27 with M = 0, PDF p. 281")
    rec("L5", "theta_2 (pin)", "rad", fem["theta_2"], th[1], t_scale, "Eqs 9.25, 9.27 with M = 0, PDF p. 281")
    rec("L6", "M_1 as printed (1/l^3)", "N.mm", 0.0, h_printed["M_1"], m_scale, "Eq. 9.25 as printed", judged=False)
    rec("L6", "M_2 as printed (1/l^3)", "N.mm", 0.0, h_printed["M_2"], m_scale, "Eq. 9.27 as printed", judged=False)
    return out


def cases() -> list[LiteratureCase]:
    """The published systems of v002 (one: Harris2007_vol2 sec. 9.2.2, Fig. 9.3, two loads, settled pins)."""
    return [LiteratureCase("harris2007_9_2_2", harris_shaft, read_records_csv(PUBLISHED),
                           "Harris2007_vol2 sec. 9.2.2, Fig. 9.3, Eqs 9.20-9.27, PDF p. 278-281")]
