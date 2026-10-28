"""The published systems v005 knows (D-0036 amendment item 4): Harris and Kotzalas (2007) vol. 2, section 9.3.2.

* Fig. 9.9 (PDF p. 284): a uniform shaft on three rigid bearings, one force P in span 1 at a from bearing 1,
  no moment loads, bearing deflection differences negligible (the assumptions printed before Eq. 9.36). Eqs
  9.36-9.41 reduce to the reactions of Eqs 9.42-9.44 (PDF p. 285). **FLAG: the text before Eq. 9.42 cites
  "Equation 9.4" for Eq. 9.41** (PDF p. 285).
* Fig. 9.10 (PDF p. 285): the same three bearings under a RIGID shaft (EI -> inf): the bearing deflections lie
  on a line, Eq. 9.45 (similar triangles); each bearing follows F_r = K delta_r^n (Eq. 9.46, "radially loaded
  bearing with rigid rings": the maximum rolling-element load is proportional to F_r), so Eq. 9.47 and Eq. 9.48
  (PDF p. 286), "valid for bearings that support a radial load only", solved "simultaneously with the
  equilibrium equations". "See Example 9.5": the worked numbers are on the book's CD-ROM (J-14), not printed.

No numbers are printed for either figure, so the inputs are the case's own (construction) and the judge is
the printed equations at machine precision (Eqs 9.42-9.44) or solved to round-off (Eq. 9.48, by ``brentq``).

Signs (Harris ch. 9, Eqs 9.6-9.10, Fig. 9.4, PDF p. 278-279; traced at R5-141 in r01): y and delta_r positive
DOWN (along P), F upward on the shaft. Here P acts along -Y, so in the plane XY ``delta_r = -v`` and
``F = R_xy`` (the AxisForge reaction, force of the support on the shaft along +Y).

Eq. 9.48 with a signed root. Harris writes (F/K)^(1/n) for compressed bearings (F > 0). A radial bearing
without clearance also pushes the other way (another arc of its rolling elements loaded), so the case uses
``sign(F) |F/K|^(1/n)``; it equals Harris's form whenever every F > 0 and lets the flexible shaft carry a
negative F_3 (as Eq. 9.44 does for the rigid-bearing case).
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.core.loads import RadialLoad
from axisforge.outputs._format import read_records_csv
from scipy.optimize import brentq

# OPEN R5-153 (as R2-02, R5-22): LiteratureCase is defined in v101 and imported until a common place is ruled
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import K_HERTZ, LOAD_P, N_HERTZ, SUPPORTS_MM, three_support_shaft

__all__ = ["cases", "harris_eqs_9_42_9_44", "harris_eq_9_48", "harris_shaft", "spans", "LiteratureCase", "PUBLISHED"]

PUBLISHED = Path(__file__).with_name("published") / "harris2007_vol2.csv"


def harris_eqs_9_42_9_44(*, l1: float, l2: float, a: float, P: float) -> tuple[float, float, float]:
    """Reactions of three rigid bearings under one force P in span 1 (Harris2007_vol2 Eqs 9.42-9.44, PDF p. 285).

    Parameters
    ----------
    l1, l2 : float
        Spans bearing 1-2 and 2-3 [mm].
    a : float
        Position of P from bearing 1, 0 < a < l1 [mm].
    P : float
        Force, downward in Fig. 9.9 [N].

    Returns
    -------
    (F_1, F_2, F_3) : tuple of float
        Bearing forces on the shaft, upward (against P) [N]; F_3 < 0 (bearing 3 holds the shaft down).

    Notes
    -----
    Uniform EI, no moment loads, bearing deflection differences negligible (PDF p. 284): the reactions of a
    continuous beam on three rigid pins (Clapeyron), independent of EI.
    """
    F_1 = P * (l1 - a) * (2.0 * l1 * (l1 + l2) - a * (l1 + a)) / (2.0 * l1**2 * (l1 + l2))     # Eq. 9.42
    F_2 = P * a * ((l1 + l2) ** 2 - a**2 - l2**2) / (2.0 * l1**2 * l2)                         # Eq. 9.43
    F_3 = -P * a * (l1**2 - a**2) / (2.0 * l1 * l2 * (l1 + l2))                                 # Eq. 9.44
    return F_1, F_2, F_3


def harris_eq_9_48(*, l1: float, l2: float, a: float, P: float, K: float | tuple = K_HERTZ,
                   n: float = N_HERTZ) -> tuple[float, float, float]:
    """Reactions of three nonlinear bearings under a RIGID shaft (Harris2007_vol2 Eqs 9.45-9.48, PDF p. 285-286).

    Unknowns F_1, F_2, F_3 (on the shaft, upward, against P). Equilibrium of forces and of moments about
    bearing 1 gives ``F_3 = (P a - F_2 l1) / (l1 + l2)`` and ``F_1 = P - F_2 - F_3``; Eq. 9.48,

        delta_1 - delta_2 = (l1 / l2) (delta_2 - delta_3),    delta_i = sign(F_i) |F_i / K_i|^(1/n)  (Eq. 9.47),

    is then one equation in F_2, solved by ``scipy.optimize.brentq`` to 1e-15 relative. The function of F_2 is
    monotonic (each delta_i is monotonic in F_i and F_1, F_3 are linear in F_2), so the root is unique.

    Parameters
    ----------
    l1, l2, a, P : float
        As :func:`harris_eqs_9_42_9_44` [mm, N].
    K : float or (K_1, K_2, K_3)
        Load-deflection constants of the bearings [N/mm^n].
    n : float
        Load-deflection exponent [-]: 3/2 ball, 10/9 roller (Harris Eq. 9.46 with Q = K delta^n).

    Returns
    -------
    (F_1, F_2, F_3) : tuple of float
        Bearing forces on the shaft [N], upward.

    Notes
    -----
    Radial load only (Harris, PDF p. 286); bearings "with rigid rings" whose radial deflection is the
    maximum rolling-element deflection (PDF p. 285); a signed root extends the law to F < 0 (module
    docstring). The shaft is rigid: its own bending is absent, so a flexible shaft approaches this as EI grows.
    """
    K1, K2, K3 = (K, K, K) if isinstance(K, (int, float)) else K

    def delta(F, Ki):                       # Eq. 9.47 with a signed root [mm], down positive
        return math.copysign(abs(F / Ki) ** (1.0 / n), F)

    def split(F2):
        F3 = (P * a - F2 * l1) / (l1 + l2)  # moments about bearing 1
        return P - F2 - F3, F3              # forces

    def eq_9_48(F2):
        F1, F3 = split(F2)
        return delta(F1, K1) - delta(F2, K2) - l1 / l2 * (delta(F2, K2) - delta(F3, K3))

    # F_2 in [-P, 2P] brackets the root: eq_9_48 changes sign there for any load inside span 1
    F2 = brentq(eq_9_48, -P, 2.0 * P, xtol=1e-15 * P, rtol=4.0 * 2.220446049250313e-16, maxiter=200)
    F1, F3 = split(F2)
    return F1, F2, F3


def harris_shaft(label: str = "harris2007_fig_9_9"):
    """The shaft of Fig. 9.9 / 9.10 in this case's numbers: the construction with P only (plane XY), unsolved."""
    x, F, t = LOAD_P
    return three_support_shaft(loads=[RadialLoad(x, F, theta_deg=t, label="P")], label=label)


def spans() -> dict:
    """``l1``, ``l2``, ``a`` [mm] and ``P`` [N] of the construction in Harris's symbols (Fig. 9.9)."""
    x1, x2, x3 = SUPPORTS_MM
    return {"l1": x2 - x1, "l2": x3 - x2, "a": LOAD_P[0] - x1, "P": LOAD_P[1]}


def cases() -> list[LiteratureCase]:
    """The published systems of v005: Harris2007_vol2 Fig. 9.9 (rigid bearings) and Fig. 9.10 (rigid shaft)."""
    published = read_records_csv(PUBLISHED)
    return [LiteratureCase("harris2007_fig_9_9", harris_shaft, published,
                           "Harris2007_vol2 sec. 9.3.2, Fig. 9.9, Eqs 9.36-9.44, PDF p. 284-285"),
            LiteratureCase("harris2007_fig_9_10", harris_shaft, published,
                           "Harris2007_vol2 sec. 9.3.2.1, Fig. 9.10, Eqs 9.45-9.48, PDF p. 285-286")]
