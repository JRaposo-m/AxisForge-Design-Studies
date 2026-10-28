"""The published systems v001 knows (D-0036 amendment item 4): three beams of Onate (2013), vol. 2.

* Table 2.1 (PDF p. 84): cantilever under an end point load, 2-node Timoshenko elements with one-point
  shear integration (AxisForge's element), limit end deflection ratio ``r_w = w / w_EB`` for very
  slender beams (lambda -> inf) against the number of elements. AxisForge has no clamp (its supports
  are pins, ``boundary_conditions.py``), so the cantilever is built by symmetry: a simply supported
  span 2L with a central force 2P. By symmetry the mid-span rotation is zero (also in the discrete
  model when the mesh is symmetric), so each half is a cantilever of length L clamped at mid-span
  with the support reaction P at its tip; the mid-span deflection relative to the supports is the
  cantilever tip deflection, and n elements per half are Onate's n elements.
* Table 1.1 (PDF p. 66) with Fig. 1.17 (PDF p. 65): simply supported beam, central point load or
  uniform load; Onate prints the error of two rotation-free elements (CCB, CVB), not of AxisForge's.
  The printed system is used with AxisForge's Euler-Bernoulli element, whose error must print 0.00 %.
* Table 2.3 (PDF p. 126) with Fig. 2.17 (PDF p. 124): simply supported square beam under a uniform
  load, "slender" and "thick" closed-form central deflections for h/L = 0.01 ... 0.20.

The rows, with their pages, are in ``published/onate2013_vol2.csv`` (read from the page images of
the PDF in the library, 2026-10-28; the library note ``06_Library/system/Onate2013_vol2.md`` has no
content pass yet). Book page = PDF page - 31.

Units: Onate's beams are in SI (Fig. 1.17 caption); they are converted here, once, to AxisForge's
mm, N, MPa. A square section is replaced by the solid circle of the same I (Euler-Bernoulli needs EI
only); its shear stiffness enters the solver as ``kGA_override`` (``kga_square``).
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.core.loads import DistributedRadialLoad, RadialLoad
from axisforge.outputs._format import read_records_csv

# OPEN R5-22 (as R2-02): LiteratureCase is defined in v101 and imported until a common place is ruled
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import ensure_material, uniform_shaft

__all__ = ["cases", "cantilever_by_symmetry", "onate_fig_1_17", "onate_table_2_3", "kga_square", "LiteratureCase",
           "PUBLISHED", "HALF_LENGTH_MM", "TIP_FORCE_N", "SLENDER_KGA_FACTOR", "ONATE_LENGTH_MM", "ONATE_E_MPA",
           "ONATE_NU", "M_TO_MM"]

PUBLISHED = Path(__file__).with_name("published") / "onate2013_vol2.csv"

# --- SI -> AxisForge units (the one place where this case converts) --------------------------------
M_TO_MM = 1e3                  # [mm/m]
PA_TO_MPA = 1e-6               # [MPa/Pa]
N_PER_M_TO_N_PER_MM = 1e-3     # [N/mm per N/m]

# --- Table 2.1: the cantilever by symmetry (steel shaft of construction; r_w depends on nothing else) ---
HALF_LENGTH_MM = 100.0         # cantilever length L [mm] (chosen; r_w is dimensionless)
TIP_FORCE_N = 500.0            # cantilever tip force P [N]; the span carries 2P at mid-span
# lambda -> inf: kGA_override = 1e6 EI / L^2 makes the shear part of the tip deflection
# 3 EI / (kGA L^2) = 3e-6 of the bending part, 1/170 of the half unit printed (5e-4); larger factors
# start to cost digits in K (cond(K) grows with kGA)
SLENDER_KGA_FACTOR = 1e6

# --- Fig. 1.17 / Fig. 2.17 / Table 2.3 (SI as printed) ---------------------------------------------
_L_M = 4.0                     # beam length [m]
_E_PA = 47619.05               # Young's modulus [Pa], Fig. 1.17
_I_M4 = 2.133e-7               # second moment of area [m^4], Fig. 1.17 (h^4/12 with h = 0.04 m)
_P_N = 1.0                     # |P| [N], Fig. 1.17 (printed P = -1: downward)
_Q_N_PER_M = 1.0               # |q| [N/m], Fig. 1.17 (Q = -1) and Fig. 2.17 (q = 1.0, drawn downward)
ONATE_LENGTH_MM = _L_M * M_TO_MM
ONATE_E_MPA = _E_PA * PA_TO_MPA
# OPEN R5-23: Table 2.3 prints no E; Fig. 1.17's E is taken (its slender h/L = 0.01 value, 5 q L^4 / (384 E I),
# with I = 0.04^4/12, is the -328 printed). OPEN R5-24: Table 2.3 prints no nu, k_z; Onate's rectangular
# example (text after Eq. (2.33), PDF p. 80) has nu = 0.25, k_z = 5/6 (Fig. 2.3, PDF p. 73): used, thick column RECORD
ONATE_NU = 0.25
_KZ_RECTANGLE = 5.0 / 6.0
_ONATE_MATERIAL = "ONATE2013_E47619"


def _circle_of_same_I(I_mm4: float) -> float:
    """Diameter [mm] of the solid circle whose second moment of area is ``I_mm4`` [mm^4]: I = pi d^4 / 64."""
    return (64.0 * I_mm4 / math.pi) ** 0.25


def cantilever_by_symmetry(*, half_length_mm: float = HALF_LENGTH_MM, tip_force_N: float = TIP_FORCE_N):
    """Onate's Table 2.1 cantilever as a simply supported span 2L with 2P at mid-span, unsolved.

    The steel shaft of the construction (d = 20 mm), pins at x = 0 and 2L, force 2P along -Z at x = L
    (plane XZ). The cantilever tip deflection is the mid-span deflection (module docstring); its
    Euler-Bernoulli value is ``P L^3 / (3 EI)`` = ``(2P) (2L)^3 / (48 EI)``.
    """
    L = half_length_mm
    return uniform_shaft(length_mm=2.0 * L, supports_mm=(0.0, 2.0 * L),
                         loads=[RadialLoad(L, 2.0 * tip_force_N, theta_deg=270.0, label="2P")],
                         label="onate2013_table_2_1")


def _onate_material() -> str:
    return ensure_material(_ONATE_MATERIAL, E=ONATE_E_MPA, nu=ONATE_NU)


def onate_fig_1_17(load: str = "point"):
    """Onate's Fig. 1.17 simply supported beam, unsolved: L = 4 m, E = 47619.05 Pa, I = 2.133e-7 m^4.

    ``load="point"``: |P| = 1 N at mid-span; ``"udl"``: |q| = 1 N/m over the whole length; both along
    -Z (Onate's loads are negative, his z up: AxisForge v_xz is Onate's w). Converted to mm, N, MPa;
    the section is the circle of the same I.
    """
    I_mm4 = _I_M4 * M_TO_MM**4
    L = ONATE_LENGTH_MM
    if load == "point":
        loads = [RadialLoad(0.5 * L, _P_N, theta_deg=270.0, label="P")]
    elif load == "udl":
        # DistributedRadialLoad takes the TOTAL force [N] when given a float
        loads = [DistributedRadialLoad(0.0, L, _Q_N_PER_M * N_PER_M_TO_N_PER_MM * L, theta_deg=270.0, label="q")]
    else:
        raise ValueError(f"load must be 'point' or 'udl', got {load!r}")
    return uniform_shaft(length_mm=L, diameter_mm=_circle_of_same_I(I_mm4), supports_mm=(0.0, L), loads=loads,
                         material_id=_onate_material(), label=f"onate2013_fig_1_17_{load}")


def onate_table_2_3(h_over_L: float):
    """Onate's Fig. 2.17 / Table 2.3 simply supported square beam under q = 1 N/m, unsolved.

    L = 4 m, square side h = (h/L) L, I = h^4/12 (as the circle of the same I), E of Fig. 1.17
    (OPEN R5-23), load along -Z. The shear stiffness of the square is :func:`kga_square`.
    """
    L = ONATE_LENGTH_MM
    h = h_over_L * L
    return uniform_shaft(length_mm=L, diameter_mm=_circle_of_same_I(h**4 / 12.0), supports_mm=(0.0, L),
                         loads=[DistributedRadialLoad(0.0, L, _Q_N_PER_M * N_PER_M_TO_N_PER_MM * L, theta_deg=270.0,
                                                      label="q")],
                         material_id=_onate_material(), label=f"onate2013_table_2_3_hL{h_over_L:g}")


def kga_square(h_over_L: float) -> float:
    """Corrected shear stiffness k_z G A of Onate's square section [N] (``kGA_override`` of the solve).

    ``k_z = 5/6`` (rectangle, Fig. 2.3, PDF p. 73), ``G = E / (2 (1 + nu))`` with nu = 0.25 (OPEN R5-24),
    ``A = h^2`` [mm^2].
    """
    h = h_over_L * ONATE_LENGTH_MM
    return _KZ_RECTANGLE * ONATE_E_MPA / (2.0 * (1.0 + ONATE_NU)) * h * h


def cases() -> list[LiteratureCase]:
    """The published systems of v001: Onate Tables 2.1, 1.1 and 2.3 (module docstring)."""
    rows = read_records_csv(PUBLISHED)

    def of(*tables):
        return [r for r in rows if r["table"] in tables]

    return [LiteratureCase("onate2013_table_2_1", cantilever_by_symmetry, of("Table 2.1"),
                           "Onate2013_vol2 Table 2.1, Eqs (2.35), (2.40)-(2.42), PDF p. 81-84"),
            LiteratureCase("onate2013_table_1_1", onate_fig_1_17, of("Fig. 1.17", "Table 1.1"),
                           "Onate2013_vol2 Fig. 1.17 and Table 1.1, PDF p. 65-66"),
            LiteratureCase("onate2013_table_2_3", onate_table_2_3,
                           of("Fig. 2.17", "Table 2.3", "Text Eq. (2.33)", "Fig. 2.3"),
                           "Onate2013_vol2 Fig. 2.17 and Table 2.3, PDF p. 124-126")]
