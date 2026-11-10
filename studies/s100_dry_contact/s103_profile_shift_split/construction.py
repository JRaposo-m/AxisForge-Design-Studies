"""The systems of s103: the FZG pairs of v107 with their profile-shift sum split differently, NOT solved.

The pairs are v107's (``PAIRS``: FEUP test gears C40, spur, z 16 / 24, m_n 4.5 mm, b 40 mm, x 0.1817 / 0.1715; H501,
helical beta 15 deg, z 20 / 30, m_n 3.5 mm, b 23 mm, x 0.1381 / 0.1319; E 206 GPa, nu 0.3 assumed, R-118; the pinion
drives, Q-09; hands of H501 not printed, GL-38), imported from v107 (``# OPEN R5-130``, the R2-02 pattern). The sum
x1 + x2 of FEUP is kept; only the split x1 changes. With the sum fixed, a_w, alpha_wt and the line of action T1T2 do
not change (ISO_21771_2007 5.2 Eqs (55), (58), PDF p. 32-35): only the tip circles move (Eq. (33),
PDF p. 28), so the path A..E slides along T1T2. Load: the K9 wheel torque 323.27 N.m (Martins2015 Table 4, on the
wheel, R-124), gear-1 torque ``T2 z1 / z2``; F_tw = 2000 T / d_w does not change with the split (d_w fixed).

Tips: two variants (``# OPEN R5-138``, R-306 is José's). ``addendum_reduction=False`` is v107's k = 0 (tips from x
alone); ``True`` is AxisForge's tip alteration k = (a_w - a) / m_n - (x1 + x2) (Q23, the same on both gears and
constant at a fixed sum), which reproduces FEUP's printed tips and eps_alpha (C40 82.457 / 118.365 mm, eps 1.438;
printed 82.46 / 1.44 and FZG type C 118.36; H501 80.373 / 116.564, eps 1.463, printed 80.37 / 116.57, 1.45).

``as_cut`` gives the same pair in AxisForge's manufactured family, zero allowances, cut by the counterpart of the
ISO 53 basic rack (``GeneratingRackProfile.from_basic_rack``: h_aP0 1.25 m_n, rho_aP0 0.38 m_n): its undercut limit
x_Emin (7.7 Eq. (135)), root form diameters d_Ff (7.6 Eq. (128)) and the check d_Nf >= d_Ff (5.4.1 Eq. (66), (67)).

Units: AxisForge mm, N, N.m, deg.
"""
from __future__ import annotations

from dataclasses import replace

from axisforge.core.machine_elements.gears import Gear, GearPair, ManufacturedCylindricalFamily

# OPEN R5-130 (R2-02 pattern): the FZG pairs, torque table and assumptions are v107's, imported
from studies.v100_dry_contact.v107_fzg_contact_path.construction import (DRIVER, E_MPA, FAMILY, NU, PAIRS,
                                                                        WHEEL_TORQUE_NM, build_pair)

__all__ = ["split_pair", "as_cut", "pair_data", "x_sum", "STAGE", "PAIRS", "WHEEL_TORQUE_NM", "DRIVER", "E_MPA", "NU",
           "FAMILY", "MADE", "N_SWEEP", "PUBLISHED_SPLITS"]

STAGE = "K9"
N_SWEEP = 25                    # x1 values between the bounds of the sweep (model.bounds), ends included
MADE = ManufacturedCylindricalFamily()
# other published splits of the same sum (x1 only): GEARpie / Gear-Calculation H501 x 0.1809 / 0.0891, sum 0.27, as in
# AxisForge baseline_cases.py:89-90 (R-307: differs from FEUP's 0.1381 / 0.1319)
PUBLISHED_SPLITS = {"H501": {"GEARpie (R-307)": 0.1809}}


def x_sum(name: str) -> float:
    """FEUP's profile-shift sum x1 + x2 of the pair [-], kept by every split."""
    return float(sum(PAIRS[name]["x"]))


def pair_data(name: str, x1: float) -> dict:
    """The basic data of v107 ``PAIRS[name]`` with the shifts ``(x1, x_sum - x1)``: the ``data`` of v107 ``check``."""
    return dict(PAIRS[name], x=(x1, x_sum(name) - x1))


def split_pair(name: str, x1: float | None = None, *, equal_sliding: bool = False, addendum_reduction: bool = False,
               check: bool = True) -> GearPair:
    """The pair ``name`` with x1 and x2 = x_sum - x1, zero backlash, pinion driving, NOT solved.

    v107 ``build_pair`` with the shifts of its gear data replaced (v107 is not changed; ``# OPEN R5-131``).

    Parameters
    ----------
    name : {"C40", "H501"}
        v107 pair.
    x1 : float, optional
        Profile shift of the pinion [-]; None: FEUP's.
    equal_sliding : bool
        True: AxisForge splits the sum itself (``assemble_pair(x_split="equal_specific_sliding")``: equal specific
        sliding at the two roots, zeta_f1 = zeta_f2, ISO_21771_2007 5.6.3 Eq. (116) = (117), PDF p. 45-48); ``x1`` is
        then ignored.
    addendum_reduction : bool
        True: AxisForge's tip alteration k (Q23, Eq. (33): k < 0 shortens both tips by -k m_n); False: k = 0
        (v107). ``# OPEN R5-138`` (R-306).
    check : bool
        AxisForge ``validate`` on assembly (T1A >= 0, T2E >= 0, eps_alpha >= 1); False only to locate the bounds of
        the sweep (``model.bounds``).
    """
    base = build_pair(name)                                       # FEUP's split, validated by v107's construction
    x1 = base.x1 if x1 is None else x1
    gears = [Gear.assemble(FAMILY, replace(g.data, x=x), check=check)
             for g, x in zip((base.gear1, base.gear2), (x1, x_sum(name) - x1))]
    kw = {"x_split": "equal_specific_sliding"} if equal_sliding else {}
    return GearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], label=name, check=check,
                             addendum_reduction=addendum_reduction, **kw)


def as_cut(pair: GearPair) -> GearPair:
    """The same pair as cut: AxisForge ``ManufacturedCylindricalFamily``, zero allowances, counterpart rack, NOT checked.

    The working gears of ``pair`` already carry its x and k, so the nominal geometry is identical (AxisForge
    ``from_ideal``). Adds x_Emin (ISO_21771_2007 7.7 Eq. (135), PDF p. 63), d_Ffs = d_Ffi (7.6 Eq. (128), PDF p. 62;
    None under undercut) on each gear and d_Nf1, d_Nf2 on the pair; ``MADE.validate_pair`` reports undercut, a pointed
    tip and d_Nf < d_Ff (the mate's tip on the fillet). Kept out of the bridge (it takes the ideal family).
    """
    gears = [MADE.from_ideal(g, E_sns=0.0, E_sni=0.0, check=False) for g in (pair.gear1, pair.gear2)]
    return GearPair.assemble(MADE, gear1=gears[0], gear2=gears[1], label=pair.label, check=False)
