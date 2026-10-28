"""The systems of v003: single-row bearings and the sign-check shaft, returned NOT solved.

* ``deep_groove_6210(s_mm)``: a 6210-like deep groove ball bearing (d 50, D 90, B 20 mm; 9 balls of
  12.7 mm on a 70 mm pitch diameter; groove radii 0.52 Dw, the AxisForge family default), steel on steel,
  assembled for ISO/TS 16281. ``s_mm`` is the internal clearance, **diametral** (AxisForge
  ``families/geometry.py``: ``alpha_0 = arccos(1 - s/(2A))``, ISO/TS 16281 4.2.2), the ``P_d`` of Harris
  and r03 (bearing-expert review of r03, R-105).
* ``angular_contact_7210(alpha_0_deg)``: a 7210-like angular contact ball bearing (13 balls of 12.7 mm,
  alpha_0 40 deg), the RECORD case (its contact angle follows the load in AxisForge, r03 keeps it fixed).
* ``cylindrical_nu210(s_mm)``: an NU210-like cylindrical roller bearing (14 rollers 11 x 12 mm, 30
  laminae), the RECORD case (AxisForge applies the ISO/TS 16281 reference profile in laminae, r03 has none).
* ``sign_check_shaft(theta_deg)``: a plain shaft (L 200 mm, d 50 mm) on two of the 6210 at 10 and 190 mm
  with one ``RadialLoad`` of 6600 N at mid-span, for the azimuth sign check (R-003). Ported from
  ``v100_dry_contact/v104_iso16281_ball_contact/construction.py`` at the tag ``studies-pre-sweep``.

Ball counts, groove radii and roller sizes are typical values, not catalogue data. Units: mm, N, MPa
(AxisForge); the SI conversion for r03 is made once, in ``check.py``.
"""
from __future__ import annotations

import axisforge.core as af_c

__all__ = ["deep_groove_6210", "angular_contact_7210", "cylindrical_nu210", "sign_check_shaft", "E_STEEL_MPA",
           "NU_STEEL", "DW_MM", "DPW_MM", "Z_6210", "SIGN_CHECK"]

E_STEEL_MPA = 210_000.0     # Young's modulus of balls, rollers and rings [MPa]
NU_STEEL = 0.3              # Poisson's ratio [-]
DW_MM = 12.7                # ball diameter [mm]
DPW_MM = 70.0               # pitch diameter [mm]
Z_6210 = 9                  # balls [-]
_MATERIAL = dict(e1=E_STEEL_MPA, e2=E_STEEL_MPA, nu1=NU_STEEL, nu2=NU_STEEL)

# Sign check (R-003): plain shaft of the 6210 bore, bearings (20 mm wide) centred 10 mm from each end, one
# radial load at mid-span. Two directions: -Z (theta 270 deg), and theta 300 deg with both Fy = +0.5 F and
# Fz = -0.866 F non-zero, so the Y sign and the sense of rotation of D-0010 are tested too. RadialLoad.theta_deg
# is measured from +Y toward +Z (axisforge core/loads.py). Clearance 0, Fa = 0.
SIGN_CHECK = dict(shaft_length_mm=200.0, shaft_diameter_mm=50.0, bearing_positions_mm=(10.0, 190.0),
                  load_position_mm=100.0, load_N=6600.0, load_theta_deg=(270.0, 300.0), s_mm=0.0)
_SHAFT_MATERIAL = "S355"    # the ShaftSection default material_id; registered below if missing


def deep_groove_6210(s_mm: float = 0.0, *, position: float = 0.0, arrangement: str = "locating",
                     label: str = "6210", Dpw_mm: float = DPW_MM, E_MPa: float = E_STEEL_MPA):
    """6210-like deep groove ball bearing for ISO/TS 16281, assembled and not solved.

    Parameters
    ----------
    s_mm : float
        Diametral internal clearance [mm], ``>= 0``; it fixes the free contact angle
        ``alpha_0 = arccos(1 - s/(2A))``, ``A = ri + re - Dw`` (ISO/TS 16281 4.2.2).
    position, arrangement, label : optional
        Axial position of the width centre on a shaft [mm], "locating" / "non-locating", node label.
    Dpw_mm, E_MPa : float, optional
        Pitch diameter [mm] and Young's modulus of every body [MPa] (the Hamrock literature bearing changes them).

    Returns
    -------
    axisforge Bearing
        With the ISO/TS 16281 constants (``cp`` [N/mm^1.5], radii, ``alpha_0``); no load.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="6210", position=position,
                                  arrangement=arrangement, label=label)
    return af_c.Bearing.assemble(family=af_c.DeepGrooveBallFamily(), catalog=catalog,
                                 geometry=dict(Dw=DW_MM, Dpw=Dpw_mm, Z=Z_6210, s=s_mm,
                                               contact=af_c.ContactAnalysis.ISO16281,
                                               **{**_MATERIAL, "e1": E_MPa, "e2": E_MPa}))


def angular_contact_7210(alpha_0_deg: float = 40.0, *, label: str = "7210"):
    """7210-like single-row angular contact ball bearing (13 balls of 12.7 mm, Dpw 70 mm), not solved.

    The contact angle is an input (``alpha_0_deg``); AxisForge derives a clearance ``s = 2A(1 - cos alpha_0)``
    from it, which is NOT a clearance to give r03 (bearing-expert review): r03 gets ``clearance_m = 0``.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="7210B", position=0.0,
                                  arrangement="locating", label=label)
    return af_c.Bearing.assemble(family=af_c.AngularContactFamily(), catalog=catalog,
                                 geometry=dict(Dw=DW_MM, Dpw=DPW_MM, Z=13, alpha_0_deg=alpha_0_deg,
                                               contact=af_c.ContactAnalysis.ISO16281, **_MATERIAL))


def cylindrical_nu210(s_mm: float = 0.02, *, label: str = "NU210"):
    """NU210-like cylindrical roller bearing (14 rollers, Dwe 11 mm, Lwe 12 mm, 30 laminae), not solved.

    ``s_mm`` is the diametral clearance [mm]. AxisForge applies the ISO/TS 16281 reference profile ``P(x_k)``
    to the laminae (``contact_solver.py``, Eq. (41)); the bearing is immutable, the profile cannot be removed.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="NU210", position=0.0,
                                  arrangement="non-locating", label=label)
    return af_c.Bearing.assemble(family=af_c.CylindricalRollerFamily(), catalog=catalog,
                                 geometry=dict(Dwe=11.0, Lwe=12.0, Dpw=DPW_MM, Z=14, s=s_mm, n_s=30,
                                               contact=af_c.ContactAnalysis.ISO16281, **_MATERIAL))


def sign_check_shaft(theta_deg: float):
    """Plain shaft on two 6210 with a radial point load at mid-span, not solved (R-003 sign check).

    Statically determinate on two rigid supports (``RigidSupportFEMSolver``): each bearing carries half the
    load, and the support pushes the shaft opposite to the load. Inputs in ``SIGN_CHECK``.

    Parameters
    ----------
    theta_deg : float
        Direction of the load [deg], AxisForge ``RadialLoad``: from +Y toward +Z (270 is -Z).

    Returns
    -------
    axisforge ShaftSystem
        Steel shaft (S355, E 210000 MPa), bearings "6210_A" (locating), "6210_B" (non-locating), one load.
    """
    if _SHAFT_MATERIAL not in af_c.materials.base.available_materials():
        # values of v001's S355 (EN 10025-2), so two registrations cannot disagree
        af_c.materials.base.register(af_c.Material(
            material_id=_SHAFT_MATERIAL, density=7850.0,
            elastic=af_c.IsotropicElastic(E=210_000.0, poisson_ratio=0.3),
            strength=af_c.StrengthProperties(Sut=590.0, Sy=355.0), description="EN 10025-2 S355"))
    c = SIGN_CHECK
    shaft = af_c.Shaft(label="shaft")
    shaft.add_section(af_c.ShaftSection(length=c["shaft_length_mm"], diameter=c["shaft_diameter_mm"],
                                        material_id=_SHAFT_MATERIAL, label="shaft_body"))
    system = af_c.ShaftSystem(shaft, name=f"v003_sign_check_{theta_deg:g}deg")
    for x, arrangement, label in zip(c["bearing_positions_mm"], ("locating", "non-locating"), ("6210_A", "6210_B")):
        system.add_bearing(deep_groove_6210(c["s_mm"], position=x, arrangement=arrangement, label=label))
    system.add_load(af_c.RadialLoad(c["load_position_mm"], c["load_N"], theta_deg=theta_deg, label="F_load"))
    system.validate_or_raise()
    return system
