"""
Construction -- 6210-like deep groove ball bearing and the load cases of
``validation/bearings/iso16281_ball_contact``.

Single source of the inputs of the case, written directly against the AxisForge API. The bearing
is the one of the case ``validation/contact/hertz_ellipse`` (same bearing, other loads); its
numbers are repeated here on purpose, a validation case shows its whole construction and imports
no study.

``build_bearing(s)`` returns the bearing assembled and NOT solved; the script chooses the solver.
``build_shaft_system(theta_deg)`` returns a shaft on two of these bearings with a radial load at mid-span,
NOT solved (input of ``02_sign_check``, open review item A2 on ``phi_Fr``).

Model
-----
Deep groove ball bearing with the boundary dimensions of a 6210 (d 50, D 90, B 20 mm), 9 balls of
12.7 mm on a pitch diameter of 70 mm, both groove radii 0.52 Dw (the AxisForge family default),
steel on steel. The radial clearance ``s`` sets the free contact angle ``alpha_0`` (0 for s = 0).
The number of balls and the groove radii are typical values, not catalogue data.

Units
-----
mm, N, MPa -- the AxisForge unit system. The conversion to SI is made only inside
``slippy_bridge``.
"""
from __future__ import annotations

import axisforge.core as af_c

__all__ = ["build_bearing", "build_shaft_system", "LOADS", "SIGN_CHECK"]

# ---- inputs -----------------------------------------------------------------------------------
E_STEEL = 210000.0      # Young's modulus [MPa]
NU_STEEL = 0.3          # Poisson's ratio [-]
DW = 12.7               # ball diameter [mm]
DPW = 70.0              # pitch diameter [mm]
Z = 9                   # balls [-]

# Load cases: radial clearance s [mm], radial load Fr [N] (applied in the XZ plane), axial load
# Fa [N]. Example values, easy to change: a moderate radial load without clearance (pressures at
# the level of the roughness and EHL work), a heavy radial load with clearance, and the same with
# an axial load (alpha_j rises above alpha_0 and differs from ball to ball).
LOADS = {
    "radial_moderate": dict(s=0.0, Fr=3300.0, Fa=0.0),
    "radial_heavy": dict(s=0.015, Fr=8900.0, Fa=0.0),
    "combined": dict(s=0.015, Fr=8900.0, Fa=4450.0),
}

# Sign check (02): plain shaft of the bearing bore, a bearing (20 mm wide) centred 10 mm from each
# end, so its footprint touches no shoulder; one radial point load at mid-span (the solver path the
# review item A2 is about). Two load directions: along -Z, and one with both Fy and Fz non-zero, so
# that the Y sign and the sense of rotation of D-0010 are tested too. Twice the load of
# "radial_moderate": each bearing carries 3300 N by symmetry. Clearance 0, Fa = 0, no load on the bearing nodes (the node force is
# then the pure support reaction, see BearingNodeData).
SIGN_CHECK = dict(
    shaft_length_mm=200.0, shaft_diameter_mm=50.0,                # d of the 6210
    bearing_positions_mm=(10.0, 190.0),                           # centres of the bearing widths
    load_position_mm=100.0, load_N=6600.0,
    # RadialLoad.theta_deg is measured from +Y toward +Z (AxisForge core/loads.py): 270 deg is -Z;
    # 300 deg gives Fy = +0.5 F, Fz = -0.866 F
    load_theta_deg=(270.0, 300.0),
    s_mm=0.0,
)
STEEL_SHAFT_ID = "S355"     # the ShaftSection default material_id; registered below if missing


def build_bearing(s: float, *, position: float = 0.0, arrangement: str = "locating",
                  label: str = "6210"):
    """6210-like deep groove ball bearing, assembled for ISO/TS 16281 and not solved.

    Parameters
    ----------
    s : float
        Radial internal clearance [mm], >= 0. It fixes the free contact angle ``alpha_0``.
    position : float, optional (0.0)
        Axial position of the centre of the bearing width on a shaft [mm].
    arrangement : str, optional ("locating")
        Axial role in the shaft FEM: "locating" fixes v and u, "non-locating" v only.
    label : str, optional ("6210")
        Bearing label; it names the bearing node of the shaft FEM.

    Returns
    -------
    axisforge Bearing
        With the ISO/TS 16281 contact data (``cp``, radii, ``alpha_0``); no load is applied.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="6210", position=position,
                                  arrangement=arrangement, label=label)
    return af_c.Bearing.assemble(
        family=af_c.DeepGrooveBallFamily(), catalog=catalog,
        geometry=dict(Dw=DW, Dpw=DPW, Z=Z, s=s, contact=af_c.ContactAnalysis.ISO16281,
                      e1=E_STEEL, e2=E_STEEL, nu1=NU_STEEL, nu2=NU_STEEL))


def build_shaft_system(theta_deg: float):
    """Plain shaft on two 6210-like ball bearings with a radial point load at mid-span, not solved.

    Statically determinate beam on two rigid supports (the shaft FEM used with it is
    ``RigidSupportFEMSolver``): each bearing carries half of the load, and the support pushes the
    shaft opposite to the load. Inputs in ``SIGN_CHECK``.

    Parameters
    ----------
    theta_deg : float
        Direction of the load [deg], AxisForge ``RadialLoad`` convention: from +Y toward +Z
        (270 is -Z). One of ``SIGN_CHECK["load_theta_deg"]``.

    Returns
    -------
    axisforge ShaftSystem
        Shaft (steel ``S355``, E 210000 MPa), bearing "6210_A" (locating) and "6210_B"
        (non-locating), one ``RadialLoad``; speed 0 (static, the load distribution does
        not use it).
    """
    # the registry starts empty; values and description copied verbatim from
    # validation/shafts/fem_uniform_radial_load/construction.py::_ensure_material, so the two
    # registrations of "S355" cannot disagree (whichever runs first wins)
    if STEEL_SHAFT_ID not in af_c.materials.base.available_materials():
        af_c.materials.base.register(af_c.Material(
            material_id=STEEL_SHAFT_ID, density=7850.0,
            elastic=af_c.IsotropicElastic(E=210_000.0, poisson_ratio=0.3),
            strength=af_c.StrengthProperties(Sut=590.0, Sy=355.0),
            description="EN 10025-2 S355 -- structural steel"))
    c = SIGN_CHECK
    shaft = af_c.Shaft(label="shaft")
    shaft.add_section(af_c.ShaftSection(length=c["shaft_length_mm"], diameter=c["shaft_diameter_mm"],
                                        material_id=STEEL_SHAFT_ID, label="shaft_body"))
    shaft_system = af_c.ShaftSystem(shaft, name=f"sign_check_{theta_deg:g}deg")
    for x, arrangement, label in zip(c["bearing_positions_mm"], ("locating", "non-locating"),
                                     ("6210_A", "6210_B")):
        shaft_system.add_bearing(build_bearing(c["s_mm"], position=x, arrangement=arrangement,
                                               label=label))
    shaft_system.add_load(af_c.RadialLoad(c["load_position_mm"], c["load_N"],
                                          theta_deg=theta_deg, label="F_load"))
    shaft_system.validate_or_raise()
    return shaft_system
