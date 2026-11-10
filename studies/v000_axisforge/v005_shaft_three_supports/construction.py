"""The system of v005: a uniform shaft on three supports, returned NOT solved.

One steel shaft (S355, L = 270 mm, d = 30 mm) on three supports at x = 10, 110, 260 mm, so the spans are
l1 = 100 mm and l2 = 150 mm (Harris2007_vol2 Fig. 9.9, PDF p. 284; Harris prints no numbers, these spans and
the load position a = 40 mm in span 1 are this case's own, the same as r01 C17). Loads (theta from +Y toward
+Z, D-0010):

* P = 1000 N at x = 50 mm, theta 180 deg: along -Y, plane XY only. Harris's P "down" is -Y here, so
  Harris's y (down) is -v of the plane XY (r01 README, R5-141);
* Q = 600 N at x = 200 mm, theta 90 deg: along +Z, plane XZ only, in span 2 (a second plane with its own
  sign, so a swapped plane shows).

Supports. AxisForge has no point-support object: each support is a 6204 of zero width (v001
``# OPEN R5-21``), the first locating (u = 0). ``RigidSupportFEMSolver`` takes any number of them (v = 0 at
each, rotation free, ``boundary_conditions.py``), so three rigid supports need no change. Springs and
settlements are not AxisForge inputs: ``study.solve`` adds them to AxisForge's own stiffness matrix
(``StiffnessMatrixBuilder.add_nodal_stiffness``) and load vector (``# OPEN R5-150``).

Base springs and settlements (``study.solve`` inputs, the linear part): the middle support a radial spring
of ``K_MID_N_PER_MM`` (about the shaft's own stiffness there, so neither limit dominates), the right support
rigid radially with a rotational spring ``K_THETA_N_MM_PER_RAD`` (a stiff bearing seat), settlements of the
order of the elastic deflection, different in each plane.

Nonlinear part: every support a radial Hertz law ``F = K delta^n`` (Harris2007_vol2 Eq. 9.46, PDF p. 285),
n = 3/2 (ball bearing), ``K_HERTZ`` the case's own value (about 7 um under 640 N, a small deep-groove ball
bearing; ``# OPEN R5-152``: not derived from a bearing, s003 uses ISO/TS 16281 bearings).

Assumptions: linear elastic isotropic shaft, small displacements, supports at points (no width), bearing
laws radial only (no tilt stiffness from the bearing in the nonlinear part).

Units: mm, N, MPa, N.mm, rad (AxisForge units; r01 uses the same, nothing is converted).
"""
from __future__ import annotations

from axisforge.core.loads import RadialLoad
from axisforge.core.machine_elements.bearings import Bearing, BearingCatalog, DeepGrooveBallFamily
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings

from studies.v000_axisforge.v001_shaft_uniform.construction import E_MPA, ensure_material, uniform_shaft

__all__ = ["three_support_shaft", "stiff_material", "LENGTH_MM", "DIAMETER_MM", "SUPPORTS_MM", "LOAD_P", "LOAD_Q",
           "SPRINGS", "SETTLEMENTS_MM", "K_MID_N_PER_MM", "K_THETA_N_MM_PER_RAD", "K_HERTZ", "N_HERTZ", "N_ELEMENTS",
           "MODELS"]

LENGTH_MM = 270.0
DIAMETER_MM = 30.0
SUPPORTS_MM = (10.0, 110.0, 260.0)          # l1 = 100, l2 = 150 [mm]
# (x [mm], F [N], theta [deg] from +Y toward +Z)
LOAD_P = (50.0, 1000.0, 180.0)              # -Y: Harris's P, a = 40 mm from support 1
LOAD_Q = (200.0, 600.0, 90.0)               # +Z: plane XZ, span 2

# middle spring: about 3 EI l / (l1^2 l2^2) = 2.8e4 N/mm, the stiffness of the shaft itself at x = 110 on pins at
# 10 and 260 (point-load deflection of a simply supported beam), so the middle support takes a share between
# the two limits
K_MID_N_PER_MM = 3.0e4
# rotational spring at the right support [N.mm/rad]: of the order of 4 EI / l2 = 2.2e8 (a clamped span end)
K_THETA_N_MM_PER_RAD = 2.0e8
# {x [mm]: (k_r [N/mm] (inf = rigid), k_theta [N.mm/rad])}: supports not listed are rigid pins
SPRINGS = {110.0: (K_MID_N_PER_MM, 0.0), 260.0: (float("inf"), K_THETA_N_MM_PER_RAD)}
# {plane: {x [mm]: settlement along +v [mm]}}: of the order of the elastic deflection (about 0.01 mm);
# at the spring it is the displacement of the spring's base
SETTLEMENTS_MM = {"XY": {10.0: 0.01, 110.0: -0.02, 260.0: 0.015}, "XZ": {10.0: -0.01, 110.0: 0.0, 260.0: 0.02}}

# Hertz law of each support, F = K delta^n [N, mm]: n = 3/2 for point contact (Harris2007_vol2 Eq. 9.46, PDF
# p. 285, with Q = K delta^n); K the case's own (# OPEN R5-152)
K_HERTZ = 1.0e6                             # [N/mm^1.5]
N_HERTZ = 1.5

N_ELEMENTS = (27, 54, 108, 216)             # uniform h = 10 ... 1.25 mm: supports and loads on nodes

# Euler-Bernoulli (nodally exact) and one-point Timoshenko (Cowper; v001 presets)
MODELS: dict[str, BeamModelSettings] = {
    "euler_bernoulli": BeamModelSettings("euler_bernoulli", None, None),
    "timoshenko/cowper/single_point": BeamModelSettings("timoshenko", "cowper", "single_point"),
}

# OPEN R5-151: v001 add_pins takes exactly two positions (locating, non-locating); the extra pins repeat its
# zero-width 6204 here until add_pins takes n positions (v001 is another case: not changed)
_PIN_CATALOG = dict(d=20.0, D=42.0, b=0.0, designation="6204")
_PIN_GEOMETRY = dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02)


def stiff_material(scale: float) -> str:
    """Register S355 with E multiplied by ``scale`` (the stiff-shaft limit of Harris Eq. 9.45) and return its id.

    Harris obtains Eq. 9.45 "by setting the shaft cross-section moment of inertia I to an infinitely large
    value" (PDF p. 285); E x scale gives the same EI x scale. ``scale = 1`` is plain S355.
    """
    if scale == 1.0:
        return ensure_material()
    return ensure_material(f"S355_E_x{scale:g}", E=E_MPA * scale)


def three_support_shaft(*, supports_mm: tuple[float, ...] = SUPPORTS_MM, loads: list | None = None,
                        material_id: str | None = None, label: str = "v005_three_supports"):
    """The uniform shaft on its supports (rigid pins as AxisForge sees them), loads applied, NOT solved.

    Parameters
    ----------
    supports_mm : tuple of float
        Support positions [mm], increasing; the first is locating. Two positions give the two-pin shaft of
        the first check and of the fed-back check (``study``).
    loads : list of axisforge Load, optional
        None: ``LOAD_P`` and ``LOAD_Q`` as ``RadialLoad``.
    material_id : str, optional
        A registered material (:func:`stiff_material`); None: S355.
    label : str
        Name of the shaft system; the bearings are ``pin_<x>``.

    Returns
    -------
    ShaftSystem
        Solve it with ``RigidSupportFEMSolver`` (rigid) or ``study.solve`` (springs, settlements).
    """
    if loads is None:
        loads = [RadialLoad(x, F, theta_deg=t, label=name) for (x, F, t), name in ((LOAD_P, "P"), (LOAD_Q, "Q"))]
    system = uniform_shaft(length_mm=LENGTH_MM, diameter_mm=DIAMETER_MM, supports_mm=(supports_mm[0], supports_mm[-1]),
                           loads=loads, material_id=material_id, label=label)
    for x in supports_mm[1:-1]:
        system.add_bearing(Bearing.assemble(
            family=DeepGrooveBallFamily(),
            catalog=BearingCatalog(**_PIN_CATALOG, position=x, arrangement="non-locating", label=f"pin_{x:g}"),
            geometry=dict(_PIN_GEOMETRY)))
    system.validate_or_raise()
    return system
