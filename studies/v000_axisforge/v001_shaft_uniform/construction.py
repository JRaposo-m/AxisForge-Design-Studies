"""The system of v001: a uniform solid shaft on two rigid pins under point forces, returned NOT solved.

One steel shaft (L = 200 mm, d = 20 mm, S355) on two ideal pins at x_A = 25 mm (locating) and
x_B = 175 mm (non-locating), so both ends overhang by 25 mm. Two point forces, one per bending
plane, so that each plane is a separate r01 problem:

* F1 = 500 N at x = 75 mm (one third of the span), theta 270 deg: along -Z, plane XZ;
* F2 = 200 N at x = 200 mm (tip of the right overhang), theta 0 deg: along +Y, plane XY.

Ported from ``v000_axisforge/v001_shaft_fem_uniform_radial_load`` at the tag ``studies-pre-sweep``
(shaft, material, deep-groove geometry, ``uniform_nodes``, the beam-model presets). Changed: no gear
stage (its loads are v002's question), supports and loads placed on multiples of 25 mm so that a
uniform mesh exists from 8 elements up (the reduced-integration identity of ``check`` needs equal
elements), bearings of zero width (a bearing of width b adds nodes at x +- b/2, Mesh1D).

Assumptions: linear elastic isotropic material, small displacements, rigid supports. The FEM solver
constrains v = 0 at each bearing centre (and u = 0 at the locating one) and never the rotation
(``fem_solvers/constraints/boundary_conditions.py``), so both supports are ideal pins.

Units: mm, N, MPa, rad (AxisForge units; r01 uses the same, so nothing is converted in this case).
"""
from __future__ import annotations

import numpy as np
from axisforge.core.loads import Load, RadialLoad
from axisforge.core.machine_elements.bearings import Bearing, BearingCatalog, DeepGrooveBallFamily
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection
from axisforge.core.materials.base import IsotropicElastic, Material, StrengthProperties, available_materials, register
from axisforge.core.mechanical_system import ShaftSystem
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings

__all__ = ["uniform_shaft", "uniform_nodes", "ensure_material", "THEORIES", "N_ELEMENTS", "LENGTH_MM",
           "DIAMETER_MM", "SUPPORTS_MM", "E_MPA", "NU", "MATERIAL_ID"]

# --- shaft and material: EN 10025-2 S355 (old case) ---------------------------------------------
LENGTH_MM = 200.0
DIAMETER_MM = 20.0
MATERIAL_ID = "S355"
E_MPA = 210_000.0            # Young's modulus [MPa]
NU = 0.3                     # Poisson's ratio [-]
DENSITY_KG_M3 = 7850.0       # not read by the static solver
SUT_MPA, SY_MPA = 590.0, 355.0

# --- supports: ideal pins (rigid, rotation free) -------------------------------------------------
SUPPORTS_MM = (25.0, 175.0)  # (locating, non-locating) [mm]
# OPEN R5-21: AxisForge has no point-support object; a 6204 of zero width stands for an ideal pin.
# RigidSupportFEMSolver reads only its position and arrangement (boundary_conditions.py), and b = 0
# keeps the mesh free of the extent nodes x +- b/2 (ShaftSystem.bearing_extent, Mesh1D).
_PIN_CATALOG = dict(d=20.0, D=42.0, b=0.0, designation="6204")
_PIN_GEOMETRY = dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02)

# --- loads (position [mm], magnitude [N], theta [deg] from +Y toward +Z, D-0010) -------------------
_LOADS = ((75.0, 500.0, 270.0, "F1"),     # -Z: plane XZ only (cos 270 deg ~ -1.8e-16, so F_y ~ 1e-13 N)
          (200.0, 200.0, 0.0, "F2"))      # +Y at the overhang tip: plane XY only

# --- mesh levels: elements over the whole shaft, h = L / n = 25 ... 1.5625 mm --------------------
N_ELEMENTS = (8, 16, 32, 64, 128)

# --- beam models (old case's presets, plus the full-integration Timoshenko option) ---------------
# "single_point": one-point (reduced) integration of the shear term (Onate2013_vol2 Eq. (2.38)), the
# AxisForge recommendation; "exact": full integration, which locks for slender elements (Onate 2.4)
THEORIES: dict[str, BeamModelSettings] = {
    "euler_bernoulli": BeamModelSettings("euler_bernoulli", None, None),
    "timoshenko/cowper/single_point": BeamModelSettings("timoshenko", "cowper", "single_point"),
    "timoshenko/hutchinson/single_point": BeamModelSettings("timoshenko", "hutchinson", "single_point"),
    "timoshenko/cowper/exact": BeamModelSettings("timoshenko", "cowper", "exact"),
}


def ensure_material(material_id: str = MATERIAL_ID, *, E: float = E_MPA, nu: float = NU) -> str:
    """Register a linear elastic material in the AxisForge registry once and return its id.

    Parameters
    ----------
    material_id : str
        Registry key; S355 by default (the shaft of this case).
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio [-].

    Returns
    -------
    str
        ``material_id``. The AxisForge registry starts empty and refuses a second registration of
        an id, so an id already present is kept as it is (its E and nu are not checked).
    """
    if material_id not in available_materials():
        register(Material(material_id=material_id, density=DENSITY_KG_M3,
                          elastic=IsotropicElastic(E=E, poisson_ratio=nu),
                          strength=StrengthProperties(Sut=SUT_MPA, Sy=SY_MPA), description=f"v001 {material_id}"))
    return material_id


def uniform_shaft(*, length_mm: float = LENGTH_MM, diameter_mm: float = DIAMETER_MM,
                  supports_mm: tuple[float, float] = SUPPORTS_MM, loads: list[Load] | None = None,
                  material_id: str | None = None, label: str = "v001_uniform") -> ShaftSystem:
    """A uniform solid shaft on two ideal pins with its loads applied, NOT solved.

    Parameters
    ----------
    length_mm, diameter_mm : float
        Shaft length and diameter [mm]; one section, so EI and kGA are constant.
    supports_mm : (float, float)
        Positions of the locating and the non-locating pin [mm], inside [0, L].
    loads : list of axisforge Load, optional
        Loads in AxisForge's own objects (forces [N], positions [mm]); None: the two point forces of
        this case (module docstring).
    material_id : str, optional
        A registered material; None: S355 (registered here when missing).
    label : str
        Name of the shaft system.

    Returns
    -------
    ShaftSystem
        Shaft, two zero-width bearings (pins) and the loads; solve it with ``RigidSupportFEMSolver``.
    """
    material_id = material_id or ensure_material()
    shaft = Shaft(label=label)
    shaft.add_section(ShaftSection(length=length_mm, diameter=diameter_mm, material_id=material_id, label=label))
    system = ShaftSystem(shaft, name=label)
    for x, arrangement in zip(supports_mm, ("locating", "non-locating")):
        system.add_bearing(Bearing.assemble(
            family=DeepGrooveBallFamily(),
            catalog=BearingCatalog(**_PIN_CATALOG, position=x, arrangement=arrangement, label=f"pin_{x:g}"),
            geometry=dict(_PIN_GEOMETRY)))
    if loads is None:
        loads = [RadialLoad(x, F, theta_deg=t, label=name) for x, F, t, name in _LOADS]
    for load in loads:
        system.add_load(load)
    system.validate_or_raise()
    return system


def uniform_nodes(n_elements: int, length_mm: float = LENGTH_MM) -> list[float]:
    """``n_elements + 1`` equally spaced node positions over the whole shaft [mm].

    Passed to ``RigidSupportFEMSolver.solve(extra_mandatory=...)``; supports and loads are mesh nodes
    anyway (Mesh1D), so the mesh is uniform when they fall on these positions (multiples of 25 mm
    here, hence n a multiple of 8).

    Examples
    --------
    >>> uniform_nodes(4)
    [0.0, 50.0, 100.0, 150.0, 200.0]
    """
    if n_elements < 1:
        raise ValueError(f"n_elements must be >= 1, got {n_elements}")
    return [float(x) for x in np.linspace(0.0, length_mm, n_elements + 1)]
