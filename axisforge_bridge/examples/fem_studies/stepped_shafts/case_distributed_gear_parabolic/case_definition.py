"""
Case definition -- stepped shaft with shoulders, distributed loads only.

Single source of truth for the inputs of ``case_distributed_gear_parabolic``
and for building / solving its AxisForge model. Imported by both study
notebooks in this folder and by the closed-form reference
(``analytical_solution.py``); none of them re-declares a geometric,
material or load value.

Model
-----
Two identical stepped solid shafts (L = 200 mm, S355) with five sections
and four filleted shoulders:

====================  ==============  ============  =========================
section               x / mm          d / mm        role
====================  ==============  ============  =========================
bearing seat A        0 -- 20         20            6204, locating, x = 10
body                  20 -- 85        28            user load (shaft1)
gear seat             85 -- 115       25            spur gear, x = 100
body                  115 -- 180      28            user load (shaft2)
bearing seat B        180 -- 200      20            NU204, non-locating, x = 190
====================  ==============  ============  =========================

Loads (no point loads):

* gear mesh forces Ft, Fr distributed **uniformly** over the face width
  (92.5 -- 107.5 mm), via ``SpurHelicalMeshLink(distribute_loads=True)``;
* a user-defined **parabolic** distributed load,
  q(x) = q0 [1 - ((x - x_c)/(w/2))^2] on [x_c - w/2, x_c + w/2],
  resultant 2000 N at theta = 270 deg, w = 40 mm:
  ``shaft1`` on [40, 80] mm, ``shaft2`` on [130, 170] mm (deliberately
  not mirror images of each other about mid-span).

Assumptions
-----------
* Rigid pin supports at the bearing centres (v = 0; u = 0 at the locating
  bearing; rotation free), as in ``fem_solvers/constraints/boundary_conditions.py``.
* Linear elasticity, small displacements, isotropic material.
* Stepwise constant section properties: the shoulder fillets
  (r = 1 mm) are stress raisers only and do not modify EI or kappa*G*A.
* Uniform distribution of the gear mesh force over the face width.

Units
-----
mm, N, MPa (N/mm^2), N.mm, rad -- the AxisForge core unit system.

Notes
-----
PROVISIONAL. Shoulders are kept clear of the bearing faces (4 mm on the
locating side, 3 mm on the non-locating side): the current
``ShaftSystem._shoulder_coincidence_errors`` reports an overlap for a
shoulder that abuts a bearing face exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.loads import DistributedRadialLoad, TorqueLoad
from axisforge.core.machine_elements.bearings import (
    Bearing,
    BearingCatalog,
    CylindricalRollerFamily,
    DeepGrooveBallFamily,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import (
    SpurHelicalGearMeshing,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.materials.base import (
    IsotropicElastic,
    Material,
    StrengthProperties,
    available_materials,
    register,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalGearSystem,
    SpurHelicalMeshLink,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement,
    ShaftSystem,
)
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.results.fem_results.shaft_results import ShaftResults
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
    RigidSupportFEMSolver,
)

# =============================================================================
# Inputs
# =============================================================================

CASE_NAME = "stepped_shafts / distributed_load / case_distributed_gear_parabolic"

# --- shaft geometry [mm]: (length, outer diameter, label), left to right ---------
SECTIONS: list[tuple[float, float, str]] = [
    (20.0, 20.0, "bearing_seat_A"),
    (65.0, 28.0, "body_left"),
    (30.0, 25.0, "gear_seat"),
    (65.0, 28.0, "body_right"),
    (20.0, 20.0, "bearing_seat_B"),
]
SHOULDER_FILLET_MM = 1.0
SHAFT_LENGTH_MM = sum(L for L, _, _ in SECTIONS)

# --- material: EN 10025-2 S355 ------------------------------------------------
MATERIAL_ID = "S355"
E_MPA = 210_000.0
POISSON = 0.3
DENSITY_KG_M3 = 7850.0
SUT_MPA = 590.0
SY_MPA = 355.0

# --- bearings (catalogue dimensions [mm], internal geometry per ISO 281) -----
BALL_CATALOG = dict(d=20.0, D=42.0, b=12.0, designation="6204")
BALL_GEOMETRY = dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02)
BALL_POSITION_MM = 10.0

ROLLER_CATALOG = dict(d=20.0, D=47.0, b=14.0, designation="NU204")
ROLLER_GEOMETRY = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1)
ROLLER_POSITION_MM = 190.0

# --- gear stage: mesh forces distributed uniformly over the face width ------------
GEAR_MODULE_MM = 2.0
GEAR_Z_DRIVER = 20
GEAR_Z_DRIVEN = 40
GEAR_FACE_WIDTH_MM = 15.0
GEAR_POSITION_MM = 100.0
POWER_W = 10_471.9755          # T = 100 N.m at 1000 rpm
SHAFT1_RPM = 1000.0
SHAFT2_RPM = 500.0             # informational -- z1/z2 = 20/40
ROTATION_DIR_SOURCE = 1

# --- user load: parabolic distributed load ---------------------------------------
PARABOLIC_RESULTANT_N = 2000.0
PARABOLIC_WIDTH_MM = 40.0
PARABOLIC_THETA_DEG = 270.0
PARABOLIC_CENTRE_MM = {"shaft1": 60.0, "shaft2": 150.0}
PARABOLIC_PEAK_N_PER_MM = 1.5 * PARABOLIC_RESULTANT_N / PARABOLIC_WIDTH_MM   # int q dx = 2/3 q0 w

# --- beam-model presets used by the studies ----------------------------------------
THEORIES: dict[str, BeamModelSettings] = {
    "euler_bernoulli": BeamModelSettings("euler_bernoulli", None, None),
    "timoshenko/cowper": BeamModelSettings("timoshenko", "cowper", "single_point"),
    "timoshenko/hutchinson": BeamModelSettings("timoshenko", "hutchinson", "single_point"),
}


# =============================================================================
# Derived quantities
# =============================================================================

@dataclass(frozen=True)
class Segment:
    """One shaft section with its elastic properties.

    Attributes
    ----------
    x_lo, x_hi : float
        Axial extent [mm].
    d : float
        Outer diameter [mm] (solid section).
    E, nu, G : float
        Young's modulus, Poisson's ratio, shear modulus [MPa], [-], [MPa].
    I, A : float
        Second moment of area [mm^4] and area [mm^2].
    """

    x_lo: float
    x_hi: float
    d: float
    E: float
    nu: float
    G: float
    I: float
    A: float

    @property
    def EI(self) -> float:
        """Bending stiffness [N.mm^2]."""
        return self.E * self.I

    @property
    def GA(self) -> float:
        """Shear stiffness before the correction factor [N]."""
        return self.G * self.A


def segments() -> list[Segment]:
    """Section-wise elastic properties of the shaft, left to right.

    Returns
    -------
    list of Segment
        One entry per ``SECTIONS`` row.

    Examples
    --------
    >>> [(s.x_lo, s.x_hi, s.d) for s in segments()][:2]
    [(0.0, 20.0, 20.0), (20.0, 85.0, 28.0)]
    """
    G = E_MPA / (2.0 * (1.0 + POISSON))
    out, x = [], 0.0
    for L, d, _ in SECTIONS:
        out.append(Segment(x, x + L, d, E_MPA, POISSON, G,
                           np.pi * d**4 / 64.0, np.pi * d**2 / 4.0))
        x += L
    return out


def parabolic_intensity(centre_mm: float):
    """Parabolic load intensity q(x) [N/mm] centred at ``centre_mm``.

    Parameters
    ----------
    centre_mm : float
        Centre x_c of the load [mm].

    Returns
    -------
    callable
        q(x) = q0 [1 - ((x - x_c)/(w/2))^2] inside [x_c - w/2, x_c + w/2],
        0 outside; q0 = 1.5 F / w so that the resultant is F.

    Examples
    --------
    >>> q = parabolic_intensity(60.0)
    >>> q(60.0), q(40.0), q(100.0)
    (75.0, 0.0, 0.0)
    """
    half = PARABOLIC_WIDTH_MM / 2.0
    q0 = PARABOLIC_PEAK_N_PER_MM

    def q(x: float) -> float:
        xi = (x - centre_mm) / half
        return q0 * (1.0 - xi * xi) if abs(xi) <= 1.0 else 0.0

    return q


# =============================================================================
# Model construction
# =============================================================================

def _ensure_material() -> None:
    """Register S355 in the AxisForge material registry (idempotent)."""
    if MATERIAL_ID in available_materials():
        return
    register(Material(
        material_id=MATERIAL_ID,
        density=DENSITY_KG_M3,
        elastic=IsotropicElastic(E=E_MPA, poisson_ratio=POISSON),
        strength=StrengthProperties(Sut=SUT_MPA, Sy=SY_MPA),
        description="EN 10025-2 S355 -- structural steel",
    ))


def _bearings(shaft_name: str) -> tuple[Bearing, Bearing]:
    """Locating ball bearing + non-locating roller bearing, left to right."""
    ball = Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(**BALL_CATALOG, position=BALL_POSITION_MM,
                               arrangement="locating",
                               label=f"{shaft_name}_ball_locating"),
        geometry=dict(BALL_GEOMETRY),
    )
    roller = Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(**ROLLER_CATALOG, position=ROLLER_POSITION_MM,
                               arrangement="non-locating",
                               label=f"{shaft_name}_roller_nonlocating"),
        geometry=dict(ROLLER_GEOMETRY),
    )
    return ball, roller


def _shaft(name: str) -> Shaft:
    """Stepped shaft with a filleted shoulder at every diameter change."""
    shaft = Shaft(label=name)
    for L, d, label in SECTIONS:
        shaft.add_section(ShaftSection(length=L, diameter=d, label=f"{name}_{label}"))
    for i in range(len(SECTIONS) - 1):
        d_left, d_right = SECTIONS[i][1], SECTIONS[i + 1][1]
        shaft.set_transition(i, Shoulder(SHOULDER_FILLET_MM,
                                         max(d_left, d_right), min(d_left, d_right)))
    return shaft


def build_system() -> SpurHelicalGearSystem:
    """Build and resolve the two-shaft gear system of this case.

    Returns
    -------
    SpurHelicalGearSystem
        Resolved system with shafts ``shaft1`` (driver) and ``shaft2``
        (driven); gear mesh forces as uniform ``DistributedRadialLoad``
        over the face width, plus the parabolic user load.

    Notes
    -----
    ``shaft2`` only receives the gear reaction torque; an output coupling
    torque is added at x = L so that torsional equilibrium closes. It does
    not affect bending.

    Examples
    --------
    >>> s = build_system()
    >>> sorted({type(ld).__name__ for ld in s.shafts[0].loads})
    ['DistributedRadialLoad', 'TorqueLoad']
    """
    _ensure_material()

    ss1 = ShaftSystem(_shaft("shaft1"), name="shaft1", speed_rpm=SHAFT1_RPM)
    ss2 = ShaftSystem(_shaft("shaft2"), name="shaft2", speed_rpm=SHAFT2_RPM)
    for ss in (ss1, ss2):
        for brg in _bearings(ss.name):
            ss.add_bearing(brg)

    g1 = SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVER, b=GEAR_FACE_WIDTH_MM,
                         position=GEAR_POSITION_MM, label="g_driver")
    g2 = SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVEN, b=GEAR_FACE_WIDTH_MM,
                         position=GEAR_POSITION_MM, label="g_driven")
    meshing = SpurHelicalGearMeshing(g1, g2, label="stage1")
    ge1 = GearElement(g1, role="driver", label="g_driver")
    ge2 = GearElement(g2, role="driven", label="g_driven")
    ss1.add_gear(ge1)
    ss2.add_gear(ge2)

    half = PARABOLIC_WIDTH_MM / 2.0
    for ss in (ss1, ss2):
        xc = PARABOLIC_CENTRE_MM[ss.name]
        ss.add_load(DistributedRadialLoad(xc - half, xc + half, parabolic_intensity(xc),
                                          theta_deg=PARABOLIC_THETA_DEG,
                                          label="parabolic_user_load"))

    link = SpurHelicalMeshLink(ss1, ge1, ss2, ge2, meshing, phi_deg=0.0,
                               distribute_loads=True, label="stage1")
    system = SpurHelicalGearSystem([ss1, ss2], [link], label="case_distributed_gear_parabolic")
    system.validate_or_raise()
    system.resolve(P=POWER_W, rpm=SHAFT1_RPM, rotation_dir_source=ROTATION_DIR_SOURCE)

    driven = system.shafts[-1]
    net_torque = sum(ld.magnitude for ld in driven.torque_loads)
    driven.add_load(TorqueLoad(position=SHAFT_LENGTH_MM, magnitude=-net_torque,
                               label="output_coupling", source="user"))
    return system


# =============================================================================
# Solution
# =============================================================================

def uniform_nodes(n_elements: int) -> list[float]:
    """Equally spaced node positions over the whole shaft.

    Parameters
    ----------
    n_elements : int
        Number of equal divisions of the shaft length (>= 1).

    Returns
    -------
    list of float
        ``n_elements + 1`` positions in [0, L] [mm], passed to the solver as
        extra mandatory nodes.

    Examples
    --------
    >>> uniform_nodes(4)
    [0.0, 50.0, 100.0, 150.0, 200.0]
    """
    if n_elements < 1:
        raise ValueError(f"n_elements must be >= 1, got {n_elements}")
    return [float(x) for x in np.linspace(0.0, SHAFT_LENGTH_MM, n_elements + 1)]


def solve(system: SpurHelicalGearSystem, settings: BeamModelSettings,
          n_elements: int | None = None) -> dict[str, ShaftResults]:
    """Solve every shaft of ``system`` with one beam model.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Resolved system from :func:`build_system`.
    settings : BeamModelSettings
        Beam theory, shear correction and integration rule.
    n_elements : int, optional
        If given, ``n_elements`` uniform divisions are added on top of the
        mandatory nodes; if None, only the mandatory nodes are used.

    Returns
    -------
    dict of str to ShaftResults
        ``{shaft name: ShaftResults}``; displacements in mm, moments in
        N.mm, forces in N.

    Notes
    -----
    The gear loads are already ``DistributedRadialLoad`` objects, so the
    solver is called with its default ``distribute_gear_labels=None``.
    Passing ``{"*"}`` would drop point gear loads without distributing
    them (see notebook 02).

    Examples
    --------
    >>> res = solve(build_system(), THEORIES["euler_bernoulli"])
    >>> sorted(res)
    ['shaft1', 'shaft2']
    """
    extra = uniform_nodes(n_elements) if n_elements else None
    return {ss.name: RigidSupportFEMSolver(settings).solve(ss, extra_mandatory=extra)
            for ss in system.shafts}
