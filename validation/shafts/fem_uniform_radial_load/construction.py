"""
Construction -- uniform shaft, one radial point load, two load positions.

Single source of truth for the inputs of the validation case
``validation/shafts/fem_uniform_radial_load`` and for building its AxisForge
model, written directly against the AxisForge API (no shared construction
helpers, on purpose: the case is a proof of that API). Imported by both
notebooks in this folder as a sibling module (``import construction as cd``);
the closed-form reference (``references/beams/uniform_beam.py``) reads the
system built here. None of them re-declares a geometric, material or load
value.

``build_system()`` returns the system with its loads applied and NOT solved.
``solve()`` is a thin call to the AxisForge solver: the beam model and the
mesh are always chosen by the notebook.

Model
-----
Two identical uniform solid shafts (L = 200 mm, d = 20 mm, S355), each
carried by a deep-groove ball bearing (locating, x = 10 mm) and a
cylindrical roller bearing (non-locating, x = 190 mm). A spur gear pair
(z = 20 / 40, m_n = 2 mm, at x = 100 mm) transmits P = 10.47 kW at
1000 rpm, which supplies a fixed background load on both shafts. What
differs between the two shafts is the position of one extra radial
point load, 500 N at theta = 270 deg:

* ``shaft1`` -- load at x = 60 mm (near the locating bearing)
* ``shaft2`` -- load at x = 140 mm (near the non-locating bearing)

Assumptions
-----------
* Rigid supports. The FEM solver constrains v = 0 at each bearing
  centre (and u = 0 at the locating one) and never constrains the
  bending rotation, so both supports are ideal pins
  (``fem_solvers/constraints/boundary_conditions.py``).
* Linear elastic, small displacements, isotropic material.
* Gear mesh forces applied as point loads at the gear centre
  (``distribute_gear_labels=None``).

Units
-----
mm, N, MPa (N/mm^2), N.mm, rad -- the AxisForge core unit system.

Notes
-----
Bearings and material are declared here (top of file) on purpose. The
same bearing set is repeated in the stepped case: deliberate, each case
shows its full construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.loads import RadialLoad, TorqueLoad
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
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection
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

CASE_NAME = "uniform_shafts / point_load / case_radial_single_position"

# --- shaft geometry [mm] -----------------------------------------------------
SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 20.0

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

# --- gear stage (background load, identical in both shafts) -------------------
GEAR_MODULE_MM = 2.0
GEAR_Z_DRIVER = 20
GEAR_Z_DRIVEN = 40
GEAR_FACE_WIDTH_MM = 15.0
GEAR_POSITION_MM = 100.0
POWER_W = 10_471.9755          # T = 100 N.m at 1000 rpm
SHAFT1_RPM = 1000.0
SHAFT2_RPM = 500.0             # informational -- z1/z2 = 20/40
ROTATION_DIR_SOURCE = 1

# --- the variable under study: radial test load ---------------------------------
TEST_LOAD_N = 500.0
TEST_LOAD_THETA_DEG = 270.0
TEST_LOAD_POSITION_MM = {"shaft1": 60.0, "shaft2": 140.0}

# --- beam-model presets used by the studies ----------------------------------------
THEORIES: dict[str, BeamModelSettings] = {
    "euler_bernoulli": BeamModelSettings("euler_bernoulli", None, None),
    "timoshenko/cowper": BeamModelSettings("timoshenko", "cowper", "single_point"),
    "timoshenko/hutchinson": BeamModelSettings("timoshenko", "hutchinson", "single_point"),
}


# =============================================================================
# Derived section properties
# =============================================================================

@dataclass(frozen=True)
class SectionProperties:
    """Elastic properties of the uniform solid circular section.

    Attributes
    ----------
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio [-].
    G : float
        Shear modulus, G = E / (2 (1 + nu)) [MPa].
    I : float
        Second moment of area, pi d^4 / 64 [mm^4].
    A : float
        Cross-section area, pi d^2 / 4 [mm^2].
    """

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


def section_properties() -> SectionProperties:
    """Section and material properties of this case.

    Returns
    -------
    SectionProperties
        E, nu, G [MPa], I [mm^4], A [mm^2] for the solid d = 20 mm
        S355 shaft.

    Examples
    --------
    >>> p = section_properties()
    >>> round(p.I, 3), round(p.G, 1)
    (7853.982, 80769.2)
    """
    d = SHAFT_DIAMETER_MM
    return SectionProperties(
        E=E_MPA,
        nu=POISSON,
        G=E_MPA / (2.0 * (1.0 + POISSON)),
        I=np.pi * d**4 / 64.0,
        A=np.pi * d**2 / 4.0,
    )


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
    shaft = Shaft(label=name)
    shaft.add_section(ShaftSection(length=SHAFT_LENGTH_MM,
                                   diameter=SHAFT_DIAMETER_MM, label=name))
    return shaft


def build_system() -> SpurHelicalGearSystem:
    """Build and resolve the two-shaft gear system of this case.

    The system is fully resolved (gear-mesh forces injected) and
    independent of the beam theory used later to solve it.

    Returns
    -------
    SpurHelicalGearSystem
        Resolved system with shafts ``shaft1`` (driver) and ``shaft2``
        (driven). Loads in N, positions in mm.

    Notes
    -----
    ``shaft2`` only receives the gear reaction torque; an output
    coupling torque is added at x = L so that torsional equilibrium
    closes (otherwise the torsion DOF is unconstrained). It does not
    affect bending.

    Examples
    --------
    >>> [ss.name for ss in build_system().shafts]
    ['shaft1', 'shaft2']
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

    for ss in (ss1, ss2):
        ss.add_load(RadialLoad(TEST_LOAD_POSITION_MM[ss.name], TEST_LOAD_N,
                               theta_deg=TEST_LOAD_THETA_DEG, label="radial_test_load"))

    link = SpurHelicalMeshLink(ss1, ge1, ss2, ge2, meshing, phi_deg=0.0, label="stage1")
    system = SpurHelicalGearSystem([ss1, ss2], [link], label="case_radial_single_position")
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
        ``n_elements + 1`` positions in [0, L] [mm]. Passed to the solver
        as extra mandatory nodes; bearings, gear and loads stay on nodes
        regardless.

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
        If given, the mesh is refined with ``n_elements`` uniform
        divisions on top of the mandatory nodes. If None, only the
        mandatory nodes are used (coarsest admissible mesh).

    Returns
    -------
    dict of str to ShaftResults
        ``{shaft name: ShaftResults}``; displacements in mm, moments in
        N.mm, forces in N.

    Notes
    -----
    The Timoshenko element is linear with reduced shear integration and
    its nodal deflection error decreases as h^2: the mandatory-node mesh
    (``n_elements=None``) has ~10 % error for this case. See notebook 02
    before choosing ``n_elements``.

    Examples
    --------
    >>> res = solve(build_system(), THEORIES["euler_bernoulli"])
    >>> round(res["shaft1"].v_max, 4)
    0.3662
    """
    extra = uniform_nodes(n_elements) if n_elements else None
    return {ss.name: RigidSupportFEMSolver(settings).solve(ss, extra_mandatory=extra)
            for ss in system.shafts}
