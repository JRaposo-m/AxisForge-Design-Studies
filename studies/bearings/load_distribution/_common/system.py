"""
The two-shaft gear system shared by every load distribution study.

This module only BUILDS the model (shafts, bearings, gears, loads); it does not solve
anything. The system is the same for every study: two identical shafts, a gear at mid-span
and two bearings per shaft.

Construction rule (checked by SystemSpec)
-----------------------------------------
Each shaft has exactly two bearings:

    bearing 1  locating       DeepGrooveSpec or AngularContactSpec   carries the axial load
    bearing 2  non-locating   CylindricalRollerSpec                  axially free (Fa = 0)

Why the bearing type does not change the reactions
--------------------------------------------------
The shaft FEM uses rigid supports, so each shaft is statically determinate: the reactions
(Fr, Fa) and the slopes psi at the bearing nodes depend only on the positions, the
arrangement and the gear loads, NOT on the bearing types or their parameters. The FEM is
therefore solved once per study and only the bearing under study is re-assembled at each point
of the sweep. This stops being true once the bearing stiffness is fed back into the shaft
model (hyperstatic coupling); see the family README.

Axial load
----------
``helix_angle_deg = 0`` gives spur gears (Fa = 0). A helical stage gives Fa = Ft tan(beta) at
the locating bearing, with opposite signs on the two shafts. Fa at the gear pitch radius
also adds a moment, so with helical gears the two bearings of one shaft no longer carry the
same Fr even though the gear sits at mid-span.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

from .bearings import (
    AngularContactSpec,
    BearingSpec,
    CylindricalRollerSpec,
    DeepGrooveSpec,
)

# =============================================================================
# Inputs
# =============================================================================

SHAFT_NAMES = ("shaft_1", "shaft_2")

# --- shaft geometry [mm]: (length, outer diameter, label), left to right --------------------
SECTIONS: list[tuple[float, float, str]] = [
    (20.0, 20.0, "bearing_seat_A"),
    (65.0, 28.0, "body_left"),
    (30.0, 25.0, "gear_seat"),
    (65.0, 28.0, "body_right"),
    (20.0, 20.0, "bearing_seat_B"),
]
SHOULDER_FILLET_MM = 1.0

# --- materials ---------------------------------------------------------------------------------
SHAFT_MATERIAL = dict(material_id="S355", E=210_000.0, poisson=0.3, density=7850.0,
                      Sut=590.0, Sy=355.0, description="S355 steel")
BEARING_MATERIAL = dict(material_id="GCr15", E=210_000.0, poisson=0.3, density=7850.0,
                        description="GCr15 steel")
GEAR_MATERIAL = dict(material_id="C45", E=210_000.0, poisson=0.3, density=7850.0,
                     description="C45 steel")

# --- bearing slots: arrangement -> (index in the label, axial position [mm]) ----------------
BEARING_SLOTS: dict[str, tuple[int, float]] = {
    "locating": (1, 10.0),
    "non-locating": (2, 190.0),
}
ALLOWED_TYPES: dict[str, tuple[type, ...]] = {
    "locating": (DeepGrooveSpec, AngularContactSpec),
    "non-locating": (CylindricalRollerSpec,),
}

# --- gear stage: mesh forces distributed uniformly over the face width -------------------------
GEAR_MODULE_MM = 2.0
GEAR_Z_DRIVER = 20
GEAR_Z_DRIVEN = 40
GEAR_FACE_WIDTH_MM = 15.0
GEAR_POSITION_MM = 100.0       # mid-span
MESH_PHI_DEG = 0.0             # direction of the line of centres (shaft_2 relative to shaft_1)
NOMINAL_POWER_W = 10_471.9755  # T = 100 N m at 1000 rpm; default of SystemSpec.power_W
SHAFT1_RPM = 1000.0
ROTATION_DIR_SOURCE = 1
SHAFT_RPM = {
    "shaft_1": SHAFT1_RPM,
    "shaft_2": SHAFT1_RPM * GEAR_Z_DRIVER / GEAR_Z_DRIVEN,
}

# --- beam models available to the FEM: key -> (theory, shear coefficient, integration) --------
THEORY_ARGS: dict[str, tuple[str, str, str]] = {
    "timoshenko/cowper": ("timoshenko", "cowper", "single_point"),
}


# =============================================================================
# System specification
# =============================================================================


@dataclass(frozen=True)
class SystemSpec:
    """What changes from one study to another: the two bearings, the helix angle and the power.

    Attributes
    ----------
    locating: BearingSpec
        Bearing 1 of every shaft; DeepGrooveSpec or AngularContactSpec.
    non_locating: BearingSpec
        Bearing 2 of every shaft; CylindricalRollerSpec.
    helix_angle_deg: float
        Normal helix angle of both gears [deg]; 0 gives spur gears (Fa = 0).
    power_W: float
        Power transmitted by the gear stage [W] at SHAFT1_RPM on the driving shaft; it fixes
        the torque and therefore the mesh forces and the bearing reactions.
    label: str
        Label of the AxisForge system.

    Raises
    ------
    TypeError
        If a bearing type is not allowed in its slot.
    ValueError
        If the helix angle is negative or not below 45 deg, or the power is not positive.
    """

    locating: BearingSpec = field(default_factory=DeepGrooveSpec)
    non_locating: BearingSpec = field(default_factory=CylindricalRollerSpec)
    helix_angle_deg: float = 0.0
    power_W: float = NOMINAL_POWER_W
    label: str = "load_distribution"

    def __post_init__(self):
        for arrangement, spec in (("locating", self.locating), ("non-locating", self.non_locating)):
            allowed = ALLOWED_TYPES[arrangement]
            if not isinstance(spec, allowed):
                names = " | ".join(t.__name__ for t in allowed)
                raise TypeError(f"the {arrangement} bearing must be {names}, "
                                f"got {type(spec).__name__}")
        if not 0.0 <= self.helix_angle_deg < 45.0:
            raise ValueError(f"helix_angle_deg must be in [0, 45), got {self.helix_angle_deg}")
        if not self.power_W > 0.0:
            raise ValueError(f"power_W must be positive, got {self.power_W}")
        if isinstance(self.locating, AngularContactSpec) and self.helix_angle_deg == 0.0:
            warnings.warn("angular contact bearing with spur gears: Fa = 0, the bearing floats "
                          "axially", stacklevel=2)

    def bearings(self) -> tuple[BearingSpec, BearingSpec]:
        """The two bearing specifications, locating first.

        Returns
        -------
        specs: tuple of BearingSpec
            (locating, non_locating).
        """
        return self.locating, self.non_locating

    def describe(self) -> dict:
        """Plain description for the run information of report.txt.

        Returns
        -------
        description: dict
            Both bearings, the helix angle and the power.
        """
        return dict(locating=self.locating.describe(), non_locating=self.non_locating.describe(),
                    helix_angle_deg=self.helix_angle_deg, power_W=self.power_W)


# =============================================================================
# Model construction
# =============================================================================


@lru_cache(maxsize=1)
def _materials():
    """Shaft, bearing and gear materials, registered once in the AxisForge registry.

    The core registry starts empty and ``register`` refuses duplicates, so the registration is
    guarded.

    Returns
    -------
    materials: tuple of af_c.Material
        (shaft, bearing, gear).
    """
    import axisforge.core as af_c

    def make(data, with_strength):
        strength = (af_c.StrengthProperties(Sut=data["Sut"], Sy=data["Sy"])
                    if with_strength else None)
        kwargs = dict(material_id=data["material_id"], density=data["density"],
                      elastic=af_c.IsotropicElastic(E=data["E"], poisson_ratio=data["poisson"]),
                      description=data["description"])
        if strength is not None:
            kwargs["strength"] = strength
        return af_c.Material(**kwargs)

    materials = (make(SHAFT_MATERIAL, True), make(BEARING_MATERIAL, False),
                 make(GEAR_MATERIAL, False))
    for material in materials:
        if material.material_id not in af_c.materials.base.available_materials():
            af_c.materials.base.register(material)
    return materials


def contact_data() -> dict:
    """ISO/TS 16281 contact data shared by every bearing (steel on steel).

    e1 / nu1 belong to the rolling element, e2 / nu2 to the raceways; here both are the
    bearing steel.

    Returns
    -------
    contact: dict
        contact model, e1, e2, nu1, nu2.
    """
    import axisforge.core as af_c

    bearing_material = _materials()[1]
    return dict(contact=af_c.ContactAnalysis.ISO16281,
                e1=bearing_material.E, e2=bearing_material.E,
                nu1=bearing_material.poisson_ratio, nu2=bearing_material.poisson_ratio)


def beam_model(theory_key: str):
    """Beam model settings for the rigid-support FEM.

    Parameters
    ----------
    theory_key: str
        Key of THEORY_ARGS.

    Returns
    -------
    settings: af_m.BeamModelSettings
        The beam model.
    """
    import axisforge.mesh as af_m
    return af_m.BeamModelSettings(*THEORY_ARGS[theory_key])


def bearing_label(shaft_name: str, kind: str, n: int) -> str:
    """Label of bearing ``n`` of a shaft; it names the bearing node of the FEM.

    Parameters
    ----------
    shaft_name: str
        One of SHAFT_NAMES.
    kind: str
        BearingSpec.kind.
    n: int
        Index of the slot (1 = locating, 2 = non-locating).

    Returns
    -------
    label: str
        For example "shaft_1_ball_bearing_1".
    """
    return f"{shaft_name}_{kind}_bearing_{n}"


def assemble_bearing(spec: BearingSpec, shaft_name: str):
    """Assemble one bearing in the slot of its arrangement.

    The label depends only on the shaft, the kind and the slot, never on the parameters, so a
    re-assembled bearing always matches the FEM bearing node.

    Parameters
    ----------
    spec: BearingSpec
        The bearing to assemble.
    shaft_name: str
        One of SHAFT_NAMES.

    Returns
    -------
    bearing: af_c.Bearing
        The assembled bearing.
    """
    n, position = BEARING_SLOTS[spec.arrangement]
    return spec.assemble(bearing_label(shaft_name, spec.kind, n), position, contact_data())


def build_shaft(label: str, sections: list[tuple[float, float, str]] = SECTIONS,
                fillet_mm: float = SHOULDER_FILLET_MM):
    """Stepped shaft with a shoulder fillet at every change of diameter.

    Parameters
    ----------
    label: str
        Shaft label.
    sections: list of (length [mm], diameter [mm], label)
        Sections from left to right.
    fillet_mm: float
        Shoulder fillet radius [mm].

    Returns
    -------
    shaft: af_c.Shaft
        The shaft, without bearings or loads.
    """
    import axisforge.core as af_c

    shaft = af_c.Shaft(label=label)
    for length, diameter, section_label in sections:
        shaft.add_section(af_c.ShaftSection(length=length, diameter=diameter,
                                            label=f"{label}_{section_label}"))
    for i in range(len(sections) - 1):
        d_left, d_right = sections[i][1], sections[i + 1][1]
        if d_left == d_right:          # equal diameters: plain boundary, no shoulder
            continue
        shaft.set_transition(i, af_c.Shoulder(fillet_mm, max(d_left, d_right),
                                              min(d_left, d_right)))
    return shaft


def build_system(spec: SystemSpec):
    """Build the two-shaft gear system; nothing is solved.

    Parameters
    ----------
    spec: SystemSpec
        Bearings and helix angle.

    Returns
    -------
    system: af_c.SpurHelicalGearSystem
        Shafts with both bearings, gear-mesh loads resolved and the driven shaft torque
        closed by an output coupling.
    """
    import axisforge.core as af_c

    _materials()

    shaft_systems = {}
    for name in SHAFT_NAMES:
        shaft_system = af_c.ShaftSystem(build_shaft(name), name=name, speed_rpm=SHAFT_RPM[name])
        for bearing_spec in spec.bearings():
            shaft_system.add_bearing(assemble_bearing(bearing_spec, name))
        shaft_systems[name] = shaft_system
    ss1, ss2 = (shaft_systems[name] for name in SHAFT_NAMES)

    # spur gears are built without beta_n_deg, exactly as before the migration
    helix = {} if spec.helix_angle_deg == 0.0 else dict(beta_n_deg=spec.helix_angle_deg)
    g_driver = af_c.SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVER, b=GEAR_FACE_WIDTH_MM,
                                    position=GEAR_POSITION_MM, label="g_driver",
                                    material_id=GEAR_MATERIAL["material_id"], **helix)
    g_driven = af_c.SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVEN, b=GEAR_FACE_WIDTH_MM,
                                    position=GEAR_POSITION_MM, label="g_driven",
                                    material_id=GEAR_MATERIAL["material_id"], **helix)
    meshing = af_c.SpurHelicalGearMeshing(g_driver, g_driven, label="stage1")
    ge_driver = af_c.GearElement(g_driver, role="driver", label="g_driver")
    ge_driven = af_c.GearElement(g_driven, role="driven", label="g_driven")
    ss1.add_gear(ge_driver)
    ss2.add_gear(ge_driven)

    link = af_c.SpurHelicalMeshLink(ss1, ge_driver, ss2, ge_driven, meshing,
                                    phi_deg=MESH_PHI_DEG, label="stage1")
    system = af_c.SpurHelicalGearSystem([ss1, ss2], [link], label=spec.label)
    system.validate_or_raise()
    system.resolve(P=spec.power_W, rpm=SHAFT1_RPM, rotation_dir_source=ROTATION_DIR_SOURCE)

    # the driven shaft only receives the mesh reaction torque: close it with a sink
    net = sum(load.magnitude for load in ss2.torque_loads)
    ss2.add_load(af_c.TorqueLoad(position=ss2.shaft.total_length, magnitude=-net,
                                 label="output_coupling", source="user"))
    return system
