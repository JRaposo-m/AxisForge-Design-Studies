# studies\bearings\clearance_load_distribution\construction.py
"""
Construction of the two-shaft spur gear system used by the clearance study.

This module only BUILDS the model (shafts, bearings, gears, loads). It does not
solve anything: the FEM and the ISO/TS 16281 load distribution belong to study.py.

What study.py is meant to import
--------------------------------
    build_system(s)    full SpurHelicalGearSystem, bearings assembled with clearance s,
                       gear-mesh loads resolved, driven shaft torque closed
    build_bearings(s)  {shaft_name: {label: Bearing}} for a given s, WITHOUT rebuilding
                       the shafts. With rigid supports the reactions do not depend on s
                       (statically determinate), so the FEM can be solved once and only
                       the bearings re-assembled for every s of the sweep.
    THEORY             beam model(s) for RigidSupportFEMSolver
    SHAFT_NAMES, S_REFERENCE_MM, ...

Clearance convention
--------------------
s is the TOTAL radial internal clearance in mm, i.e. the diametral play Pd of
ISO 5753-1 (the catalogue "radial internal clearance" Gr), NOT the radial offset of
one ring: contact_angle_and_clearance uses cos(alpha_0) = 1 - s / (2 A).
DeepGrooveBallFamily and CylindricalRollerFamily reject s < 0 (no preload).
"""

import axisforge.core as af_c
import axisforge.mesh as af_m

# =============================================================================
# Inputs
# =============================================================================

SHAFT_NAMES = ("shaft_1", "shaft_2")

# --- shaft geometry [mm]: (length, outer diameter, label), left to right ---------
# in this study all shafts will be the same, however it is possible to define different
# geometries for each shaft, as long as the number of sections is the same
SECTIONS: list[tuple[float, float, str]] = [
    (20.0, 20.0, "bearing_seat_A"),
    (65.0, 28.0, "body_left"),
    (30.0, 25.0, "gear_seat"),
    (65.0, 28.0, "body_right"),
    (20.0, 20.0, "bearing_seat_B"),
]
SECTIONS_BY_SHAFT: dict[str, list[tuple[float, float, str]]] = {
    name: SECTIONS for name in SHAFT_NAMES
}

SHOULDER_FILLET_MM = 1.0
SHAFT_LENGTH_MM = sum(L for L, _, _ in SECTIONS)

# --- shaft material properties --------------------------------
SHAFT_MATERIAL_ID = "S355"
SHAFT_E_MPA = 210_000.0
SHAFT_POISSON = 0.3
SHAFT_DENSITY_KG_M3 = 7850.0
SHAFT_SUT_MPA = 590.0
SHAFT_SY_MPA = 355.0
# --- bearing material properties -------------------------------
BEARING_MATERIAL_ID = "GCr15"
BEARING_E_MPA = 210_000.0
BEARING_POISSON = 0.3
BEARING_DENSITY_KG_M3 = 7850.0
# --- gear material properties ----------------------------------
GEAR_MATERIAL_ID = "C45"
GEAR_E_MPA = 210_000.0
GEAR_POISSON = 0.3
GEAR_DENSITY_KG_M3 = 7850.0

# --- bearings (catalogue dimensions [mm], internal geometry per ISO 281) -----
BALL_CATALOG = dict(d=20.0, D=42.0, b=12.0, designation="6204")
# s is left out on purpose: it is the study variable (see build_bearings).
# TODO: confirm Dw and Z against the manufacturer data for the 6204.
BALL_GEOMETRY = dict(Dw=7.0, Dpw=31.0, Z=9)

# Cylindrical roller bearing (NU/N type) of bearing 2: same 20 x 42 x 12 envelope.
# s is left out here too (study variable).
# TODO: PLACEHOLDER geometry -- replace Dwe, Lwe, Z (and the designation) with the
# manufacturer data of the bearing you want. Dpw = (d + D) / 2 is only the mean diameter.
# n_s = number of laminae of the ISO/TS 16281 slice model (family requires >= 30).
ROLLER_CATALOG = dict(d=20.0, D=42.0, b=12.0, designation="NU1004")
ROLLER_GEOMETRY = dict(Dwe=5.0, Lwe=5.0, Dpw=31.0, Z=14, n_s=50)

BRG1_POSITION_MM = 10.0
BRG2_POSITION_MM = 190.0
# (index, position [mm], arrangement, kind): bearing 1 (ball) carries any axial load,
# bearing 2 (cylindrical roller) floats
BEARING_SPECS = (
    (1, BRG1_POSITION_MM, "locating", "ball"),
    (2, BRG2_POSITION_MM, "non-locating", "roller"),
)
BEARING_KINDS = tuple(kind for *_, kind in BEARING_SPECS)
# Reference clearance used when a system is built without a sweep. The FEM reactions
# do not depend on it (rigid supports), it only has to be a valid s >= 0.
S_REFERENCE_MM = 0.010

# --- gear stage: mesh forces distributed uniformly over the face width ------------
GEAR_MODULE_MM = 2.0
GEAR_Z_DRIVER = 20
GEAR_Z_DRIVEN = 40
GEAR_FACE_WIDTH_MM = 15.0
GEAR_POSITION_MM = 100.0
MESH_PHI_DEG = 0.0             # direction of the line of centres (shaft_2 relative to shaft_1)
POWER_W = 10_471.9755          # T = 100 N.m at 1000 rpm
SHAFT1_RPM = 1000.0
ROTATION_DIR_SOURCE = 1
SHAFT_RPM = {
    "shaft_1": SHAFT1_RPM,
    "shaft_2": SHAFT1_RPM * GEAR_Z_DRIVER / GEAR_Z_DRIVEN,
}

# --- beam-model used by the study ----------------------------------------

THEORY: dict[str, af_m.BeamModelSettings] = {
    "timoshenko/cowper": af_m.BeamModelSettings("timoshenko", "cowper", "single_point")
}

# =============================================================================
# Model construction
# =============================================================================

# --- materials ---------------------------------------------------------------
shaft_material = af_c.Material(
    material_id=SHAFT_MATERIAL_ID,
    density=SHAFT_DENSITY_KG_M3,
    elastic=af_c.IsotropicElastic(E=SHAFT_E_MPA, poisson_ratio=SHAFT_POISSON),
    strength=af_c.StrengthProperties(Sut=SHAFT_SUT_MPA, Sy=SHAFT_SY_MPA),
    description="S355 steel",
)

bearing_material = af_c.Material(
    material_id=BEARING_MATERIAL_ID,
    density=BEARING_DENSITY_KG_M3,
    elastic=af_c.IsotropicElastic(E=BEARING_E_MPA, poisson_ratio=BEARING_POISSON),
    description="GCr15 steel",
)

gear_material = af_c.Material(
    material_id=GEAR_MATERIAL_ID,
    density=GEAR_DENSITY_KG_M3,
    elastic=af_c.IsotropicElastic(E=GEAR_E_MPA, poisson_ratio=GEAR_POISSON),
    description="C45 steel",
)

# --- ISO/TS 16281 contact data for the bearings ------------------------------
# e1/nu1: rolling element, e2/nu2: raceways (inner and outer). Same steel here.
BALL_CONTACT = dict(
    contact=af_c.ContactAnalysis.ISO16281,
    e1=bearing_material.E, e2=bearing_material.E,
    nu1=bearing_material.poisson_ratio, nu2=bearing_material.poisson_ratio,
)
ROLLER_CONTACT = dict(BALL_CONTACT)      # ISO/TS 16281 is only valid for steel on steel


def register_materials() -> None:
    """The core registry starts empty and register() refuses duplicates, so this
    is safe to call any number of times."""
    for material in (shaft_material, bearing_material, gear_material):
        if material.material_id not in af_c.materials.base.available_materials():
            af_c.materials.base.register(material)


def build_shaft(label: str, sections: list[tuple[float, float, str]],
                fillet_mm: float = SHOULDER_FILLET_MM) -> af_c.Shaft:
    shaft = af_c.Shaft(label=label)
    for L, d, section_label in sections:
        shaft.add_section(af_c.ShaftSection(length=L, diameter=d, label=f"{label}_{section_label}"))
    for i in range(len(sections) - 1):
        d_left, d_right = sections[i][1], sections[i + 1][1]
        if d_left == d_right:        # equal diameters: plain boundary, no shoulder
            continue
        shaft.set_transition(i, af_c.Shoulder(fillet_mm,
                                         max(d_left, d_right), min(d_left, d_right)))
    return shaft


def _assemble(kind: str, label: str, position: float, arrangement: str, s: float) -> af_c.Bearing:
    if kind == "ball":
        family, catalog_dims = af_c.DeepGrooveBallFamily(), BALL_CATALOG
        geometry = dict(**BALL_GEOMETRY, s=s, **BALL_CONTACT)
    elif kind == "roller":
        family, catalog_dims = af_c.CylindricalRollerFamily(), ROLLER_CATALOG
        geometry = dict(**ROLLER_GEOMETRY, s=s, **ROLLER_CONTACT)
    else:
        raise ValueError(f"unknown bearing kind {kind!r}")
    return af_c.Bearing.assemble(
        family=family,
        catalog=af_c.BearingCatalog(**catalog_dims, position=position,
                                    arrangement=arrangement, label=label),
        geometry=geometry,
    )


def bearing_label(shaft_name: str, kind: str, n: int) -> str:
    return f"{shaft_name}_{kind}_bearing_{n}"


def build_bearings(s: float) -> dict[str, dict[str, af_c.Bearing]]:
    """{shaft_name: {label: Bearing}} for radial internal clearance s [mm].
    Labels do not depend on s, so they always match the FEM bearing nodes."""
    return {
        name: {bearing_label(name, kind, n): _assemble(kind, bearing_label(name, kind, n),
                                                       position, arrangement, s)
               for n, position, arrangement, kind in BEARING_SPECS}
        for name in SHAFT_NAMES
    }


def build_system(s: float = S_REFERENCE_MM) -> af_c.SpurHelicalGearSystem:
    register_materials()

    n_sections = {len(SECTIONS_BY_SHAFT[name]) for name in SHAFT_NAMES}
    if len(n_sections) != 1:
        raise ValueError(f"all shafts must have the same number of sections, got {n_sections}")

    # --- shafts + bearings
    bearings = build_bearings(s)
    shaft_systems: dict[str, af_c.ShaftSystem] = {}
    for name in SHAFT_NAMES:
        shaft_system = af_c.ShaftSystem(build_shaft(name, SECTIONS_BY_SHAFT[name]),
                                   name=name, speed_rpm=SHAFT_RPM[name])
        for bearing in bearings[name].values():
            shaft_system.add_bearing(bearing)
        shaft_systems[name] = shaft_system
    ss1, ss2 = (shaft_systems[name] for name in SHAFT_NAMES)

    # --- gears + mesh
    g_driver = af_c.SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVER, b=GEAR_FACE_WIDTH_MM,
                               position=GEAR_POSITION_MM, label="g_driver",
                               material_id=GEAR_MATERIAL_ID)
    g_driven = af_c.SpurHelicalGear(mn=GEAR_MODULE_MM, z=GEAR_Z_DRIVEN, b=GEAR_FACE_WIDTH_MM,
                               position=GEAR_POSITION_MM, label="g_driven",
                               material_id=GEAR_MATERIAL_ID)
    meshing = af_c.SpurHelicalGearMeshing(g_driver, g_driven, label="stage1")
    ge_driver = af_c.GearElement(g_driver, role="driver", label="g_driver")
    ge_driven = af_c.GearElement(g_driven, role="driven", label="g_driven")
    ss1.add_gear(ge_driver)
    ss2.add_gear(ge_driven)

    # --- link the shafts, validate, resolve the gear-mesh loads
    link = af_c.SpurHelicalMeshLink(ss1, ge_driver, ss2, ge_driven, meshing,
                               phi_deg=MESH_PHI_DEG, label="stage1")
    system = af_c.SpurHelicalGearSystem([ss1, ss2], [link], label="clearance_load_distribution")
    system.validate_or_raise()
    system.resolve(P=POWER_W, rpm=SHAFT1_RPM, rotation_dir_source=ROTATION_DIR_SOURCE)

    # --- the driven shaft only receives the mesh reaction torque: close it with a sink
    net = sum(ld.magnitude for ld in ss2.torque_loads)
    ss2.add_load(af_c.TorqueLoad(position=ss2.shaft.total_length, magnitude=-net,
                            label="output_coupling", source="user"))
    return system


if __name__ == "__main__":
    # quick look at what is built, no solve
    system = build_system()
    for shaft_system in system.shafts:
        print(f"== {shaft_system.name}  ({shaft_system.speed_rpm:.1f} rpm)")
        for bearing in shaft_system.bearings:
            print(bearing.summary())
        for load in shaft_system.loads:
            print("  ", load)