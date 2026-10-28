"""The system of v002: two stepped shafts joined by a cylindrical gear pair, loads resolved, NOT solved.

Two identical stepped solid shafts (S355, L = 200 mm) with five sections and four filleted
shoulders, each on two ideal pins; a cylindrical gear pair (spur or helical, either hand) at
mid-span joins them. ``Transmission.resolve`` turns the power into the mesh loads on both shafts
through ``mesh_forces_to_shaft_loads`` (the only place where a mesh force becomes shaft loads,
D-0010 item 19): the radial force F_c and the tangential force F_t on the shaft axis, the axial
force F_a, the torque, and, for a helical pair, the bending couple of F_a at the working pitch
radius, an ``ExternalMoment`` of magnitude F_a r_w. The FEM solve is the caller's.

====================  ===========  =======  ======================================
section               x [mm]       d [mm]   role
====================  ===========  =======  ======================================
bearing seat A        0 - 20       20       pin A (locating) at x = 10
body                  20 - 85      28
gear seat             85 - 115     25       gear at x = 100, face width 20 (90-110)
body                  115 - 180    28
bearing seat B        180 - 200    20       pin B (non-locating) at x = 190
====================  ===========  =======  ======================================

Ported from ``v000_axisforge/v002_shaft_fem_stepped_distributed_gear`` at the tag
``studies-pre-sweep`` (sections, fillets, material, gear pair, power, the two-shaft ``Transmission``
with the sink of the driven shaft). Changed: pins of zero width in place of the 6204 / NU204 of
width 12 / 14 mm (v001, ``# OPEN R5-21``), so a pin is one node; the gear face width 15 -> 20 mm and
no parabolic user load, so every support, step, load and face limit is a multiple of 5 mm and a
uniform mesh exists from 40 elements up (``# OPEN R5-32``); helix angle, hand and the angle of the
line of centres are inputs (the old case was a spur pair at phi = 0).

Assumptions: linear elastic isotropic material, small displacements, rigid supports (v = 0 at each
pin, u = 0 at the locating one, rotation free: ``fem_solvers/constraints/boundary_conditions.py``);
section constant per segment, fillets do not change EI or kGA; ideal mesh (no losses), one mesh
force at the pitch point at mid face width, independent of the deflections (``Transmission.resolve``).

Units: mm, N, MPa, N.mm, rad (AxisForge); power W, speed rpm, torque N.m at the gear interface
(AxisForge's own; nothing is converted in this case).
"""
from __future__ import annotations

import math

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.mechanical_system import GearElement, PairStage, ResolvedTransmission, ShaftSystem, Transmission

# OPEN R5-30: material registration, mesh helper and beam-model presets are v001's (a v importing a v, as R2-02)
from studies.v000_axisforge.v001_shaft_uniform.construction import THEORIES, add_pins, ensure_material, uniform_nodes

__all__ = ["gear_system", "stepped_shaft", "uniform_nodes", "THEORIES", "SECTIONS", "LENGTH_MM", "SUPPORTS_MM",
           "STEPS_MM", "GEAR_POSITION_MM", "N_ELEMENTS", "BETA_DEG", "HAND", "PHI_DEG", "TORQUE_NM", "SPEED_RPM", "ROTATION",
           "MODULE_MM", "Z", "ALPHA_N_DEG", "FACE_WIDTH_MM"]

# --- shaft: (length [mm], outer diameter [mm], label), left to right (old case) -------------------
SECTIONS: tuple[tuple[float, float, str], ...] = (
    (20.0, 20.0, "bearing_seat_A"),
    (65.0, 28.0, "body_left"),
    (30.0, 25.0, "gear_seat"),
    (65.0, 28.0, "body_right"),
    (20.0, 20.0, "bearing_seat_B"),
)
SHOULDER_FILLET_MM = 1.0       # stress raiser only: EI and kGA stay those of the abrupt step
LENGTH_MM = sum(length for length, _, _ in SECTIONS)
STEPS_MM = tuple(sum(length for length, _, _ in SECTIONS[:k]) for k in range(1, len(SECTIONS)))   # 20, 85, 115, 180

# --- supports: ideal pins (rigid, rotation free) --------------------------------------------------
SUPPORTS_MM = (10.0, 190.0)    # (locating, non-locating) [mm]: the old bearing centres
# OPEN R5-21 (v001): no point-support object in AxisForge; v001 add_pins places 6204s of zero width

# --- gear pair: ISO 21771 inputs, no profile shift (so d_w = d and alpha_wt = alpha_t) -------------
MODULE_MM = 2.0                # normal module m_n [mm]
Z = (20, 40)                   # tooth numbers (pinion on shaft1 = gear 1, wheel on shaft2 = gear 2)
ALPHA_N_DEG = 20.0             # normal pressure angle of the basic rack [deg]
FACE_WIDTH_MM = 20.0           # OPEN R5-32: 15 mm in the old case; 20 mm puts the face limits on the 5 mm grid
GEAR_POSITION_MM = 100.0       # centre of the face width on both shafts [mm]
BETA_DEG = 20.0                # base case: helix angle at the reference cylinder [deg]
HAND = 1                       # base case: hand of gear 1 (+1 right, -1 left); gear 2 has the other hand
PHI_DEG = 30.0                 # base case: line of centres from +Y toward +Z [deg]; both planes carry load

# --- power: T = 100 N.m on the pinion at 1000 rpm, sense +1 about +X (old case) --------------------
TORQUE_NM = 100.0
SPEED_RPM = 1000.0
ROTATION = 1

# --- uniform mesh levels: h = 5 ... 0.625 mm --------------------------------------------------------
N_ELEMENTS = (40, 80, 160, 320)


def stepped_shaft(name: str, material_id: str | None = None) -> Shaft:
    """The stepped shaft of this case (``SECTIONS``) with a filleted shoulder at every diameter change.

    Parameters
    ----------
    name : str
        Label of the shaft and prefix of its section labels.
    material_id : str, optional
        A registered material; None: S355 (v001 ``ensure_material``).

    Returns
    -------
    Shaft
        Five sections, four ``Shoulder`` transitions of fillet radius 1 mm.
    """
    material_id = material_id or ensure_material()
    shaft = Shaft(label=name)
    for length, d, label in SECTIONS:
        shaft.add_section(ShaftSection(length=length, diameter=d, material_id=material_id, label=f"{name}_{label}"))
    for i in range(len(SECTIONS) - 1):
        d_left, d_right = SECTIONS[i][1], SECTIONS[i + 1][1]
        shaft.set_transition(i, Shoulder(SHOULDER_FILLET_MM, max(d_left, d_right), min(d_left, d_right)))
    return shaft


def gear_system(*, beta_deg: float = BETA_DEG, hand: int = HAND, phi_deg: float = PHI_DEG,
                distribute: bool = False, torque_Nm: float = TORQUE_NM, label: str = "v002") -> ResolvedTransmission:
    """The two stepped shafts with the mesh loads of the gear pair, resolved by ``Transmission``, NOT solved.

    Parameters
    ----------
    beta_deg : float
        Helix angle at the reference cylinder [deg], >= 0; 0 is a spur pair (no axial force, no couple).
    hand : int
        Hand of gear 1, +1 right or -1 left (gear 2 gets the other hand, an external pair); ignored
        for a spur pair.
    phi_deg : float
        Angle of the line of centres from the axis of shaft1 to the axis of shaft2, from +Y toward
        +Z [deg] (``PairStage.phi_deg``): F_c lies along phi, F_t along phi + 90 deg, and the couple
        acts in the plane of X and phi.
    distribute : bool
        False: F_c and F_t are point loads at the gear position; True: uniform over the face width
        (``PairStage.distribute_loads``).
    torque_Nm : float
        Torque the source delivers to shaft1 [N.m]; the power is ``torque_Nm * omega`` [W] at
        ``SPEED_RPM``, sense ``ROTATION`` about +X.
    label : str
        Label of the transmission.

    Returns
    -------
    ResolvedTransmission
        ``shafts``: loaded copies of shaft1 (driver, pinion = gear 1, torque input at x = 0) and
        shaft2 (driven, wheel = gear 2, torque sink at x = L); ``meshes``: the ``MeshForces``;
        ``checks``: the closure checks of ``Transmission`` (action = reaction, torque = moment of the
        mesh force, power of the mesh, separating radial force, torque sum per shaft, train power).
    """
    material_id = ensure_material()
    helical = beta_deg != 0.0
    shafts = []
    for name, rpm in (("shaft1", SPEED_RPM), ("shaft2", SPEED_RPM * Z[0] / Z[1])):
        ss = ShaftSystem(stepped_shaft(name, material_id), name=name, speed_rpm=rpm)
        add_pins(ss, SUPPORTS_MM, label=f"{name}_pin")
        shafts.append(ss)
    family = CylindricalFamily()
    # an external pair has opposite hands (D-0010 item 17); a spur gear has hand 0
    hands = (hand, -hand) if helical else (0, 0)
    gears = [Gear.assemble(family, CylindricalGearData(mn=MODULE_MM, z=z, b=FACE_WIDTH_MM, alpha_n_deg=ALPHA_N_DEG,
                                                       beta_n_deg=beta_deg, hand=h, position=GEAR_POSITION_MM,
                                                       label=f"g{k + 1}"))
             for k, (z, h) in enumerate(zip(Z, hands))]
    pair = GearPair.assemble(family, gear1=gears[0], gear2=gears[1], label="stage1")
    elements = [GearElement(g, label=f"g{k + 1}") for k, g in enumerate(gears)]
    for ss, ge in zip(shafts, elements):
        ss.add_gear(ge)
    stage = PairStage(pair=pair, shaft_a=shafts[0], gear_a=elements[0], shaft_b=shafts[1], gear_b=elements[1],
                      phi_deg=phi_deg, distribute_loads=distribute, label="stage1")
    # the output coupling of shaft2 at x = L closes its torque (AxisForge Q16: every output shaft needs a sink)
    system = Transmission(shafts, [stage], source=shafts[0], sinks=((shafts[1], LENGTH_MM),), label=label)
    omega = SPEED_RPM * 2.0 * math.pi / 60.0           # [rad/s]
    return system.resolve(P_W=torque_Nm * omega, n_rpm=SPEED_RPM, rotation=ROTATION)
