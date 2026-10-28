"""The system of s006: one parallel cylindrical stage (H501 m_n, z, x) on two uniform shafts, loads resolved, NOT solved.

Two identical uniform steel shafts (d 40 mm, S355) on ideal pins at x = 0 and x = L (the span), the
pinion on one, the wheel on the other, at the same axial station. Three gear positions:

* ``"centred"``: x_g = 0.5 L, shaft length L;
* ``"offset"``: x_g = 0.3 L, shaft length L;
* ``"overhung"``: x_g = 1.2 L (0.2 L outside pin B), shaft length 1.2 L + b/2 (the face ends at the shaft end).

The pair takes m_n, z and x of the GEARpie H501 pair of AxisForge's baseline cases (m_n 3.5 mm, z 20/30,
x 0.1809/0.0891, b 23 mm, alpha_n 20 deg, basic rack 1/1.25/0.38) with the helix angle and the hand as
inputs (H501 has beta 15 deg; its hand is not in the GEARPIE record, both hands are run); the centre distance
follows from x (``GearPair.assemble``,
ISO 21771 Eq. (55)-(58)). One torque on the pinion, 60.75 N.m (the baseline T), at 1000 rpm, sense +1.
"Both drivers" are two transmissions written from either shaft (``Transmission`` has no driver
argument, proposals.md s006): the source is the pinion shaft or the wheel shaft, the power the same
(T_pinion omega_pinion), so the pinion torque is the same in both.

Assumptions: linear elastic shafts, rigid pins (no bearing compliance: s004), gear bodies rigid and
fixed to their shaft cross-section at x_g (their axis turns with the section rotation), ideal mesh,
mesh forces independent of the deflections (``Transmission.resolve``), no housing. Torque input at
x = 0 of the source shaft, sink at the free end of the other shaft.

Units: mm, N, MPa, N.mm, rad (AxisForge); torque N.m, power W, speed rpm at the gear interface.
"""
from __future__ import annotations

import math

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair
from axisforge.core.mechanical_system import GearElement, PairStage, ResolvedTransmission, Transmission

from studies.v000_axisforge.v001_shaft_uniform import uniform_shaft

__all__ = ["stage", "gear_station", "shaft_length", "POSITIONS", "SPANS_MM", "BETAS_DEG", "MODULE_MM", "Z", "X",
           "FACE_WIDTH_MM", "ALPHA_N_DEG", "DIAMETER_MM", "TORQUE_NM", "SPEED_RPM", "PHI_DEG"]

# --- pair: m_n, z, x of H501 (AxisForge baseline_cases.GEARPIE["H501"]) ---------------------------
MODULE_MM = 3.5
Z = (20, 30)                    # pinion (gear 1), wheel (gear 2)
X = (0.1809, 0.0891)            # profile-shift coefficients [-]
FACE_WIDTH_MM = 23.0
ALPHA_N_DEG = 20.0              # basic rack haP 1, cP 0.25 (hfP 1.25), rfP 0.38: the AxisForge defaults
PHI_DEG = 0.0                   # line of centres along +Y (pinion axis -> wheel axis); the results are frame-free

# --- shafts and load ------------------------------------------------------------------------------
DIAMETER_MM = 40.0              # both shafts, fixed (proposals.md s006); pinion root circle ~ 63 mm
TORQUE_NM = 60.75               # on the pinion [N.m] (baseline T_NM)
SPEED_RPM = 1000.0              # pinion [rpm]
ROTATION = 1                    # pinion sense about +X

# --- sweep (proposals.md s006) ----------------------------------------------------------------------
SPANS_MM = (100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0)
POSITIONS = {"centred": 0.5, "offset": 0.3, "overhung": 1.2}      # x_g / L
BETAS_DEG = (0.0, 15.0, 30.0)


def gear_station(span_mm: float, position: str) -> float:
    """Axial station of the face centre x_g [mm] on both shafts: ``POSITIONS[position] * span``."""
    return POSITIONS[position] * span_mm


def shaft_length(span_mm: float, position: str) -> float:
    """Shaft length [mm]: the span, or for the overhung gear 1.2 L + b/2 (the face ends at the shaft end)."""
    return span_mm + (0.2 * span_mm + 0.5 * FACE_WIDTH_MM if position == "overhung" else 0.0)


def stage(*, span_mm: float, position: str, beta_deg: float = 0.0, hand: int = 1, driver: str = "pinion",
          torque_Nm: float = TORQUE_NM) -> ResolvedTransmission:
    """The two shafts with the mesh loads of the stage, resolved by ``Transmission``, NOT solved.

    Parameters
    ----------
    span_mm : float
        Distance between the pins [mm] (pins at 0 and L on both shafts).
    position : str
        ``"centred"``, ``"offset"`` or ``"overhung"`` (``POSITIONS``).
    beta_deg : float
        Helix angle at the reference cylinder [deg]; 0 = spur (no axial force, no couple).
    hand : int
        Hand of the pinion, +1 right or -1 left; the wheel has the other hand. Ignored for spur.
    driver : str
        ``"pinion"`` (source = pinion shaft) or ``"wheel"`` (source = wheel shaft).
    torque_Nm : float
        Torque on the pinion [N.m]; the source power is ``torque_Nm * omega_pinion`` [W] either way.

    Returns
    -------
    ResolvedTransmission
        ``shafts`` = (pinion shaft, wheel shaft) in that order whatever the driver (``shafts[0]``
        holds gear 1, as ``v002.gear_identities`` reads it); ``meshes``; ``checks`` (closure).
    """
    if driver not in ("pinion", "wheel"):
        raise ValueError(f"driver must be 'pinion' or 'wheel', got {driver!r}")
    x_g, length = gear_station(span_mm, position), shaft_length(span_mm, position)
    shafts = [uniform_shaft(length_mm=length, diameter_mm=DIAMETER_MM, supports_mm=(0.0, span_mm), loads=[],
                            label=name) for name in ("pinion_shaft", "wheel_shaft")]
    family = CylindricalFamily()
    helical = beta_deg != 0.0
    hands = (hand, -hand) if helical else (0, 0)       # an external pair has opposite hands (D-0010 item 17)
    gears = [Gear.assemble(family, CylindricalGearData(mn=MODULE_MM, z=z, x=x, b=FACE_WIDTH_MM, alpha_n_deg=ALPHA_N_DEG,
                                                       beta_n_deg=beta_deg, hand=h, position=x_g))
             for z, x, h in zip(Z, X, hands)]
    pair = GearPair.assemble(family, gear1=gears[0], gear2=gears[1], label="H501-like")
    elements = [GearElement(g, label=name) for g, name in zip(gears, ("pinion", "wheel"))]
    for ss, ge in zip(shafts, elements):
        ss.add_gear(ge)
    # the path is written from the driving shaft; phi is the line of centres from shaft_a to shaft_b
    a, b = (0, 1) if driver == "pinion" else (1, 0)
    phi = PHI_DEG if driver == "pinion" else PHI_DEG + 180.0       # e_c stays pinion -> wheel along +Y
    link = PairStage(pair=pair, shaft_a=shafts[a], gear_a=elements[a], shaft_b=shafts[b], gear_b=elements[b],
                     phi_deg=phi, label="stage")
    system = Transmission(shafts, [link], source=shafts[a], sinks=((shafts[b], shaft_length(span_mm, position)),),
                          label=f"s006_{position}_L{span_mm:g}_b{beta_deg:g}_h{hand:+d}_{driver}")
    n_source = SPEED_RPM if driver == "pinion" else SPEED_RPM * Z[0] / Z[1]
    P_W = torque_Nm * SPEED_RPM * 2.0 * math.pi / 60.0               # T_pinion omega_pinion [W]
    return system.resolve(P_W=P_W, n_rpm=n_source, rotation=ROTATION if driver == "pinion" else -ROTATION)
