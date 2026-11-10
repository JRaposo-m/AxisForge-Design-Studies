"""The systems of v105: ball bearings and their load cases, returned NOT solved.

The bearings are v003's (``deep_groove_6210``, ``angular_contact_7210``: 12.7 mm balls on a 70 mm pitch
diameter, grooves 0.52 Dw, steel), imported from v003, not rebuilt (OPEN R5-80, the R2-02 pattern). A load
case is a bearing, its radial and axial load and whether it is judged:

* ``LIGHT``: the 6210 at zero clearance (``alpha_j = alpha_0 = 0``: bridge and AxisForge radii are the same
  contact) under radial loads from 100 N, where the most loaded contact is just inside r02's size ratio 0.1
  (``a/r1``, the ball), to 10 kN, far outside it: the size sweep.
* ``CLEARANCE``: the 6210 with 20 um diametral clearance under 5000 N: the elements run at ``alpha_j = 0``,
  AxisForge's radii are at ``alpha_0`` = 11.4 deg (SB-48): RECORD.
* ``ANGULAR``: the 7210 (``alpha_0`` 40 deg) under 3000 N radial, 4000 N axial: ``alpha_j`` 40.2-42.4 deg,
  RECORD.

Units: mm, N (AxisForge); the bridge and r02 work in SI.
"""
from __future__ import annotations

# OPEN R5-80 (R2-02 pattern): the bearings of v003 are imported, not duplicated
from slippy_bridge.contact.core.machine_elements.bearings import AngularContactBallContact, DeepGrooveBallContact

from studies.v000_axisforge.v003_iso16281_load_zone import angular_contact_7210, deep_groove_6210

__all__ = ["load_cases", "LIGHT_FR_N"]

LIGHT_FR_N = (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)     # [N] size sweep, 6210, s = 0


def load_cases() -> list[dict]:
    """Every load case: ``name``, ``build`` (the bearing, unsolved), ``Fr_N``, ``Fa_N``, ``contact`` (the bridge
    class that translates it) and ``role``."""
    cases = [{"name": f"6210 s 0 Fr {Fr:g} N", "build": lambda: deep_groove_6210(0.0), "Fr_N": Fr, "Fa_N": 0.0,
              "contact": DeepGrooveBallContact, "role": "size sweep"} for Fr in LIGHT_FR_N]
    cases.append({"name": "6210 s 20 um Fr 5000 N", "build": lambda: deep_groove_6210(0.02), "Fr_N": 5000.0,
                  "Fa_N": 0.0, "contact": DeepGrooveBallContact, "role": "RECORD: alpha_j != alpha_0 (SB-48)"})
    cases.append({"name": "7210 Fr 3000 N Fa 4000 N", "build": angular_contact_7210, "Fr_N": 3000.0,
                  "Fa_N": 4000.0, "contact": AngularContactBallContact, "role": "RECORD: alpha_j != alpha_0 (SB-48)"})
    return cases
