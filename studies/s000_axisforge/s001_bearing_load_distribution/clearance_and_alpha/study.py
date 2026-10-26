"""
Question: how do the radial internal clearance s and the free contact angle alpha_0 change the
load distribution of a bearing at the nominal load?

The two parameters are one geometric quantity seen from two sides: for a ball bearing
cos(alpha_0) = 1 - s / (2 A), A = ri + re - Dw. Each type sweeps the parameter it is defined by:

    deep_groove          s_mm        (alpha_0 follows from s; reported as alpha0_deg)
    cylindrical_roller   s_mm        (no contact angle)
    angular_contact      alpha0_deg  (s follows from alpha_0; reported as s_mm)

Every row reports both s_mm and alpha0_deg, so compare/ puts the three types on the same x.

Load:  nominal (100 N m at 1000 rpm), spur gears (Fa = 0)
psi:   the FEM slope at the bearing (its own effect is the misalignment question)

Warning: a single-row angular contact bearing with spur gears has no external axial load, but
every loaded ball produces an axial component Q sin(alpha) of the same sign, and the roller
partner carries no axial load. Axial equilibrium is then not possible with a radial load; the
result of this case tests how the solver behaves on it, it does not describe a working
bearing. The physically meaningful angular contact case is the helical sub-case (planned).
"""
from __future__ import annotations

import dataclasses

from _common.bearings import BearingSpec
from _common.study import LoadDistributionStudy

# total radial internal clearance Gr [mm]: 0 -> no clearance; ~0.005-0.020 is the usual CN range of
# a small bearing, C3/C4 above it
S_VALUES_MM = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
# free contact angle [deg], 0 < alpha_0 <= 45
ALPHA0_VALUES_DEG = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


class ClearanceAndAlphaStudy(LoadDistributionStudy):
    """Sweep of the clearance (deep groove, roller) or of the free contact angle (angular)."""

    question = "clearance_and_alpha"
    axes_by_kind = {
        "ball": {"s_mm": S_VALUES_MM},
        "roller": {"s_mm": S_VALUES_MM},
        "angular": {"alpha0_deg": ALPHA0_VALUES_DEG},
    }
    helix_angle_deg = 0.0

    def configure(self, spec: BearingSpec, **point) -> BearingSpec:
        """The bearing under study with the clearance or the contact angle of this point.

        Parameters
        ----------
        spec: BearingSpec
            The bearing given to the study.
        **point: float
            ``s_mm`` [mm] for the deep groove and the roller, ``alpha0_deg`` [deg] for the
            angular contact bearing.

        Returns
        -------
        spec: BearingSpec
            A copy with ``s`` or ``alpha_0_deg`` set.
        """
        if "s_mm" in point:
            return dataclasses.replace(spec, s=point["s_mm"])
        return dataclasses.replace(spec, alpha_0_deg=point["alpha0_deg"])
