"""
Question: how does the inner-ring misalignment (tilt) psi change the load distribution of a bearing,
and how does the radial internal clearance s change that effect?

Axes:  psi_mrad, prescribed inner-ring tilt projected on the Fr plane [mrad] (x axis of the
       figures); the FEM slope of this system is about 2 mrad (recorded in report.txt as
       psi_fem_mrad)
       s_mm, total radial internal clearance Gr [mm] (one line per value in the figures)
Load:  nominal (100 N m at 1000 rpm), spur gears (Fa = 0)

Only psi >= 0 is swept: a deep groove ball bearing and a symmetric roller are mirror-symmetric,
so -psi gives the same Q_j (delta_a and Mz change sign). This does NOT hold for an angular
contact bearing under Fa (see the planned psi_sign_vs_axial question).

The prescribed psi is applied through the default ``psi`` hook (the ``psi_mrad`` axis); the
clearance is set on the bearing by ``configure``.
"""
from __future__ import annotations

import dataclasses

from _common.bearings import BearingSpec
from _common.study import LoadDistributionStudy


class MisalignmentStudy(LoadDistributionStudy):
    """Sweep of the prescribed inner-ring tilt, for several radial internal clearances."""

    question = "misalignment"
    axes = {
        "psi_mrad": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],   # x axis of the figures
        "s_mm": [0.0, 0.010, 0.020],                          # one line per value
    }
    helix_angle_deg = 0.0

    def configure(self, spec: BearingSpec, *, s_mm: float, **_) -> BearingSpec:
        """The bearing under study with clearance ``s_mm`` (psi is applied by the psi hook).

        Parameters
        ----------
        spec: BearingSpec
            A type with a clearance parameter ``s`` (deep groove, cylindrical roller).
        s_mm: float
            Total radial internal clearance [mm].

        Returns
        -------
        spec: BearingSpec
            A copy with ``s = s_mm``.
        """
        return dataclasses.replace(spec, s=s_mm)
