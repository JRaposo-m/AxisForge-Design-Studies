"""
validation/fem_studies/resolution/common/bearings.py

Shared bearing geometry + fixed BC orientation for every case script in
this validation suite. Imported by every leaf case_*.py so the bearing
set-up never drifts between case groups -- what varies from one case
script to the next is the LOAD (type, position, combination) or the
beam theory (timoshenko/euler), never the bearing set-up.

Fixed BC orientation:

    x=10.0 mm   ball (DeepGrooveBallFamily)      arrangement="locating"
    x=190.0 mm  roller (CylindricalRollerFamily)  arrangement="non-locating"

Ball is always locating -- CylindricalRollerFamily.assemble_geometry()
raises ValueError on arrangement="locating" (no flange to react axial
load). A BC-sweep case group belongs in its own folder.

CHANGED: builds the bearings directly from the AxisForge core
(Bearing.assemble + BearingCatalog + the two families) instead of
receiving the old fixtures' make_* factories, which no longer exist.
build_case_bearings() takes only `name` now.

Dropped because the core no longer takes them:
  - C, C0 (roller): computed by the family from the assembled geometry.
  - E (ball): only used with contact=ContactAnalysis.ISO16281 (as
    e1/e2/nu1/nu2); contact stays NONE here.
ASSUMPTION: ball width b=12.0 mm (6204). The old ball set had no b
(BearingCatalog.b defaults to 0.0). No effect on the FEM result.
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily,
    DeepGrooveBallFamily,
)

BALL_CATALOG_KW = dict(d=20.0, D=42.0, b=12.0, designation="6204")
BALL_GEOMETRY_KW = dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02)

ROLLER_CATALOG_KW = dict(d=20.0, D=47.0, b=14.0, designation="NU204")
ROLLER_GEOMETRY_KW = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1)

BALL_POSITION_MM = 10.0
ROLLER_POSITION_MM = 190.0


def build_case_bearings(name: str) -> tuple[Bearing, Bearing]:
    """Ball locating @10mm, roller non-locating @190mm, labelled with
    `name`, returned left-to-right (ball first)."""
    ball = Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(
            **BALL_CATALOG_KW,
            position=BALL_POSITION_MM,
            arrangement="locating",
            label=f"{name}_ball_locating",
        ),
        geometry=dict(BALL_GEOMETRY_KW),
    )
    roller = Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(
            **ROLLER_CATALOG_KW,
            position=ROLLER_POSITION_MM,
            arrangement="non-locating",
            label=f"{name}_roller_nonlocating",
        ),
        geometry=dict(ROLLER_GEOMETRY_KW),
    )
    return (ball, roller)