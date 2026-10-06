"""
validation/fem_studies/resolution/common/bearings.py
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings import BearingCatalog, Bearing, CylindricalRollerFamily, DeepGrooveBallFamily

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