"""
validation/abaqus_comparison/scripts/common/bearings.py

Shared bearing geometry + fixed BC orientation for every case script in
this validation suite. Imported by every leaf case_*.py so the bearing
stiffness/BC never drifts between case groups -- what varies from one
case script to the next is the LOAD (type, position, combination) or the
beam theory (timoshenko/euler), never the bearing set-up.

Fixed BC orientation (per the explicit decision that split BC sweeping
out of the load matrix -- see the suite's top-level README.md):

    x=10.0 mm   ball (DeepGrooveBallFamily)      arrangement="locating"
    x=190.0 mm  roller (CylindricalRollerFamily)  arrangement="non-locating"

Ball is always locating -- CylindricalRollerFamily.assemble_geometry()
raises ValueError on arrangement="locating" (an NU/N-type roller bearing
has no flange to react axial load). If/when a dedicated BC-sweep case
group is added (the other orientation, or a floating/floating pair),
it belongs in its own folder, not mixed into this load/theory matrix.

Usage (inside a case_*.py, AFTER resolving ConstructionCapabilities).
Every case_*.py bootstraps sys.path by walking UP to the ancestor
directory literally named "scripts" (depth-agnostic -- works no matter
how many levels separate the case file from scripts/, see
common/paths.py's own docstring), then imports this module as a plain
top-level "common" package (this folder is not set up as an installed/
dotted package):

    import sys
    from pathlib import Path
    _p = Path(__file__).resolve()
    while _p.name != "scripts":
        _p = _p.parent
    sys.path.insert(0, str(_p))
    from common.bearings import build_case_bearings

    objs = construction.resolve()
    bearings = build_case_bearings(
        objs["make_deep_groove_ball_bearing"],
        objs["make_cylindrical_roller_bearing"],
        name="shaft1",
    )
"""
from __future__ import annotations

from typing import Callable

BALL_KW = dict(d=20.0, D=42.0, Dw=7.0, Dpw=31.0, Z=9, E=25.0, s=0.02)

# NU204-like -- dims taken from CylindricalRollerFamily's own docstring
# example (core/machine_elements/Bearings/families/roller_bearing/radial/
# subtypes/cylindrical_roller.py), so at least a real catalogue shape,
# even if arbitrarily paired with the ball bearing above.
ROLLER_KW = dict(
    d=20.0, D=47.0, b=14.0, C=28_500.0, C0=22_000.0,
    Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1,
)

BALL_POSITION_MM = 10.0
ROLLER_POSITION_MM = 190.0


def build_case_bearings(
    make_deep_groove_ball_bearing: Callable[..., "Bearing"],
    make_cylindrical_roller_bearing: Callable[..., "Bearing"],
    name: str,
) -> tuple["Bearing", "Bearing"]:
    """
    Fixed-orientation (ball locating @10mm, roller non-locating @190mm)
    bearing pair for one shaft, labelled with `name`. Returned already
    ordered left-to-right (ball first) since BALL_POSITION_MM <
    ROLLER_POSITION_MM -- keep it that way if these constants ever change,
    ShaftSystem/report summaries assume bearings read left-to-right.

    make_deep_groove_ball_bearing / make_cylindrical_roller_bearing :
        the factory functions from ConstructionCapabilities.resolve()
        (capabilities "bearings.deep_groove_ball" /
        "bearings.cylindrical_roller") -- passed in rather than imported
        here, so this module stays capability-agnostic and every case
        script keeps declaring its own ConstructionCapabilities.
    """
    ball = make_deep_groove_ball_bearing(
        **BALL_KW,
        position=BALL_POSITION_MM,
        arrangement="locating",
        label=f"{name}_ball_locating",
    )
    roller = make_cylindrical_roller_bearing(
        **ROLLER_KW,
        position=ROLLER_POSITION_MM,
        arrangement="non-locating",
        label=f"{name}_roller_nonlocating",
    )
    return (ball, roller)