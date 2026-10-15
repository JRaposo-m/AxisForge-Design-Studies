"""
tests/fem_regression_values.py -- expected results of the shaft FEM
validation cases, for regression of AxisForge (``pytest``).

Moved on 2026-10-06 from ``axisforge_bridge/validation/results_library.py``:
regression values belong to the tests, not to the validation cases.

Two kinds of entries, kept apart on purpose:

``FEM_MINIMAL_MESH``
    AxisForge results on the coarsest admissible mesh (mandatory nodes only),
    per case, beam model and shaft. These are **regression gold**, not
    "correct" values: the linear Timoshenko element has ~10 % nodal error on
    this mesh (see each case's notebook 02). Their only job is to change
    when -- and only when -- something in AxisForge or in the case changes
    the numbers. A deliberate change in AxisForge (a fix) means
    regenerating them, with the reason recorded in the commit.

``EXACT``
    Closed-form maxima from ``references/beams`` (Timoshenko,
    Cowper kappa), evaluated on 2001 equally spaced points over the shaft.
    Independent of the AxisForge discretisation; they move only if the
    reference code or the case definition changes.

Generated on 2026-10-06 from the AxisForge ``secondary`` branch (commit
c90c74b), and checked bit-for-bit against the pre-reorganisation scripts.
Values of ``shaft1`` (the driving shaft) regenerated on 2026-10-15 with the
mesh loads of D-0010 (José's rulings Q05, Q06 on the AxisForge branch
``refactor/gears-mechanical-system``): on the driving gear the tangential
force now points along -Z (theta 270 deg), the direction of the user test
load, and the radial force along -Y (180 deg), toward its own axis; before,
they were at 90 and 0 deg. ``shaft2``: |v|max and Mmax unchanged (its radial
force turned from 180 to 0 deg, so the solution in the XY plane changes sign).

Units: v [mm], M [N.mm].
"""
from __future__ import annotations

__all__ = ["FEM_MINIMAL_MESH", "EXACT", "RTOL"]

RTOL = 1e-9
"""Relative tolerance for regression checks (platform/BLAS round-off only)."""

# case -> beam model -> shaft -> (v_max [mm], M_max [N.mm], number of nodes)
FEM_MINIMAL_MESH: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = {
    "fem_uniform_radial_load": {
        "euler_bernoulli": {
            "shaft1": (0.4179504223628188, 251222.5365910839, 12),
            "shaft2": (0.36621404473030983, 227733.97397878167, 12),
        },
        "timoshenko/cowper": {
            "shaft1": (0.3900226237443234, 241739.93705785024, 12),
            "shaft2": (0.34054536031697663, 218245.0583962735, 12),
        },
        "timoshenko/hutchinson": {
            "shaft1": (0.38955396258991, 241739.9370578587, 12),
            "shaft2": (0.3401205218351349, 218245.0583962666, 12),
        },
    },
    "fem_stepped_distributed_gear": {
        "euler_bernoulli": {
            "shaft1": (0.15901564552171946, 278636.0144667345, 18),
            "shaft2": (0.10044650435248095, 194021.33543188724, 18),
        },
        "timoshenko/cowper": {
            "shaft1": (0.15933695877531648, 273996.99787363224, 18),
            "shaft2": (0.10087248683920662, 188908.95815360008, 18),
        },
        "timoshenko/hutchinson": {
            "shaft1": (0.15903308198939153, 273996.9978736333, 18),
            "shaft2": (0.10066628599184625, 188908.95815360002, 18),
        },
    },
}

# case -> shaft -> (v_max [mm], M_max [N.mm]), Timoshenko with Cowper kappa
EXACT: dict[str, dict[str, tuple[float, float]]] = {
    "fem_uniform_radial_load": {
        "shaft1": (0.42912019695192627, 251222.53659113726),
        "shaft2": (0.376339390904566, 227733.9739787203),
    },
    "fem_stepped_distributed_gear": {
        "shaft1": (0.166317210609219, 277377.387999758),
        "shaft2": (0.10542019208894254, 192595.10451188826),
    },
}
