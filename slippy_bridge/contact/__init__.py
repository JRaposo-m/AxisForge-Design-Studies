"""Dry normal contact of two elastic bodies, in five levels imported in this order (D-0034, roughness R4-01):

    core       the problem, no SlipPY: Body (abstract), SweptCircleBody, Contact (checks, relative curvatures, psi, E*),
               Kinematics (sub-model data); machine_elements: bearings and gears (InvoluteFlank) from AxisForge
    mesh       the grid around a contact and the arguments of the SlipPY step
    roughness  rough heights (macro: RoughnessSpec, rough_heights, rough_measures) and the periodic patch
               (micro); imported as ``contact.roughness.macro`` / ``.micro``, not re-exported here
    results    data only: HertzResult, StaticResult
    solvers    computation with SlipPY: hertz, surface_pair, solve (both take rough ``heights``)

A level imports only the levels before it. Each conversion is made in one place: mm -> m,
MPa -> Pa and N/mm -> N/m in ``core.machine_elements`` (the AxisForge boundary); a line load
N/m -> N on the period in ``mesh.grid.Grid.step_args``. Rolling direction = bridge x; array axis 0 =
grid x, which is bridge x when ``Grid.angle = 0`` (the default).
"""
from .core.contact import Body, Contact, SweptCircleBody
from .core.machine_elements.bearings import (AngularContactBallContact, BearingContact, CylindricalRollerContact,
                                             DeepGrooveBallContact)
from .core.machine_elements.gears import CylindricalGearContact, GearContact, InvoluteFlank
from .core.submodels import Kinematics
from .mesh.grid import SPAN_X_0, Grid
from .results.hertz import HertzResult
from .results.static import StaticResult
from .solvers.hertz import hertz
from .solvers.static import solve
from .solvers.surfaces import surface_pair

__all__ = ["Body", "SweptCircleBody", "Contact", "Kinematics",
           "BearingContact", "DeepGrooveBallContact", "AngularContactBallContact", "CylindricalRollerContact",
           "InvoluteFlank", "GearContact", "CylindricalGearContact",
           "Grid", "SPAN_X_0", "HertzResult", "StaticResult", "hertz", "surface_pair", "solve"]
