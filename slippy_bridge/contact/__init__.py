"""Dry normal contact of two elastic bodies, in four levels imported in this order (D-0034):

    core       the problem, no SlipPY: Body (abstract), SweptCircleBody, Contact (checks, relative curvatures, psi, E*),
               Kinematics (sub-model data); machine_elements: bearings and gears from AxisForge
    mesh       the grid around a contact and the arguments of the SlipPY step
    results    data only: HertzResult, StaticResult
    solvers    computation with SlipPY: hertz, surface_pair, solve

A level imports only the levels before it. Each conversion is made in one place: mm -> m,
MPa -> Pa and N/mm -> N/m in ``core.machine_elements`` (the AxisForge boundary); a line load
N/m -> N on the period in ``mesh.grid.Grid.step_args``. Rolling direction = bridge x; array axis 0 =
grid x, which is bridge x when ``Grid.angle = 0`` (the default).
"""
from .core.contact import Body, Contact, SweptCircleBody
from .core.machine_elements.bearings import (BallRacewayContact, BearingContact, DeepGrooveBallContact,
                                             RollerRacewayContact, bearing_contacts)
from .core.machine_elements.gears import GearContact, ToothContact, gear_contacts, path_diameters
from .core.submodels import Kinematics
from .mesh.grid import SPAN_X_0, Grid
from .results.hertz import HertzResult
from .results.static import StaticResult
from .solvers.hertz import hertz
from .solvers.static import solve
from .solvers.surfaces import surface_pair

__all__ = ["Body", "SweptCircleBody", "Contact", "Kinematics",
           "BearingContact", "DeepGrooveBallContact", "BallRacewayContact", "RollerRacewayContact",
           "bearing_contacts",
           "GearContact", "ToothContact", "gear_contacts", "path_diameters",
           "Grid", "SPAN_X_0", "HertzResult", "StaticResult", "hertz", "surface_pair", "solve"]
