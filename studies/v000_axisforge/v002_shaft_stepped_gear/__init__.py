"""v002 shaft stepped gear: AxisForge shaft FEM of a stepped shaft with the loads of a gear mesh against r01 (README)."""
from . import literature
from .check import along_shaft, check, gear_identities, mesh_judge, physical_couples, reference, rigid_body, segments
from .construction import THEORIES, gear_system, stepped_shaft, uniform_nodes

__all__ = ["check", "reference", "segments", "rigid_body", "along_shaft", "gear_identities", "mesh_judge", "physical_couples",
           "literature",
           "gear_system", "stepped_shaft", "uniform_nodes", "THEORIES"]
