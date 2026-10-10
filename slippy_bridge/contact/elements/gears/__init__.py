"""Gears: registered subtypes, adapters to come.

The spur/helical mesh is registered so that the bridge knows the element and its subtype, but
its adapter is not written yet: it needs the position of the contact along the line of action
(and, for lubrication, its change in time), the load per unit length with load sharing, and the
radii of curvature at that position (AxisForge gives the points T1A..T1E and T1T2, not rho(s)).
"""
from ...base import register

__all__ = []


def _spur_helical():
    from axisforge.core.machine_elements.gears.parallel_axis import SpurHelicalGearMeshing
    return SpurHelicalGearMeshing


@register("gear", _spur_helical())
def _spur_helical_contacts(mesh, *args, **kwargs):
    raise NotImplementedError("gear contacts: adapter not written yet (position along the line of "
                              "action, load sharing and rho(s) to be decided; see gears/__init__.py)")
