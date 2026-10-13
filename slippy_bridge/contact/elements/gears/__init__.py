"""Gears: registered subtypes, adapters to come.

The cylindrical pair (spur, helical, internal: one ``GearPair`` class, its subtype is the family,
as for a bearing) is registered so that the bridge knows the element and its subtype, but
its adapter is not written yet: it needs the position of the contact along the line of action
(and, for lubrication, its change in time), the load per unit length with load sharing, and the
radii of curvature at that position (AxisForge gives the points T1A..T1E and T1T2, not rho(s)).
"""
from axisforge.core.machine_elements.gears import CylindricalFamily

from ...base import Contact, ElementAdapter, register

__all__ = ["SpurHelicalContacts"]


@register(CylindricalFamily)
class SpurHelicalContacts(ElementAdapter):
    """Adapter of an AxisForge ``GearPair`` of the cylindrical family: registered, not written yet (see the module text)."""

    element = "gear"

    @classmethod
    def contacts(cls, mesh, *args, **kwargs) -> list[Contact]:
        raise NotImplementedError("gear contacts: adapter not written yet (position along the line of "
                                  "action, load sharing and rho(s) to be decided; see gears/__init__.py)")
