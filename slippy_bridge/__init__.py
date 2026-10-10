"""slippy_bridge - the only place where design-studies joins AxisForge to SlipPY.

Studies and validation cases import contact geometry and surfaces from here, so an import from
``slippy_bridge`` says "this uses SlipPY". Another contact solver would get its own bridge; the
studies change only their import.

* :class:`ContactCase` - inputs of one Hertzian contact (SI, x = rolling direction).
* :func:`contact_case` - AxisForge bearing (any family) + race + load (+ ``alpha_j``) -> case.
* :func:`grid`, :func:`composite`, :func:`two_profiles` - case -> SlipPY surfaces (static BEM and
  lubrication), rolling along array axis 0.

Decisions: D-0019 (proposed), D-0018 (reuse first).
"""
from .case import ContactCase
from .geometry import contact_case, rolling_radius
from .surfaces import composite, grid, semi_axes, two_profiles

__all__ = ["ContactCase", "contact_case", "rolling_radius", "semi_axes", "grid", "composite",
           "two_profiles"]
