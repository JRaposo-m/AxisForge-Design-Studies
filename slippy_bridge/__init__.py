"""slippy_bridge - the only place where design-studies joins AxisForge to SlipPY.

Studies and validation cases import from here, so an import from ``slippy_bridge`` says "this
uses SlipPY". Another contact solver would get its own bridge; the studies change only their
import. Organised by physics, then by machine element:

    contact/            dry contact
        base.py         ContactCase, Kinematics, Contact
        elements/       bearings.py (BearingLocation, bearing_contacts, every AxisForge family
                        with single-row geometry); gears.py (GearLocation, gear_contacts; the
                        conversion is a scaffold)
        reference.py    Hertz (SlipPY hertz_full) on the bridge axes
        surfaces.py     Grid and pair: composite or two_profiles, smooth or rough
                        (rolling = array axis 0)
        rough.py        roughness generation
        static_step.py  SlipPY StaticStep + measurements + checks -> record
        measures.py     measurements of a pressure field, as records
    lubrication/        imports from contact/, never the reverse
        ehl.py          steady EHL solve (scaffold)

Decisions: D-0019 (proposed), D-0018 (reuse first).
"""
from . import contact

__all__ = ["contact"]
