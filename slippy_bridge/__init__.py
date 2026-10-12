"""slippy_bridge - the only place where design-studies joins AxisForge to SlipPY.

Studies and validation cases import from here, so an import from ``slippy_bridge`` says "this
uses SlipPY". Another contact solver would get its own bridge; the studies change only their
import. Organised by physics, then by machine element:

    contact/            dry contact
        base.py         ContactCase, Contact (element, subtype, location, kinematics),
                        ElementAdapter and its registry
        elements/       bearings/ (BearingContacts, every AxisForge family), gears/
                        (SpurHelicalContacts, registered, adapter to come)
        reference.py    Hertz (SlipPY hertz_full) on the bridge axes
        smooth.py       Grid and pair(case, grid, build, rough): composite or two_profiles
                        surfaces, smooth or rough (rolling = array axis 0)
        rough.py        roughness generation
        static_step.py  SlipPY StaticStep + measurements + boolean checks -> record
        outputs.py      measurements as records (written with axisforge.outputs._format)
    lubrication/        (to come) imports from contact/, never the reverse

Decisions: D-0019 (proposed), D-0018 (reuse first).
"""
from . import contact

__all__ = ["contact"]
