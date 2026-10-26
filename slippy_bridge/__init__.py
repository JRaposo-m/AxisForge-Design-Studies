"""slippy_bridge - the only place where design-studies joins AxisForge to SlipPY (D-0019).

Studies and validation cases import from here, so an import from ``slippy_bridge`` says "this
uses SlipPY". Another contact solver would get its own bridge; the studies change only their
import. ``contact/`` holds dry normal contact, laid out as AxisForge is, in dependency order
(D-0034); a level imports only the levels above it in this list. Target tree:

    contact/
        core/               the problem, no SlipPY import
            contact.py      Body (signed radii, exact profile), Contact (abstract: checks,
                            relative curvatures, psi, E*)
            submodels.py    Kinematics, data of the sub-models (not used by Hertz or the BEM)
            machine_elements/
                            AxisForge -> Contact, the only place that reads AxisForge models
                            (mm, MPa -> m, Pa): bearings.py, gears.py
        mesh/grid.py        Grid around a contact, arguments of the SlipPY step
        results/            data only: hertz.py (HertzResult), static.py (StaticResult)
        solvers/            computation: hertz.py (SlipPY hertz_full), surfaces.py (SlipPY
                            surfaces of the two bodies), static.py (SlipPY static step)

Lubrication will be a sibling of ``contact/``, importing from it and never the reverse.
Conventions: SI units inside; rolling direction = bridge x = array axis 0.

Decisions: D-0019, D-0034 (proposed), D-0018 (reuse first).
"""
from . import contact

__all__ = ["contact"]
