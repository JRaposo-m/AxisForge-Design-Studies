"""slippy_bridge - the only place where design-studies joins AxisForge to SlipPY (D-0019).

Studies and validation cases import from here, so an import from ``slippy_bridge`` says "this
uses SlipPY". Another contact solver would get its own bridge; the studies change only their
import.

    contact/    dry normal contact; its levels and their order: ``slippy_bridge.contact`` (D-0034)

Lubrication will be a sibling of ``contact/``, importing from it and never the reverse.

Decisions: D-0019, D-0034 (proposed), D-0018 (reuse first).
"""
from . import contact

__all__ = ["contact"]
