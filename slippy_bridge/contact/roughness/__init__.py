"""Roughness of the contact surfaces, at two scales; constructions only (the solver is not changed).

    macro    rough surfaces on the curved bodies of a contact (RoughnessSpec, rough_heights,
             rough_surface_pair, rough_measures)
    micro    the periodic, nominally flat rough patch of a later contact-law sub-model
             (PatchSpec, periodic_patch)

The fifth level of ``slippy_bridge.contact`` (after core, mesh, results, solvers, D-0034): it may
import every level before it, and nothing imports it. ``micro`` imports ``macro``, never the
reverse. Roughness may later serve ``lubrication/`` too (mixed EHL); it stays here until then
(OPEN R4-01).
"""
from . import macro

__all__ = ["macro"]
