"""Roughness of the contact surfaces, at two scales: constructions and measures (the solver takes the heights).

    macro    rough surfaces on the curved bodies of a contact (RoughnessSpec, rough_heights,
             rough_measures); ``solvers.solve(..., heights=(h1, h2))`` solves them
    micro    the periodic, nominally flat rough patch of a later contact-law sub-model
             (PatchSpec, periodic_patch); later its law (micro/law.py, a later run)

The third level of ``slippy_bridge.contact`` (core -> mesh -> roughness -> results -> solvers, D-0034
amended by R4-01): it may import core and mesh, and results and solvers may import it. ``micro``
imports ``macro``, never the reverse. Roughness may later serve ``lubrication/`` too (mixed EHL); it stays
here until then (OPEN R4-01).
"""
from . import macro, micro

__all__ = ["macro", "micro"]
