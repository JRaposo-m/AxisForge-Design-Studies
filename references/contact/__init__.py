"""
references.contact -- Hertz contact, written from Johnson 1985, ch. 4: ``hertz.hertz_point``
(elliptical point contact) and ``hertz.hertz_line`` (line contact, plane strain), with the surface
pressure ``hertz.hertz_pressure`` and the subsurface stresses ``hertz.hertz_line_stress`` (McEwen)
and ``hertz.hertz_point_axis_stress`` (circular contact, on the axis). It is the judge
of AxisForge's ``PointContactStiffness`` and of ``slippy.contact.hertz_full``, and imports neither.
``ehl`` (PROVISIONAL, scaffold 2026-10-17): EHL dimensionless groups and published film formulas,
the judge of ``slippy_bridge.lubrication.ehl.run``.
"""
from .ehl import canzi_moes, celine, e_prime, film_scalings, groups, moes, moes_inverse

__all__ = ["e_prime", "groups", "canzi_moes", "moes_inverse", "moes", "celine", "film_scalings"]
