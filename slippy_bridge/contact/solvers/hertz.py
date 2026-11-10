"""Hertz solution of a contact, from SlipPY ``hertz_full``, on the bridge axes.

SlipPY's Hertz (with José's local fixes in ``slippy/contact/hertz.py``) sizes grids, gives the
reference of the BEM solve and is compared with it. The independent judge for validation cases
stays in ``references/contact`` (no SlipPY import, D-0021).

References
----------
Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press. Section 4.2(b), p. 95-98
(elliptical contact); section 4.2(c), p. 100-101 (line contact); section 4.2, list (i)-(iv),
p. 91-92 (assumptions).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from slippy.contact import hertz_full

from ..results.hertz import HertzResult

if TYPE_CHECKING:
    from ..core.contact import Contact

__all__ = ["hertz"]

# OPEN SB-07: 0.1 is a bridge convention (Johnson says a << R, no number); per-body a << R1,2 not checked
SIZE_RATIO_MAX = 0.1           # [-] largest b kappa_max (or a kappa_min) of a valid Hertz reference


def hertz(contact: Contact) -> HertzResult:
    """Hertz pressure, contact ellipse and approach of ``contact``.

    SlipPY ``hertz_full(r1, r2, ..., angle)`` takes the principal radii of each body and the
    ``angle`` between their x axes [rad] (body 2 from body 1, reduced to [-pi/2, pi/2)); it finds
    the principal relative axis at ``alpha`` from body 1's x axis and at ``beta`` from that axis to
    body 2's x axis, ``alpha + beta = angle`` (Johnson 1985 Eqs. (4.4)-(4.5)).

    The bridge passes the equivalent body against a flat, ``r1 = (1/kappa_max, 1/kappa_min)``,
    ``r2 = inf``, at ``angle = 0``: the gap is the same and the rotation is held once, in
    ``contact.psi``. So SlipPY's angle branch (``const_b = kappa_max - kappa_min >= 0``) never
    sees the case it mishandles, two bodies both more curved across than along.

    SlipPY's own ``pressure_f`` must not be used: it puts the major axis along its x whatever the
    curvatures; :meth:`HertzResult.pressure` is the bridge's field.

    Parameters
    ----------
    contact : Contact
        Any contact (``load`` [N] for a point, [N/m] for a line).

    Returns
    -------
    HertzResult
        See :class:`HertzResult` for the source of each value. ``valid`` is False, with the
        reason, when the contact is not small against the relative radii (``size_ratio >
        SIZE_RATIO_MAX``) or reaches the end of a body (``Contact.end_distances``: an end closer than the half-width
        ``b``, ``d_end / b < 1``); the values are returned all the same.

    Raises
    ------
    ValueError
        From SlipPY, if it cannot solve the contact (the contact checks exclude the known cases).
    """
    b1, b2 = contact.body1, contact.body2
    r_across = math.inf if contact.kappa_min == 0 else 1.0 / contact.kappa_min    # a line: flat along y
    sol = hertz_full(r1=(1.0 / contact.kappa_max, r_across), r2=math.inf, moduli=(b1.E, b2.E), v=(b1.nu, b2.nu),
                     load=contact.load, angle=0.0, line=contact.line)
    # SlipPY gives [major, minor] for a point, [half-width, inf] for a line; sorted for safety
    b, a = sorted(float(r) for r in sol["contact_radii"])
    size_ratio = b * contact.kappa_max if contact.line else max(b * contact.kappa_max, a * contact.kappa_min)
    # the ends of the bodies in half-widths: below 1 the Hertz strip itself reaches an end (necessary, not
    # sufficient: Johnson 1985 4.1 (iii) wants the bodies large against the contact, several b below the surface)
    # OPEN SB-71: only d_end/b < 1 invalidates; a near-end margin (e.g. d_end/b < 3) is not flagged
    ends = {key.removesuffix("_m") + "_over_b": d / b for key, d in contact.end_distances().items()}
    reasons = ([f"size_ratio = {size_ratio:.3g} > {SIZE_RATIO_MAX}"] if size_ratio > SIZE_RATIO_MAX else []) + [
        f"{key} = {value:.3g} < 1 (the contact reaches the end of a body)" for key, value in ends.items() if value < 1.0]
    return HertzResult(p0=float(sol["max_pressure"]), a=a, b=b, psi=contact.psi,
                       delta=float(sol.get("total_deflection", math.nan)), b_over_a=b / a,
                       E_star=float(sol["e_star"]), size_ratio=size_ratio, valid=not reasons,
                       line=contact.line, reason="; ".join(reasons), ends_over_b=ends)
