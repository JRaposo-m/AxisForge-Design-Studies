"""Hertz reference of a contact, from SlipPY's ``hertz_full``, oriented on the bridge axes.

Used to size grids, give initial guesses and compare. It is SlipPY's Hertz (with João's local
fixes in ``slippy/contact/hertz.py``); the independent judge for validation cases stays in
``references/contact`` (no SlipPY import).

``hertz_full`` returns ``contact_radii`` in an order not tied to x/y; here ``s_x`` is always the
semi-axis along rolling (array axis 0) and ``s_y`` across it (``inf`` for a line contact).
"""
from __future__ import annotations

import math

import numpy as np

from .base import ContactCase

__all__ = ["relative_radii", "semi_axes", "hertz"]


def relative_radii(case: ContactCase) -> tuple[float, float]:
    """Relative radii ``(R_x, R_y)`` [m], ``1/R = 1/r1 + 1/r2`` per direction (``inf`` = flat).

    Raises
    ------
    ValueError
        If a relative curvature is negative (the bodies do not form a Hertz contact).
    """
    out = []
    for a, b in zip(case.r1, case.r2):
        k = (0.0 if math.isinf(a) else 1.0 / a) + (0.0 if math.isinf(b) else 1.0 / b)
        if k < 0:
            raise ValueError(f"negative relative curvature {k}: the bodies do not form a Hertz contact")
        out.append(math.inf if k == 0 else 1.0 / k)
    return tuple(out)


def semi_axes(case: ContactCase) -> tuple[float, float]:
    """Hertz contact semi-axes ``(s_x, s_y)`` [m] along the rolling direction and across it.

    ``inf`` on the axis of a line contact. The shorter semi-axis lies along the smaller relative radius.
    """
    return _oriented(case, _hertz_full(case))


def hertz(case: ContactCase) -> dict:
    """Hertz solution on the bridge axes.

    Returns
    -------
    dict
        ``p0_Pa``, ``s_x_m``, ``s_y_m``, ``R_x_m``, ``R_y_m``, ``delta_m`` (total approach; ``nan`` for
        a line contact) and ``pressure`` (callable ``p(X, Y)`` [Pa] on bridge coordinates, for
        initial guesses and plots).
    """
    ref = _hertz_full(case)
    s_x, s_y = _oriented(case, ref)
    p0 = float(ref["max_pressure"])
    R_x, R_y = relative_radii(case)

    def pressure(X, Y):
        r2 = (X / s_x) ** 2 + (0.0 if math.isinf(s_y) else (Y / s_y) ** 2)
        return p0 * np.sqrt(np.clip(1.0 - r2, 0.0, None))

    return {"p0_Pa": p0, "s_x_m": s_x, "s_y_m": s_y, "R_x_m": R_x, "R_y_m": R_y,
            "delta_m": float(ref.get("total_deflection", math.nan)), "pressure": pressure}


def _hertz_full(case: ContactCase) -> dict:
    import slippy.contact as c
    return c.hertz_full(**case.hertz_args())


def _oriented(case: ContactCase, ref: dict) -> tuple[float, float]:
    rx, ry = relative_radii(case)
    radii = sorted(float(r) for r in ref["contact_radii"])
    if math.isinf(rx) or math.isinf(ry):                     # line contact: one infinite axis
        half = radii[0]
        return (math.inf, half) if math.isinf(rx) else (half, math.inf)
    small, large = radii[0], radii[-1]
    return (small, large) if rx <= ry else (large, small)
