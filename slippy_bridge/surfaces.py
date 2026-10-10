""":class:`ContactCase` -> SlipPY surfaces, for static BEM and for lubrication alike.

Two builds of the same contact (finding ``composite_vs_two_profiles_2026-10-10``: equal dry and in
smooth EHL, because the half-space solvers only see the gap ``h1 + h2``):

* :func:`composite` - the relative (equivalent) body against a flat; what Hertz and
  Hamrock-Dowson assume. Default for phase 1 (lubrication, start of roughness).
* :func:`two_profiles` - each body with its own exact profile (sphere, torus, cylinder, flat).
  For checking the paraboloid approximation and for roughness on both bodies.

Conventions fixed here, because SlipPY does not check them:

* **Rolling direction = array axis 0.** The Reynolds solver differences the gap along axis 0
  (``slippy/contact/unified_reynolds_solver.py:341``). :func:`grid` returns ``X`` (rolling) varying
  along axis 0 and ``Y`` (across) along axis 1. Case radii are ``(r_x, r_y)`` with x = rolling.
* **Dyadic grid spacing** ``h = 2**k`` m, so SlipPY's ``np.arange(0, n*h, h)`` gives exactly ``n``
  points on rectangular grids (``slippy/contact/_model_utils.py:139-147``).
* **Both surfaces discrete** (a zero array for a flat), same shape.
* Each body keeps its own material (``Elastic`` E, v from the case), in both builds.

Line contacts are periodic along the roller axis (axis 1): the static or lubrication step must
be created with ``periodic_geometry=True, periodic_axes=(False, True)`` and the total load
``case.load * Y_length``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .case import ContactCase

__all__ = ["semi_axes", "grid", "composite", "two_profiles"]

_ids = itertools.count()       # SlipPY material names must be unique per process


def semi_axes(case: ContactCase) -> tuple[float, float]:
    """Hertz contact semi-axes ``(s_x, s_y)`` [m] along the rolling direction and across it.

    ``inf`` on the axis of a line contact. The values come from ``slippy.contact.hertz_full``; the
    order of its ``contact_radii`` is not tied to x/y, so the axes are assigned from the relative
    radii of the case (the shorter semi-axis lies along the smaller relative radius).
    """
    import slippy.contact as c
    rx, ry = _relative(case.r1, case.r2)
    radii = sorted(float(r) for r in c.hertz_full(**case.hertz_args())["contact_radii"])
    if math.isinf(rx) or math.isinf(ry):                     # line contact: one infinite axis
        half = radii[0]
        return (math.inf, half) if math.isinf(rx) else (half, math.inf)
    small, large = radii[0], radii[-1]
    return (small, large) if rx <= ry else (large, small)


def grid(case: ContactCase, h: float | None = None, half_x: float = 4.0, half_y: float = 1.3,
         n_y_line: int = 16) -> tuple[np.ndarray, np.ndarray, float]:
    """Rectangular grid around the contact centre.

    Parameters
    ----------
    case : ContactCase
    h : float, optional
        Grid spacing [m]; rounded to the nearest power of two. Default ``s_x / 10``.
    half_x : float
        Half-length along rolling, in units of ``s_x``. Not only EHL inlet room: SlipPY static BEM
        on a DGBB inner contact gave p0 -3.2 % at 2, -0.5 % at 4, +0.05 % at 8 (grid-independent),
        although the pressure is zero outside the contact. Check the domain in every study.
    half_y : float
        Half-width across rolling, in units of ``s_y`` (point contacts).
    n_y_line : int
        Nodes across rolling for a line contact (periodic axis).

    Returns
    -------
    X, Y : ndarray
        Coordinates [m]; ``X`` (rolling) varies along axis 0, ``Y`` along axis 1. Centre on a node.
    h : float
        The grid spacing used [m].
    """
    s_x, s_y = semi_axes(case)
    h = 2.0 ** round(math.log2(s_x / 10.0 if h is None else h))
    n_x = 2 * round(half_x * s_x / h)
    n_y = n_y_line if case.line else 2 * round(half_y * s_y / h)
    x = (np.arange(n_x) - n_x // 2) * h
    y = (np.arange(n_y) - n_y // 2) * h
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, h


def _profile(r: tuple[float, float], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """SlipPY profile (``-sag``, apex 0) of a body with signed principal radii ``r = (r_x, r_y)``.

    One exact formula for every body: a circle of signed radius ``r_y`` (across rolling) swept
    around an axis at signed distance ``r_x`` from the apex. It gives the sphere (``r_x = r_y``),
    the cylinder (``r_y = inf``), the raceway torus convex or concave in either direction, and a
    thrust raceway flat along rolling (``r_x = inf``). Near the apex
    ``sag ~ x^2/(2 r_x) + y^2/(2 r_y)``, the paraboloid of Hertz. Generalises
    ``hertz_vs_bem/two_profiles/run.py::body_surface``.
    """
    rx, ry = r
    if (math.isfinite(ry) and np.abs(Y).max() >= abs(ry)) or (math.isfinite(rx) and np.abs(X).max() >= abs(rx)):
        raise ValueError(f"the grid reaches beyond the body of radii {r}: reduce half_x/half_y")
    s_y = np.zeros_like(Y) if math.isinf(ry) else ry - math.copysign(1.0, ry) * np.sqrt(ry**2 - Y**2)
    if math.isinf(rx):
        return -s_y
    rho = rx - s_y                                            # local radius in the rolling direction
    z = -(rx - math.copysign(1.0, rx) * np.sqrt(rho**2 - X**2))
    if np.isnan(z).any():
        raise ValueError(f"the grid reaches beyond the body of radii {r}: reduce half_x/half_y")
    return z


def _relative(r1: tuple[float, float], r2: tuple[float, float]) -> tuple[float, float]:
    """Relative radii ``1/R = 1/r1 + 1/r2`` per direction (``inf`` = flat)."""
    out = []
    for a, b in zip(r1, r2):
        k = (0.0 if math.isinf(a) else 1.0 / a) + (0.0 if math.isinf(b) else 1.0 / b)
        if k < 0:
            raise ValueError(f"negative relative curvature {k}: the bodies do not conform to a Hertz contact")
        out.append(math.inf if k == 0 else 1.0 / k)
    return tuple(out)


def _surfaces(case: ContactCase, z1: np.ndarray, z2: np.ndarray, h: float):
    import slippy.contact as c
    import slippy.surface as s
    i = next(_ids)
    s1 = s.Surface(profile=z1, grid_spacing=h)
    s2 = s.Surface(profile=z2, grid_spacing=h)
    s1.material = c.Elastic(f"{case.label}_body1_{i}", {"E": case.moduli[0], "v": case.v[0]})
    s2.material = c.Elastic(f"{case.label}_body2_{i}", {"E": case.moduli[1], "v": case.v[1]})
    return s1, s2


def composite(case: ContactCase, X: np.ndarray, Y: np.ndarray, h: float):
    """Equivalent body (relative radii) on body 1, a flat on body 2; each keeps its material.

    Returns
    -------
    surface_1, surface_2 : slippy Surface
        Discrete, same shape as ``X``; pass them to ``slippy.contact.ContactModel``.
    """
    if case.angle != 0.0:
        raise ValueError("the builds assume aligned principal axes (angle = 0)")
    return _surfaces(case, _profile(_relative(case.r1, case.r2), X, Y), np.zeros_like(X), h)


def two_profiles(case: ContactCase, X: np.ndarray, Y: np.ndarray, h: float):
    """Each body with its own exact profile; each keeps its material.

    Returns
    -------
    surface_1, surface_2 : slippy Surface
        Discrete, same shape as ``X``.
    """
    if case.angle != 0.0:
        raise ValueError("the builds assume aligned principal axes (angle = 0)")
    return _surfaces(case, _profile(case.r1, X, Y), _profile(case.r2, X, Y), h)
