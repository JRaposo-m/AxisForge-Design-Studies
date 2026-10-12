"""Surfaces of a contact (:class:`ContactCase` -> SlipPY surfaces), for static BEM and
lubrication alike: the :class:`Grid` around the contact and the surface :func:`pair` on it,
smooth or with roughness (generated in ``rough.py`` or measured).

Two builds of the same contact, chosen by the ``build`` argument of :func:`pair` (finding
``composite_vs_two_profiles_2026-10-10``: equal dry and in smooth EHL). The half-space solvers
only see the gap, and the two builds have the same gap to second order in x/r and y/r (the Hertz
paraboloid, Johnson 1985); they differ at the ends of a long conformal ellipse:

* ``"composite"`` - the relative (equivalent) body against a flat; what Hertz and
  Hamrock-Dowson assume. Default for phase 1 (lubrication, start of roughness).
* ``"two_profiles"`` - each body with its own exact profile (sphere, torus, cylinder, flat).
  For checking the paraboloid approximation and for roughness on both bodies.

Conventions fixed here, because SlipPY does not check them:

* **Rolling direction = array axis 0.** The Reynolds solver differences the gap along axis 0
  (``slippy/contact/unified_reynolds_solver.py:341``). :class:`Grid` holds ``X`` (rolling) varying
  along axis 0 and ``Y`` (across) along axis 1. Case radii are ``(r_x, r_y)`` with x = rolling.
* **Dyadic grid spacing** ``h = 2**k`` m, so SlipPY's ``np.arange(0, n*h, h)`` gives exactly ``n``
  points on rectangular grids (``slippy/contact/_model_utils.py:139-147``).
* **Both surfaces discrete** (a zero array for a flat), same shape.
* Each body keeps its own material (``Elastic`` ``E``, ``v`` from ``case.E``, ``case.nu``), in both builds.

Line contacts are periodic along the roller axis (axis 1): the static or lubrication step must
be created with ``periodic_geometry=True, periodic_axes=(False, True)`` and the total load
``case.load * Y_length``.

References
----------
Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press, ch. 4 (the gap of two
bodies as a paraboloid of the relative radii: the composite body).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from axisforge.results._base import DC

from .base import ContactCase
from .reference import relative_radii, semi_axes

__all__ = ["Grid", "pair"]

_ids = itertools.count()       # SlipPY material names must be unique per process


@dataclass(**DC)
class Grid:
    """Rectangular, uniform grid on which both surfaces of a contact are sampled.

    The contact centre (apex of the bodies) is the origin and lies on a node. SlipPY has one
    spacing for both directions, so ``h`` is the spacing along rolling and across it.

    A grid built by hand is checked for its axes and spacing (see Raises). One more thing
    :meth:`around` guarantees is assumed and not checked: the origin on node
    ``(n_x//2, n_y//2)`` (``outputs.measure`` counts ``apex_offset_nodes`` from it, and so do
    callers that cut through the centre node).

    Parameters
    ----------
    X, Y : ndarray, shape (n_x, n_y)
        Node coordinates [m]. ``X`` (rolling direction) varies along array axis 0, ``Y`` (across
        rolling; the roller axis of a line contact) along axis 1. float64; single-precision
        coordinates fail the step check.
    h : float
        Grid spacing [m]; a power of two when built by :meth:`around`. Another value is accepted
        here, but SlipPY may refuse it with a shape error: its ``np.arange(0, n*h, h)`` can then
        give ``n + 1`` points. Use :meth:`around`.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` are not 2-D with the same shape; if ``h`` is not positive; if ``X``
        does not vary only along axis 0 with step ``h``; if ``Y`` does not vary only along axis 1
        with step ``h``.
    """

    X: np.ndarray
    Y: np.ndarray
    h: float

    def __post_init__(self) -> None:
        if np.ndim(self.X) != 2 or np.shape(self.X) != np.shape(self.Y):
            raise ValueError(f"Grid: X and Y must be 2-D with the same shape, got {np.shape(self.X)} "
                             f"and {np.shape(self.Y)}")
        # a negative h would pass the step check on a reversed grid and reach SlipPY as a negative
        # grid_spacing with the rolling direction reversed
        if not self.h > 0:
            raise ValueError(f"Grid: h must be positive, got {self.h}")
        # SlipPY differences the gap along axis 0 as the rolling direction and has one spacing for
        # both axes; a swapped or differently spaced grid would be solved without any warning.
        for name, a, axis in (("X", self.X, 0), ("Y", self.Y, 1)):
            if np.any(np.diff(a, axis=1 - axis) != 0.0):
                raise ValueError(f"Grid: {name} must vary only along axis {axis} (rolling = axis 0)")
            # relative tolerance, not equality: a hand-built grid need not be dyadic
            if not np.allclose(np.diff(a, axis=axis), self.h, rtol=1e-9, atol=0.0):
                raise ValueError(f"Grid: the step of {name} along axis {axis} is not h = {self.h}")

    @classmethod
    def around(cls, case: ContactCase, h: float | None = None, span_x: float = 4.0, span_y: float = 1.3,
               n_y_line: int = 16) -> Grid:
        """Grid around the contact centre, sized in Hertz semi-axes of ``case``.

        Parameters
        ----------
        case : ContactCase
        h : float, optional
            Grid spacing [m]; rounded down to a power of two (never coarser). Default ``s_x / 10``.
        span_x : float
            Domain half-length along rolling, in units of ``s_x`` [-] (not SlipPY's ``span``, which
            is the size of the influence matrix in nodes). Not only EHL inlet room: SlipPY static BEM
            on a DGBB inner contact gave p0 -3.2 % at 2, -0.5 % at 4, +0.05 % at 8 (grid-independent),
            although the pressure is zero outside the contact. The cause is the periodic image
            loads of SlipPY's default frequency-domain kernel
            (``Elastic(use_frequency_domain=True)``), not a property of the elastic half-space.
            Check the domain in every study.
        span_y : float
            Domain half-width across rolling, in units of ``s_y`` [-] (point contacts).
        n_y_line : int
            Nodes across rolling for a line contact (periodic axis).

        Returns
        -------
        Grid
            ``X`` (rolling) varies along axis 0, ``Y`` along axis 1; centre on a node.
        """
        s_x, s_y = semi_axes(case)
        h = 2.0 ** math.floor(math.log2(s_x / 10.0 if h is None else h))   # never coarser than asked

        # Node count: the contact centre is the origin and sits on node n//2, because
        # x = (i - n//2) * h is zero there. With k = ceil(span * s / h) intervals per side:
        #   n = 2k + 2 (always even) -> nodes -(k+1) ... +k
        #   the +x side is the short one and still reaches k*h >= span*s; the -x side has one
        #   node more. With n = 2k the +x side would stop at (k-1)*h < span*s.
        # Example: k = 3 -> n = 8 -> nodes -4h ... +3h.
        n_x = 2 * math.ceil(span_x * s_x / h) + 2        # both sides at least span_x (centre on a node)
        n_y = n_y_line if case.line else 2 * math.ceil(span_y * s_y / h) + 2
        x = (np.arange(n_x) - n_x // 2) * h              # origin on node n_x//2
        y = (np.arange(n_y) - n_y // 2) * h              # origin on node n_y//2
        X, Y = np.meshgrid(x, y, indexing="ij")
        return cls(X=X, Y=Y, h=h)


def _profile(r: tuple[float, float], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """SlipPY profile (``-sag``, apex 0) of a body with signed principal radii ``r = (r_x, r_y)``.

    One exact formula for every body: a circle of signed radius ``r_y`` (across rolling) swept
    around an axis at signed distance ``r_x`` from the apex. It gives the sphere (``r_x = r_y``),
    the cylinder (``r_y = inf``), the raceway torus convex or concave in either direction, and a
    thrust raceway flat along rolling (``r_x = inf``). Near the apex
    ``sag ~ x^2/(2 r_x) + y^2/(2 r_y)``, the paraboloid of Hertz.
    """
    r_x, r_y = r
    if ((math.isfinite(r_y) and np.abs(Y).max() >= abs(r_y))
            or (math.isfinite(r_x) and np.abs(X).max() >= abs(r_x))):
        raise ValueError(f"the grid reaches beyond the body of radii {r}: reduce span_x/span_y")
    sag_y = np.zeros_like(Y) if math.isinf(r_y) else r_y - math.copysign(1.0, r_y) * np.sqrt(r_y**2 - Y**2)
    if math.isinf(r_x):
        return -sag_y
    rho = r_x - sag_y                                         # local radius in the rolling direction
    z = -(r_x - math.copysign(1.0, r_x) * np.sqrt(rho**2 - X**2))
    if np.isnan(z).any():
        raise ValueError(f"the grid reaches beyond the body of radii {r}: reduce span_x/span_y")
    return z


def _surfaces(case: ContactCase, z1: np.ndarray, z2: np.ndarray, h: float):
    import slippy.contact as c
    import slippy.surface as s
    i = next(_ids)
    s1 = s.Surface(profile=z1, grid_spacing=h)
    s2 = s.Surface(profile=z2, grid_spacing=h)
    s1.material = c.Elastic(f"{case.label}_body1_{i}", {"E": case.E[0], "v": case.nu[0]})
    s2.material = c.Elastic(f"{case.label}_body2_{i}", {"E": case.E[1], "v": case.nu[1]})
    return s1, s2


def pair(case: ContactCase, grid: Grid, build: str = "composite", rough=(None, None)):
    """The two SlipPY surfaces of a contact on ``grid``, smooth or rough.

    The two builds are described in the module text; the roughness is shared between the bodies
    according to the build:

    * ``"composite"``: the roughness of both bodies is summed on surface 1 (the equivalent body);
      surface 2 stays flat. Valid for static contact and for lubrication in pure rolling, where
      both roughnesses travel together (a moving-roughness, hence transient, problem).
    * ``"two_profiles"``: each body gets its own roughness. Needed once the surfaces slide or
      move at different speeds, because each roughness then travels with its own body.

    Parameters
    ----------
    case : ContactCase
        Body 1 = ``case.r1``, body 2 = ``case.r2``; principal axes aligned (``angle = 0``).
    grid : Grid
        From :meth:`Grid.around`.
    build : {"composite", "two_profiles"}
    rough : (z_rough_1, z_rough_2)
        Roughness heights of body 1 and body 2 on the nodes of ``grid`` [m], with the sign of the
        profiles (positive towards the other body). Each is ``None`` (smooth body), an ndarray
        (from ``rough.fractal``, or measured and resampled by the caller) or a SlipPY surface
        (its ``profile`` is read). Heights measured out of the material of each body (peaks > 0)
        already have this sign. The two surfaces face each other: a map measured on body 2 from
        outside must be mirrored by the caller so that its nodes fall on the same (x, y) as
        body 1; nothing is flipped here. SlipPY shifts the gap to first touch
        (``slippy/contact/_model_utils.py:174-177``), so with roughness the approach is counted
        from the highest asperity and a constant offset in a roughness array has no effect.

    Returns
    -------
    surface_1, surface_2 : slippy Surface
        Discrete, shape of ``grid.X``, spacing ``grid.h``; pass them to
        ``slippy.contact.ContactModel``.

    Raises
    ------
    ValueError
        For an unknown ``build``; ``case.angle != 0``; a roughness whose shape is not
        ``grid.X.shape``; a SlipPY roughness surface whose ``grid_spacing`` is not ``grid.h``; a
        grid that reaches beyond a body.

    Notes
    -----
    The roughness is added to the profile array before the surfaces are made. SlipPY's
    ``Surface + Surface`` would crop silently to the smaller shape and drop the material; here the
    shapes must match and each surface keeps the material of the case.
    """
    if build not in ("composite", "two_profiles"):
        raise ValueError(f"build must be 'composite' or 'two_profiles', got {build!r}")
    if case.angle != 0.0:
        raise ValueError("the builds assume aligned principal axes (angle = 0)")
    X, Y, h = grid.X, grid.Y, grid.h
    z_rough = []
    for name, z in zip(("rough[0]", "rough[1]"), rough):
        if z is None:
            z_rough.append(0.0)                            # smooth body
            continue
        if hasattr(z, "profile"):                          # SlipPY surface
            # same nodes only if the spacing is the same: no resampling. The comparison is exact
            # for a grid from Grid.around, whose h is dyadic
            if z.grid_spacing != h:
                raise ValueError(f"{name} has grid_spacing {z.grid_spacing}, the grid {h}")
            z = z.profile
        z = np.asarray(z, dtype=float)
        if z.shape != X.shape:
            raise ValueError(f"{name} has shape {z.shape}, the grid {X.shape}")
        z_rough.append(z)
    if build == "composite":
        # the heights of both bodies point into the gap, so they add on the equivalent body
        z1 = _profile(relative_radii(case), X, Y) + z_rough[0] + z_rough[1]
        z2 = np.zeros_like(X)
    else:
        z1, z2 = _profile(case.r1, X, Y) + z_rough[0], _profile(case.r2, X, Y) + z_rough[1]
    return _surfaces(case, z1, z2, h)
