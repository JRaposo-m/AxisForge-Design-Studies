"""The grid on which both bodies of a contact are sampled, and the SlipPY step arguments it implies.

Conventions fixed here, because SlipPY does not check them:

* **Rolling direction = array axis 0.** :class:`Grid` holds ``X`` (rolling, bridge x) varying
  along axis 0 and ``Y`` (across, bridge y) along axis 1. The SlipPY Reynolds solver differences
  the gap along axis 0 (``slippy/contact/unified_reynolds_solver.py``); SlipPY's internal y is
  array axis 0 = bridge x.
* **One spacing, dyadic.** SlipPY has one grid spacing for both axes; ``h = 2**k`` m so that its
  ``np.arange(0, n*h, h)`` gives exactly ``n`` points (``slippy/contact/_model_utils.py``, R-049).
* **The one place of the load given to the step** (:meth:`Grid.step_args`). A line contact is
  periodic along y (the roller axis): one period of an infinitely long strip, whose load per
  unit length [N/m] becomes the load on the period [N] here.

Imports nothing from SlipPY; the Hertz ellipse that sizes the grid is given by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.contact import DC

if TYPE_CHECKING:
    from ..core.contact import Contact

__all__ = ["Grid", "SPAN_X_0"]

# [-] domain half-length along rolling, in half-extents of the contact. SlipPY's default
# frequency-domain kernel (Elastic(use_frequency_domain=True)) repeats the loads every 2 n h on
# each non-periodic axis (both axes for a point contact), so p0 depends on the domain even where
# the pressure is zero. The image error is not monotonic in the domain size (R-509); on the 6210
# the binding direction is across rolling, where span_y = 1.3 sits near a zero of the error (R-046).
SPAN_X_0 = 4.0


@dataclass(**DC)
class Grid:
    """Rectangular, uniform grid on which both surfaces of a contact are sampled.

    The contact centre (apex of the bodies) is the origin and lies on a node. A grid built by hand
    is checked for its axes and spacing (see Raises). One more thing :meth:`around` guarantees is
    assumed and not checked: the origin on node ``(n_x//2, n_y//2)`` (measurements through the
    centre node count from it).

    Parameters
    ----------
    X, Y : ndarray, shape (n_x, n_y)
        Node coordinates [m]. ``X`` (rolling direction) varies along array axis 0, ``Y`` (across
        rolling; the roller axis of a line contact) along axis 1. float64; single-precision
        coordinates fail the step check.
    h : float
        Grid spacing [m], a power of two.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` are not 2-D with the same shape; if ``h`` is not a positive power of
        two; if ``X`` does not vary only along axis 0 with step ``h``; if ``Y`` does not vary only
        along axis 1 with step ``h``.
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
        if not (self.h > 0 and math.isfinite(self.h)):
            raise ValueError(f"Grid: h must be positive and finite, got {self.h}")
        # SlipPY's np.arange(0, n*h, h) can give n + 1 points for another h: a shape error (R-049)
        # OPEN SB-17: hand-built grids must have a dyadic h
        if not math.log2(self.h).is_integer():
            raise ValueError(f"Grid: h must be a power of two [m] (SlipPY node count), got {self.h}")
        # SlipPY differences the gap along axis 0 as the rolling direction and has one spacing for
        # both axes; a swapped or differently spaced grid would be solved without any warning.
        for name, a, axis in (("X", self.X, 0), ("Y", self.Y, 1)):
            if np.any(np.diff(a, axis=1 - axis) != 0.0):
                raise ValueError(f"Grid: {name} must vary only along axis {axis} (rolling = axis 0)")
            # relative tolerance, not equality: coordinates built as i*h or by cumulative sums
            if not np.allclose(np.diff(a, axis=axis), self.h, rtol=1e-9, atol=0.0):
                raise ValueError(f"Grid: the step of {name} along axis {axis} is not h = {self.h}")

    # OPEN SB-05: Grid.around takes the HertzResult as an argument (mesh cannot call solvers)
    @classmethod
    def around(cls, contact: Contact, hertz, *, h: float | None = None, span_x: float = SPAN_X_0,
               span_y: float = 1.3, n_y_line: int = 16) -> Grid:
        """Grid around the contact centre, sized on the Hertz contact ellipse.

        The domain covers the bounding box of the Hertz ellipse on the bridge axes, half-extents
        ``(X_half, Y_half) = hertz.half_extents()`` (``b`` along x and ``a`` along y when
        ``psi = 0``, the bearing case). A line contact (its line along y) has ``X_half = b`` and
        ``n_y_line`` nodes along its periodic axis.

        Parameters
        ----------
        contact : Contact
            Read for ``line``.
        hertz : object with ``b`` and ``half_extents()``
            The Hertz ellipse of ``contact`` (``solvers.hertz(contact)``, a ``HertzResult``, whose
            ``psi`` is ``contact.psi`` by construction): minor semi-axis ``b`` [m].
        h : float, optional
            Grid spacing [m]; rounded down to a power of two (never coarser). Default ``b / 10``.
        span_x : float
            Domain half-length along rolling, in units of ``X_half`` [-] (not SlipPY's ``span``,
            the size of the influence matrix in nodes). Default ``SPAN_X_0``; check the domain in
            every study.
        span_y : float
            Domain half-width across rolling, in units of ``Y_half`` [-] (point contacts).
        n_y_line : int
            Nodes across rolling for a line contact (periodic axis).

        Returns
        -------
        Grid
            Centre on node ``(n_x//2, n_y//2)``.

        Raises
        ------
        ValueError
            If ``b`` or a given ``h`` is not positive and finite, or a point contact has ``a = inf``.
        """
        b = hertz.b
        if not (b > 0 and math.isfinite(b)):
            raise ValueError(f"Grid.around: the minor semi-axis b must be positive and finite, got {b}")
        if h is not None and not (h > 0 and math.isfinite(h)):
            raise ValueError(f"Grid.around: h must be positive and finite, got {h}")
        X_half, Y_half = hertz.half_extents()
        if not contact.line and not math.isfinite(Y_half):
            raise ValueError(f"Grid.around: a point contact needs a finite major semi-axis a, got {hertz.a}")
        h = 2.0 ** math.floor(math.log2(b / 10.0 if h is None else h))   # dyadic, never coarser than asked

        # Node count: the contact centre is the origin and sits on node n//2, because
        # x = (i - n//2) * h is zero there. With k = ceil(span * s / h) intervals per side:
        #   n = 2k + 2 (always even) -> nodes -(k+1) ... +k
        #   the +x side is the short one and still reaches k*h >= span*s; the -x side has one
        #   node more. With n = 2k the +x side would stop at (k-1)*h < span*s.
        # Example: k = 3 -> n = 8 -> nodes -4h ... +3h.
        n_x = 2 * math.ceil(span_x * X_half / h) + 2
        n_y = n_y_line if contact.line else 2 * math.ceil(span_y * Y_half / h) + 2
        x = (np.arange(n_x) - n_x // 2) * h              # origin on node n_x//2
        y = (np.arange(n_y) - n_y // 2) * h              # origin on node n_y//2
        X, Y = np.meshgrid(x, y, indexing="ij")
        return cls(X=X, Y=Y, h=h)

    @property
    def length_y_m(self) -> float:
        """Length of the domain across rolling, ``n_y h`` [m]: the period of a line contact."""
        return self.Y.shape[1] * self.h

    # OPEN SB-06: no normal_load override; the load given to the step is always contact.load (x period for a line)
    def step_args(self, contact: Contact, **step_kwargs) -> dict:
        """Keyword arguments of the SlipPY step (``StaticStep``) for ``contact`` on this grid.

        A point contact gets ``normal_load = contact.load`` [N]; periodic flags given for it are
        the caller's choice and keep the full load. A line contact is an infinitely long cylinder,
        modelled by SlipPY as one period of a strip periodic along the roller axis (array axis 1):
        it gets ``periodic_geometry=True``, ``periodic_axes=(False, True)`` and the load on one
        period, ``contact.load`` [N/m] times the period :attr:`length_y_m` [m].

        Parameters
        ----------
        contact : Contact
            ``contact.load`` [N] (point) or [N/m] (line).
        **step_kwargs
            Other keywords of the step, passed through.

        Returns
        -------
        dict
            ``normal_load`` [N], the periodic flags of a line contact and ``step_kwargs``.

        Raises
        ------
        ValueError
            If ``step_kwargs`` sets the load (``normal_load``, ``interference``, ``mean_gap``);
            for a line contact given ``periodic_geometry`` other than True or ``periodic_axes``
            other than ``(False, True)``: the load would still be that of one period of an
            infinite cylinder while the solve is not periodic.
        """
        load_keys = sorted({"normal_load", "interference", "mean_gap"} & step_kwargs.keys())
        if load_keys:
            raise ValueError(f"step_args: {', '.join(load_keys)} not accepted; the load is contact.load")
        if contact.line:
            geometry = step_kwargs.setdefault("periodic_geometry", True)
            if geometry is not True:
                raise ValueError(f"a line contact needs periodic_geometry=True (periodic along the roller axis), "
                                 f"got {geometry!r}")
            axes = step_kwargs.setdefault("periodic_axes", (False, True))
            if not (isinstance(axes, (tuple, list)) and tuple(axes) == (False, True)):
                raise ValueError(f"a line contact needs periodic_axes=(False, True) (periodic along the roller "
                                 f"axis), got {axes!r}")
            load = contact.load * self.length_y_m                  # [N/m] * [m] = [N]
        else:
            load = contact.load
        return {"normal_load": load, **step_kwargs}
