"""The grid on which both bodies of a contact are sampled, and the SlipPY step arguments it implies.

Conventions fixed here, because SlipPY does not check them:

* **Grid x = array axis 0.** :class:`Grid` holds ``X`` (along grid x) varying along axis 0 and
  ``Y`` (along grid y) along axis 1. Grid x is at ``Grid.angle`` from bridge x (the rolling
  direction); ``angle = 0``, the default, puts the grid on the bridge axes. The SlipPY Reynolds
  solver differences the gap along axis 0 (``slippy/contact/unified_reynolds_solver.py``);
  SlipPY's internal y is array axis 0.
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

# [-] domain half-length along grid x (rolling when Grid.angle = 0), in half-extents of the
# contact. SlipPY's default frequency-domain kernel (Elastic(use_frequency_domain=True)) repeats the
# loads every 2 n h on each non-periodic axis (both axes for a point contact), so p0 depends on the
# domain even where the pressure is zero. The image error is not monotonic in the domain size
# (R-509); on the 6210 (psi = 0, grid on the bridge axes) the binding direction is across rolling,
# where span_y = 1.3 sits near a zero of the error (R-046). That tuning does not carry over to a
# turned grid (crossed cylinders at 45 deg: approach -0.98 % on the bridge grid, -1.60 % on the
# principal grid, C9 review).
SPAN_X_0 = 4.0


@dataclass(**DC)
class Grid:
    """Rectangular, uniform grid on which both surfaces of a contact are sampled.

    The grid may be turned in the contact plane: its x axis is at ``angle`` (gamma) from bridge x.
    A node of grid coordinates ``(X, Y)`` lies at the bridge coordinates
    ``x_b = X cos(gamma) - Y sin(gamma)``, ``y_b = X sin(gamma) + Y cos(gamma)``.
    ``gamma = contact.psi`` (``around(..., angle="principal")``) puts the axes of the contact
    ellipse on the grid lines. The contact keeps its quantities on the bridge axes
    (``Contact.psi``, ``HertzResult.psi``); only the node coordinates are in the grid frame. A line
    contact needs its line along grid y, the periodic axis: ``gamma = psi = 0`` (refused otherwise
    by :meth:`step_args`). The dry static solve has no in-plane direction, so a turned grid changes
    it only through discretisation and through where the domain and the images of the frequency
    kernel lie (R-509). Any later step or sub-model with a direction in array axes must refuse
    ``gamma != 0`` (or turn its vector): lubrication needs the entrainment along grid x (SlipPY's
    Reynolds solver differences along axis 0); SlipPY sub-models take ``'x'`` = array axis 1 (R-156:
    ``RigidBodyDisplacement``, ``ContactTime``, tangential load, slip, wear).

    The contact centre (apex of the bodies) is the origin and lies on a node. A grid built by hand
    is checked for its axes and spacing (see Raises). One more thing :meth:`around` guarantees is
    assumed and not checked: the origin on node ``(n_x//2, n_y//2)`` (measurements through the
    centre node count from it).

    Parameters
    ----------
    X, Y : ndarray, shape (n_x, n_y)
        Node coordinates along the grid axes [m]. ``X`` (grid x; the rolling direction when
        ``angle = 0``) varies along array axis 0, ``Y`` (grid y; the roller axis of a line
        contact) along axis 1. float64; single-precision coordinates fail the step check.
    h : float
        Grid spacing [m], a power of two.
    angle : float, optional
        Angle gamma of the grid x axis from the bridge x axis [rad], positive from x toward y, in
        (-pi/2, pi/2]. Default 0 (grid on the bridge axes).

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` are not 2-D with the same shape; if ``h`` is not a positive power of
        two; if ``X`` does not vary only along axis 0 with step ``h``; if ``Y`` does not vary only
        along axis 1 with step ``h``; if ``angle`` is outside (-pi/2, pi/2].
    """

    X: np.ndarray
    Y: np.ndarray
    h: float
    angle: float = 0.0

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
        # SlipPY differences the gap along axis 0 (Reynolds direction) and has one spacing for
        # both axes; a swapped or differently spaced grid would be solved without any warning.
        for name, a, axis in (("X", self.X, 0), ("Y", self.Y, 1)):
            if np.any(np.diff(a, axis=1 - axis) != 0.0):
                raise ValueError(f"Grid: {name} must vary only along axis {axis} (grid x = axis 0)")
            # relative tolerance, not equality: coordinates built as i*h or by cumulative sums
            if not np.allclose(np.diff(a, axis=axis), self.h, rtol=1e-9, atol=0.0):
                raise ValueError(f"Grid: the step of {name} along axis {axis} is not h = {self.h}")
        # the range of Contact.angle and Contact.psi: each orientation of the grid lines once
        if not -math.pi / 2 < self.angle <= math.pi / 2:
            raise ValueError(f"Grid: angle must be in (-pi/2, pi/2] rad, got {self.angle}")

    # OPEN SB-05: Grid.around takes the HertzResult as an argument (mesh cannot call solvers)
    @classmethod
    def around(cls, contact: Contact, hertz, *, h: float | None = None, span_x: float = SPAN_X_0,
               span_y: float = 1.3, n_y_line: int = 16, angle: float | str = 0.0) -> Grid:
        """Grid around the contact centre, sized on the Hertz contact ellipse.

        The domain covers the bounding box of the Hertz ellipse on the grid axes, half-extents
        ``(X_half, Y_half) = hertz.half_extents(angle)``, the ellipse turned by ``psi - angle``
        (``b`` along x and ``a`` along y when ``psi - angle = 0``: the bearing case at
        ``angle = 0``, any point contact at ``angle="principal"``). A line contact (its line along
        y) has ``X_half = b`` and ``n_y_line`` nodes along its periodic axis.

        Parameters
        ----------
        contact : Contact
            Read for ``line`` and ``psi``.
        hertz : object with ``b`` and ``half_extents(frame_angle)``
            The Hertz ellipse of ``contact`` (``solvers.hertz(contact)``, a ``HertzResult``, whose
            ``psi`` is ``contact.psi`` by construction): minor semi-axis ``b`` [m].
        h : float, optional
            Grid spacing [m]; rounded down to a power of two (never coarser). Default ``b / 10``.
        span_x : float
            Domain half-length along grid x, in units of ``X_half`` [-] (not SlipPY's ``span``,
            the size of the influence matrix in nodes). Default ``SPAN_X_0``; check the domain in
            every study.
        span_y : float
            Domain half-width along grid y, in units of ``Y_half`` [-] (point contacts).
        n_y_line : int
            Nodes along grid y for a line contact (periodic axis).
        angle : float or "principal"
            Grid angle gamma from bridge x [rad] (see :class:`Grid`); ``"principal"`` takes
            ``contact.psi``, the ellipse axes on the grid axes. Default 0 (bridge axes).

        Returns
        -------
        Grid
            Centre on node ``(n_x//2, n_y//2)``.

        Raises
        ------
        ValueError
            If ``b`` or a given ``h`` is not positive and finite; if a point contact has
            ``a = inf``; if ``angle`` is a string other than ``"principal"``. A line contact on a
            turned grid is refused by :meth:`step_args`.
        """
        if isinstance(angle, str):
            if angle != "principal":
                raise ValueError(f"Grid.around: angle must be a number [rad] or 'principal', got {angle!r}")
            angle = contact.psi
        b = hertz.b
        if not (b > 0 and math.isfinite(b)):
            raise ValueError(f"Grid.around: the minor semi-axis b must be positive and finite, got {b}")
        if h is not None and not (h > 0 and math.isfinite(h)):
            raise ValueError(f"Grid.around: h must be positive and finite, got {h}")
        X_half, Y_half = hertz.half_extents(angle)
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
        return cls(X=X, Y=Y, h=h, angle=angle)

    def spans(self, hertz) -> tuple[float, float]:
        """Shorter side of the domain from the contact centre, in half-extents of the Hertz ellipse
        on the grid axes (``hertz.half_extents(self.angle)``), ``(span_x, span_y)`` [-]: the ``span_x``,
        ``span_y`` that :meth:`around` was given (at least). ``span_y`` is nan for a line
        (``hertz.line``; y is periodic).
        """
        X_half, Y_half = hertz.half_extents(self.angle)
        x, y = self.X[:, 0], self.Y[0, :]
        span_x = min(-x.min(), x.max()) / X_half
        return span_x, math.nan if hertz.line else min(-y.min(), y.max()) / Y_half

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
            If ``step_kwargs`` sets the load (``normal_load``, ``interference``, ``mean_gap``) or an
            offset of the surfaces (``off_set_x``, ``off_set_y``: the apex stays on the centre node);
            for a line contact given ``periodic_geometry`` other than True or ``periodic_axes``
            other than ``(False, True)``: the load would still be that of one period of an
            infinite cylinder while the solve is not periodic; for a line contact on a turned grid
            (``angle != 0``): its line must lie along grid y, the periodic axis.
        """
        load_keys = sorted({"normal_load", "interference", "mean_gap"} & step_kwargs.keys())
        if load_keys:
            raise ValueError(f"step_args: {', '.join(load_keys)} not accepted; the load is contact.load")
        # the measurement of a solve assumes the apex on the grid centre node
        offsets = sorted({"off_set_x", "off_set_y"} & step_kwargs.keys())
        if offsets:
            raise ValueError(f"step_args: {', '.join(offsets)} not accepted; the apex stays on the centre node")
        if contact.line:
            # the strip is periodic along grid y only; Contact keeps a line along bridge y (psi = 0)
            if self.angle != 0.0:
                raise ValueError(f"a line contact needs its line along grid y (grid angle 0), got grid angle "
                                 f"{self.angle} rad")
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
