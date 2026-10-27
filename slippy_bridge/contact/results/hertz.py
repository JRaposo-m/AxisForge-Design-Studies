"""Hertz solution of one contact, on the bridge axes: data only (filled by ``solvers.hertz``).

References
----------
Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press. Section 4.2(b),
p. 95 (elliptical pressure, the unnumbered line after Eq. (3.58)), Eqs. (4.27)-(4.28), p. 95; section 4.2(c), Eqs. (4.43)-(4.45), p. 101 (line
contact; the approach is not defined locally, text after Eq. (4.37), p. 100); section 4.2, list
(i)-(iv), p. 91-92 (assumptions).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.contact import DC

__all__ = ["HertzResult"]


@dataclass(**DC)
class HertzResult:
    """Hertz pressure and contact ellipse of one contact, oriented on the bridge axes.

    The contact ellipse has its minor semi-axis ``b`` along ``psi`` (the direction of the larger
    relative curvature ``kappa_max``) and its major semi-axis ``a`` across it (Johnson 1985 p. 98:
    ``a > b``, ``a`` along the smaller curvature). A line contact along bridge y has ``a = inf``
    and ``b`` its half-width (Johnson calls the half-width ``a``, section 4.2(c)).

    Filled by ``solvers.hertz`` from SlipPY ``hertz_full``: ``p0`` = ``max_pressure``, ``a``, ``b``
    = ``contact_radii`` sorted (SlipPY gives ``[major, minor]`` for a point, ``[half-width, inf]``
    for a line), ``delta`` = ``total_deflection`` (absent for a line), ``E_star`` =
    ``e_star`` (Johnson's E*, not E').

    Parameters
    ----------
    p0 : float
        Maximum (central) pressure [Pa].
    a : float
        Major semi-axis, across ``psi`` [m]; ``inf`` for a line.
    b : float
        Minor semi-axis, along ``psi`` [m]; the half-width of a line.
    psi : float
        Direction of the minor semi-axis from bridge x [rad], positive toward y (``Contact.psi``).
    delta : float
        Total approach of distant points of the two bodies [m]; ``nan`` for a line, where it
        depends on the size of the bodies and is not defined by the local solution.
    b_over_a : float
        Ellipse aspect ratio ``b/a`` [-]; 0 for a line.
    E_star : float
        Contact modulus [Pa].
    size_ratio : float
        Size of the contact against the relative radii, ``max(b kappa_max, a kappa_min)`` [-]
        (``a kappa_min = 0`` for a line): Hertz assumes it small, a << R (Johnson 4.2 (i)-(ii)).
        Only this relative-radius criterion; the per-body condition a << R1, R2 (iii) is not
        checked (OPEN SB-07; e.g. 6210 a/|r_groove| = 0.23).
    valid : bool
        Hertz validity: ``size_ratio`` within the limit the solver applies and no end of a body
        within the half-width (``ends_over_b`` all >= 1).
    line : bool
        True for a line contact (load per unit length).
    reason : str, optional
        Why ``valid`` is False ("" when valid).
    ends_over_b : dict, optional
        Distances to the ends of the bodies in half-widths ``b`` [-], by name
        (``d_end_<...>_over_b``, from ``Contact.end_distances``); empty for bodies without ends.
    """

    p0: float
    a: float
    b: float
    psi: float
    delta: float
    b_over_a: float
    E_star: float
    size_ratio: float
    valid: bool
    line: bool
    reason: str = ""
    ends_over_b: dict = field(default_factory=dict)

    def pressure(self, X: np.ndarray, Y: np.ndarray, frame_angle: float = 0.0) -> np.ndarray:
        """Hertz pressure on the coordinates of a frame at ``frame_angle`` from bridge x [Pa], zero
        outside the contact.

        ``p0 sqrt(1 - (xi/b)^2 - (eta/a)^2)`` with ``xi`` along ``psi`` and ``eta`` across it
        (Johnson 1985 p. 95, the unnumbered line from Eq. (3.58)); a line has ``a = inf``, so the strip
        ``p0 sqrt(1 - (x/b)^2)`` (Eq. (4.44), p. 101). In the frame the ellipse is turned by
        ``psi - frame_angle``.

        Parameters
        ----------
        X, Y : ndarray
            Coordinates [m] along the frame axes, same shape: bridge coordinates (x rolling,
            y across) for ``frame_angle = 0``, grid coordinates for ``frame_angle = grid.angle``.
        frame_angle : float, optional
            Angle of the frame x axis from bridge x [rad], positive toward y. Default 0.

        Returns
        -------
        ndarray
            Pressure [Pa], same shape as ``X``.
        """
        t = self.psi - frame_angle
        c, s = math.cos(t), math.sin(t)
        xi = X * c + Y * s                                   # along psi: minor semi-axis b
        eta = -X * s + Y * c                                 # across psi: major semi-axis a (inf: 0)
        return self.p0 * np.sqrt(np.clip(1.0 - (xi / self.b) ** 2 - (eta / self.a) ** 2, 0.0, None))

    def half_extents(self, frame_angle: float = 0.0) -> tuple[float, float]:
        """Half-extents of the contact ellipse along the axes of a frame at ``frame_angle`` from
        bridge x, ``(X_half, Y_half)`` [m] (bridge axes for 0, grid axes for ``grid.angle``).

        The bounding box of the ellipse turned by ``t = psi - frame_angle``: ``X_half = hypot(b cos t,
        a sin t)``, ``Y_half = hypot(b sin t, a cos t)`` (``b``, ``a`` when ``t = 0``). A line along
        y: ``(b, inf)`` (the formula would give ``inf * 0``); a line exists only with ``t = 0``
        (``Grid`` refuses another angle).
        """
        if self.line:
            return self.b, math.inf
        c, s = math.cos(self.psi - frame_angle), math.sin(self.psi - frame_angle)
        return math.hypot(self.b * c, self.a * s), math.hypot(self.b * s, self.a * c)

    # OPEN SB-10: record keys carry the unit (p0_Pa, a_m), attributes do not
    def record(self, tag: str = "") -> dict:
        """Flat record of the scalars, the unit in the key and ``tag`` before the unit:
        ``p0{tag}_Pa``, ``a{tag}_m``, ``b{tag}_m``, ``psi{tag}_rad``, ``delta{tag}_m``,
        ``b_over_a{tag}``, ``E_star{tag}_Pa``, ``size_ratio{tag}``, ``valid{tag}``, ``line{tag}``,
        ``reason{tag}`` and each ``d_end_<...>_over_b{tag}`` (``tag="_ref"`` gives ``p0_ref_Pa``, the
        name a study reads)."""
        return {f"p0{tag}_Pa": self.p0, f"a{tag}_m": self.a, f"b{tag}_m": self.b, f"psi{tag}_rad": self.psi,
                f"delta{tag}_m": self.delta, f"b_over_a{tag}": self.b_over_a, f"E_star{tag}_Pa": self.E_star,
                f"size_ratio{tag}": self.size_ratio, f"valid{tag}": self.valid, f"line{tag}": self.line,
                f"reason{tag}": self.reason, **{f"{key}{tag}": value for key, value in self.ends_over_b.items()}}
