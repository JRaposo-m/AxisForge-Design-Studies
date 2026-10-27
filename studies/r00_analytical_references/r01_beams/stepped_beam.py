"""
Stepped beam on two pin supports or one fixed end: piecewise-constant section, point forces,
point couples and polynomial distributed loads.

Exact Euler-Bernoulli and Timoshenko deflection, rotation, bending moment
and shear force for a shaft with piecewise-constant section properties,
two pin supports or one fixed end, transverse point and distributed loads with
polynomial intensity and point couples. Judge of the stepped shaft cases (v002).

Method
------
The beam is statically determinate, so the reactions and the bending
moment follow from equilibrium alone, independently of the stiffness:

.. math::

    M(x) = \\sum_k F_k \\langle x - x_k \\rangle
         - \\sum_k C_k \\langle x - c_k \\rangle^0
         + \\sum_j \\int_{a_j}^{\\min(x, b_j)} q_j(s)\\,(x - s)\\,ds ,
    \\qquad V(x) = -\\frac{dM}{dx} \\text{ between breakpoints},

with the reactions included among the point forces F_k and, at a fixed end, the
couples C_k (positive from +x toward +v, ``uniform_beam``). A couple makes M jump
by -C_k and adds no shear force. The kinematics are then integrated section by section,

.. math::

    \\frac{d\\theta}{dx} = \\frac{M}{EI(x)}, \\qquad
    \\frac{dv}{dx} = \\theta + \\frac{V}{\\kappa G A(x)} ,

with theta and v continuous across section changes and load points. The
second term is the Timoshenko shear strain gamma = V / kGA; it is omitted
for Euler-Bernoulli (theta = dv/dx) (Onate 2013, Eqs (1.5), (2.1)-(2.6c)). The two integration constants v(0)
and theta(0) follow from v(x_A) = v(x_B) = 0 on two pins, or from v(x_A) = theta(x_A) = 0 at a fixed end.

Between consecutive breakpoints (shaft ends, section changes, supports,
point-load positions and distributed-load limits) M(x) is a polynomial of
degree deg(q) + 2. On each such interval it is represented exactly by a
polynomial of degree ``_DEGREE``, and every integration is carried out
analytically on that polynomial; the result is exact to round-off for
load intensities up to degree ``_DEGREE - 2``.

Assumptions
-----------
* Linear elasticity, small slopes, statically determinate.
* Section properties constant within each segment, abrupt steps (shoulder fillets do not modify
  EI or kGA); kappa constant per segment.
* Ideal pin supports (v = 0, rotation free) or one ideal fixed end (v = 0, theta = 0); one
  bending plane (circular sections: the two planes are uncoupled and solved by two calls).

Units: mm, N, MPa, N.mm, rad. Imports NumPy only (an r imports no AxisForge, D-0036).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial import polynomial as P

from .uniform_beam import _reactions

__all__ = ["DistributedLoad", "Segment", "PlaneSolution", "solve_plane"]

_DEGREE = 8                     # polynomial degree per interval (exact for q up to degree 6)
_GAUSS = np.polynomial.legendre.leggauss(12)   # exact for polynomial integrands up to degree 23


# =============================================================================
# Load description for one plane
# =============================================================================

@dataclass
class DistributedLoad:
    """Distributed transverse load in one plane.

    Attributes
    ----------
    x_lo, x_hi : float
        Extent [mm], x_lo < x_hi.
    q : callable
        Signed intensity q(x) [N/mm]; must be a polynomial on [x_lo, x_hi]
        of degree <= ``_DEGREE - 2`` for the solution to be exact.
    """

    x_lo: float
    x_hi: float
    q: Callable[[float], float]


@dataclass(frozen=True)
class Segment:
    """One constant-section piece of the beam.

    Attributes
    ----------
    x_lo, x_hi : float
        Extent [mm], x_lo < x_hi.
    EI : float
        Bending stiffness [N.mm^2] (``uniform_beam.section_stiffness``).
    kGA : float or None
        Corrected shear stiffness kappa G A [N] (``uniform_beam.section_stiffness``); None for
        Euler-Bernoulli.
    """

    x_lo: float
    x_hi: float
    EI: float
    kGA: float | None = None


def _gauss_integral(f: Callable[[np.ndarray], np.ndarray], a: float, b: float) -> float:
    """Gauss-Legendre integral of a vectorised function over [a, b]."""
    if b <= a:
        return 0.0
    xi, w = _GAUSS
    x = 0.5 * (a + b) + 0.5 * (b - a) * xi
    return float(0.5 * (b - a) * np.sum(w * f(x)))


def _q_vec(q: Callable[[float], float]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.array([q(float(s)) for s in np.atleast_1d(x)])


# =============================================================================
# One bending plane
# =============================================================================

@dataclass
class _Piece:
    """Polynomial representation of the solution on one interval [a, b].

    All polynomials are in the local variable u = (x - a) / h, h = b - a.
    """

    a: float
    b: float
    M: np.ndarray
    V: np.ndarray
    theta: np.ndarray
    v: np.ndarray


class PlaneSolution:
    """Exact solution of one bending plane of a stepped beam on two pins or one fixed end.

    Use :func:`solve_plane` to build it. All callables accept scalars or
    arrays of positions x [mm]. ``x_B`` and ``R_B`` are None at a fixed end at ``x_A``,
    whose reaction couple is ``C_A`` [N.mm] (0 on two pins).
    """

    def __init__(self, pieces: list[_Piece], R_A: float, R_B: float | None,
                 x_A: float, x_B: float | None, C_A: float = 0.0):
        self._pieces = pieces
        self._edges = np.array([p.a for p in pieces] + [pieces[-1].b])
        self.R_A, self.R_B, self.C_A = R_A, R_B, C_A
        self.x_A, self.x_B = x_A, x_B

    def _eval(self, field: str, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.clip(np.searchsorted(self._edges, x, side="right") - 1, 0, len(self._pieces) - 1)
        out = np.empty_like(x)
        for k in np.unique(idx):
            p = self._pieces[k]
            mask = idx == k
            out[mask] = P.polyval((x[mask] - p.a) / (p.b - p.a), getattr(p, field))
        return out

    def v(self, x) -> np.ndarray:
        """Transverse deflection [mm]."""
        return self._eval("v", x)

    def theta(self, x) -> np.ndarray:
        """Cross-section rotation [rad]."""
        return self._eval("theta", x)

    def M(self, x) -> np.ndarray:
        """Bending moment [N.mm] (right-continuous at point forces and couples)."""
        return self._eval("M", x)

    def V(self, x) -> np.ndarray:
        """Shear force V = -dM/dx between breakpoints [N] (right-continuous at point forces)."""
        return self._eval("V", x)


def solve_plane(point_loads: Sequence[tuple[float, float]],
                distributed: Sequence[DistributedLoad],
                x_A: float, x_B: float | None,
                segments: Sequence[Segment], length: float, *,
                couples: Sequence[tuple[float, float]] = ()) -> PlaneSolution:
    """Solve one bending plane of a stepped beam on two pin supports or one fixed end.

    Parameters
    ----------
    point_loads : sequence of (float, float)
        Applied point loads (position [mm], force [N]); reactions excluded.
    distributed : sequence of DistributedLoad
        Applied distributed loads in this plane.
    x_A : float
        First pin, or the fixed end when ``x_B`` is None [mm].
    x_B : float or None
        Second pin [mm], x_A < x_B; None for a fixed end at ``x_A`` (v = 0, theta = 0), at an end or
        inside the beam (each side is then a cantilever).
    segments : sequence of Segment
        Section pieces covering [0, length] left to right; a segment with
        ``kGA=None`` has no shear term (Euler-Bernoulli).
    length : float
        Shaft length [mm].
    couples : sequence of (float, float), optional
        Applied point couples (position [mm], couple [N.mm]), positive from +x toward +v.

    Returns
    -------
    PlaneSolution

    Raises
    ------
    ValueError
        If x_A >= x_B (two pins only; not checked when x_B is None).

    Notes
    -----
    Sign convention: forces and deflections share one positive direction; a couple is positive
    from +x toward +v; M = sum F_k <x - x_k> - sum C_k <x - c_k>^0 (+ distributed terms),
    V = -dM/dx between breakpoints. Statically determinate only: two pins or one fixed end.

    Examples
    --------
    Uniform shaft, L = 100 mm, pins at the ends, uniform q = 1 N/mm,
    EI = 1e9 N.mm^2: mid-span deflection 5 q L^4 / (384 EI).

    >>> sol = solve_plane([], [DistributedLoad(0.0, 100.0, lambda x: 1.0)],
    ...                   0.0, 100.0, [Segment(0.0, 100.0, 1e9)], 100.0)
    >>> round(float(sol.v(50.0)), 9), round(5 * 100.0**4 / (384 * 1e9), 9)
    (0.001302083, 0.001302083)
    """
    dist = [(d.x_lo, d.x_hi, _q_vec(d.q)) for d in distributed]

    # --- reactions from statics -------------------------------------------------
    sum_F = sum(F for _, F in point_loads)
    sum_MA = sum(F * (x - x_A) for x, F in point_loads) + sum(C for _, C in couples)
    for lo, hi, qv in dist:
        sum_F += _gauss_integral(qv, lo, hi)
        sum_MA += _gauss_integral(lambda s, qv=qv: qv(s) * (s - x_A), lo, hi)
    R_A, R_B, C_A = _reactions(sum_F, sum_MA, x_A, x_B)
    if x_B is None:
        forces, all_couples = [*point_loads, (x_A, R_A)], [*couples, (x_A, C_A)]
    else:
        forces, all_couples = [*point_loads, (x_A, R_A), (x_B, R_B)], list(couples)

    def M_exact(x: float) -> float:
        m = sum(F * (x - xk) for xk, F in forces if x >= xk) - sum(C for ck, C in all_couples if x >= ck)
        for lo, hi, qv in dist:
            if x > lo:
                m += _gauss_integral(lambda s, qv=qv: qv(s) * (x - s), lo, min(x, hi))
        return m

    # --- breakpoints ------------------------------------------------------------------
    pts = {0.0, float(length), float(x_A)} | ({float(x_B)} if x_B is not None else set())
    pts |= {float(s.x_lo) for s in segments} | {float(s.x_hi) for s in segments}
    pts |= {float(x) for x, _ in point_loads} | {float(c) for c, _ in couples}
    for lo, hi, _ in dist:
        pts |= {float(lo), float(hi)}
    edges = np.array(sorted(p for p in pts if 0.0 <= p <= length))

    def segment_at(xm: float):
        for s in segments:
            if s.x_lo <= xm <= s.x_hi:
                return s
        raise ValueError(f"no section covers x = {xm}")

    # --- piecewise exact integration with v(0) = theta(0) = 0 ------------------------
    u_nodes = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, _DEGREE + 1)))   # Chebyshev-Lobatto on [0, 1]
    pieces: list[_Piece] = []
    theta0 = v0 = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        h = b - a
        seg = segment_at(0.5 * (a + b))
        # M on [a, b): sample just inside the interval at the end points (right-continuity at a)
        xs = a + h * u_nodes
        eps = 1e-12 * max(1.0, h)
        xs_eval = np.clip(xs, a + eps, b - eps)
        M_c = P.polyfit(u_nodes, [M_exact(x) for x in xs_eval], _DEGREE)
        V_c = -P.polyder(M_c) / h
        th_c = P.polyadd([theta0], P.polyint(M_c / seg.EI) * h)
        slope = th_c if seg.kGA is None else P.polyadd(th_c, V_c / seg.kGA)
        v_c = P.polyadd([v0], P.polyint(slope) * h)
        pieces.append(_Piece(a, b, M_c, V_c, th_c, v_c))
        theta0, v0 = P.polyval(1.0, th_c), P.polyval(1.0, v_c)

    # --- impose the supports by adding the rigid-body field C2 + C1 x ---------------------
    tmp = PlaneSolution(pieces, R_A, R_B, x_A, x_B, C_A)
    vA = float(tmp.v(x_A))
    if x_B is None:                                     # fixed end: v(x_A) = theta(x_A) = 0
        C1 = -float(tmp.theta(x_A))
    else:                                               # two pins: v(x_A) = v(x_B) = 0
        C1 = -(float(tmp.v(x_B)) - vA) / (x_B - x_A)
    C2 = -vA - C1 * x_A
    for p in pieces:
        h = p.b - p.a
        p.theta = P.polyadd(p.theta, [C1])
        p.v = P.polyadd(p.v, [C2 + C1 * p.a, C1 * h])
    return PlaneSolution(pieces, R_A, R_B, x_A, x_B, C_A)
