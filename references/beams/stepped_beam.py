"""
references/beams/stepped_beam.py -- closed-form reference solution:
piecewise-constant section, two pin supports, point and polynomial
distributed loads.

Moved here on 2026-10-06 from the ``analytical_solution.py`` of the stepped
FEM case (now ``validation/shafts/fem_stepped_distributed_gear``):
nothing in it is specific to that case.

Exact Euler-Bernoulli and Timoshenko deflection, rotation, bending moment
and shear force for a shaft with piecewise-constant section properties,
two pin supports, and transverse point and distributed loads with
polynomial intensity. First used as the exact reference
for ``case_distributed_gear_parabolic``.

Method
------
The beam is statically determinate, so the reactions and the bending
moment follow from equilibrium alone, independently of the stiffness:

.. math::

    M(x) = \\sum_k F_k \\langle x - x_k \\rangle
         + \\sum_j \\int_{a_j}^{\\min(x, b_j)} q_j(s)\\,(x - s)\\,ds ,
    \\qquad V(x) = -\\frac{dM}{dx},

with the reactions included among the point forces F_k. The kinematics
are then integrated section by section,

.. math::

    \\frac{d\\theta}{dx} = \\frac{M}{EI(x)}, \\qquad
    \\frac{dv}{dx} = \\theta + \\frac{V}{\\kappa G A(x)} ,

with theta and v continuous across section changes and load points. The
second term is the Timoshenko shear strain gamma = V / kGA; it is omitted
for Euler-Bernoulli (theta = dv/dx). The two integration constants v(0)
and theta(0) follow from v(x_A) = v(x_B) = 0.

Between consecutive breakpoints (shaft ends, section changes, supports,
point-load positions and distributed-load limits) M(x) is a polynomial of
degree deg(q) + 2. On each such interval it is represented exactly by a
polynomial of degree ``_DEGREE``, and every integration is carried out
analytically on that polynomial; the result is exact to round-off for
load intensities up to degree ``_DEGREE - 2``.

Assumptions
-----------
* Linear elasticity, small displacements, solid circular sections.
* Section properties constant within each section (shoulder fillets do
  not modify EI or kGA).
* Ideal pin supports (v = 0, rotation free), consistent with the
  AxisForge rigid-support solver.
* The two bending planes (XY, XZ) are uncoupled.

Scope
-----
Independent from the FEM discretisation, NOT from the load resolution:
the loads are read from the same resolved ``ShaftSystem`` that the FEM
solves (``Load.component`` / ``DistributedRadialLoad.component_intensity``).

References
----------
.. [1] Timoshenko, S. P. & Gere, J. M., *Mechanics of Materials* --
       deflection of statically determinate beams by integration.
.. [2] Cowper, G. R. (1966). *J. Appl. Mech.* 33(2), 335-340.
.. [3] Hutchinson, J. R. (2001). *J. Appl. Mech.* 68(1), 87-92.

Units: mm, N, MPa, N.mm, rad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial import polynomial as P

from axisforge.core.loads import LoadPlane

_DEGREE = 8                     # polynomial degree per interval (exact for q up to degree 6)
_GAUSS = np.polynomial.legendre.leggauss(12)   # exact for polynomial integrands up to degree 23


# Shear correction factors: single definition in shear_coefficients.py,
# re-exported here so callers keep using ``an.kappa_cowper`` / ``an.KAPPA``.
from references.beams.shear_coefficients import (  # noqa: F401
    KAPPA,
    kappa_cowper,
    kappa_hutchinson,
)


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
    """Exact solution of one bending plane of a stepped two-pin beam.

    Use :func:`solve_plane` to build it. All callables accept scalars or
    arrays of positions x [mm].
    """

    def __init__(self, pieces: list[_Piece], R_A: float, R_B: float,
                 x_A: float, x_B: float):
        self._pieces = pieces
        self._edges = np.array([p.a for p in pieces] + [pieces[-1].b])
        self.R_A, self.R_B = R_A, R_B
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
        """Bending moment [N.mm] (right-continuous at point loads)."""
        return self._eval("M", x)

    def V(self, x) -> np.ndarray:
        """Shear force V = -dM/dx [N] (right-continuous at point loads)."""
        return self._eval("V", x)


def solve_plane(point_loads: Sequence[tuple[float, float]],
                distributed: Sequence[DistributedLoad],
                x_A: float, x_B: float,
                segments: Sequence, kappa: float | None,
                length: float) -> PlaneSolution:
    """Solve one bending plane of a stepped beam on two pin supports.

    Parameters
    ----------
    point_loads : sequence of (float, float)
        Applied point loads (position [mm], force [N]); reactions excluded.
    distributed : sequence of DistributedLoad
        Applied distributed loads in this plane.
    x_A, x_B : float
        Pin support positions [mm], x_A < x_B.
    segments : sequence
        Objects with attributes ``x_lo``, ``x_hi`` [mm], ``EI`` [N.mm^2]
        and ``GA`` [N], covering [0, length] left to right.
    kappa : float or None
        Shear correction factor; None for Euler-Bernoulli.
    length : float
        Shaft length [mm].

    Returns
    -------
    PlaneSolution

    Raises
    ------
    ValueError
        If x_A >= x_B.

    Notes
    -----
    Sign convention: forces and deflections share one positive direction;
    M = sum F_k <x - x_k> (+ distributed terms), V = -dM/dx.

    Examples
    --------
    Uniform shaft, L = 100 mm, pins at the ends, uniform q = 1 N/mm,
    EI = 1e9 N.mm^2: mid-span deflection 5 q L^4 / (384 EI).

    >>> class S: x_lo, x_hi, EI, GA = 0.0, 100.0, 1e9, 1e8
    >>> sol = solve_plane([], [DistributedLoad(0.0, 100.0, lambda x: 1.0)],
    ...                   0.0, 100.0, [S], None, 100.0)
    >>> round(float(sol.v(50.0)), 9), round(5 * 100.0**4 / (384 * 1e9), 9)
    (0.001302083, 0.001302083)
    """
    if x_A >= x_B:
        raise ValueError(f"expected x_A < x_B, got {x_A}, {x_B}")

    dist = [(d.x_lo, d.x_hi, _q_vec(d.q)) for d in distributed]

    # --- reactions from statics -------------------------------------------------
    sum_F = sum(F for _, F in point_loads)
    sum_MA = sum(F * (x - x_A) for x, F in point_loads)
    for lo, hi, qv in dist:
        sum_F += _gauss_integral(qv, lo, hi)
        sum_MA += _gauss_integral(lambda s, qv=qv: qv(s) * (s - x_A), lo, hi)
    R_B = -sum_MA / (x_B - x_A)
    R_A = -sum_F - R_B
    forces = [*point_loads, (x_A, R_A), (x_B, R_B)]

    def M_exact(x: float) -> float:
        m = sum(F * (x - xk) for xk, F in forces if x >= xk)
        for lo, hi, qv in dist:
            if x > lo:
                m += _gauss_integral(lambda s, qv=qv: qv(s) * (x - s), lo, min(x, hi))
        return m

    # --- breakpoints ------------------------------------------------------------------
    pts = {0.0, float(length), float(x_A), float(x_B)}
    pts |= {float(s.x_lo) for s in segments} | {float(s.x_hi) for s in segments}
    pts |= {float(x) for x, _ in point_loads}
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
        slope = th_c if kappa is None else P.polyadd(th_c, V_c / (kappa * seg.GA))
        v_c = P.polyadd([v0], P.polyint(slope) * h)
        pieces.append(_Piece(a, b, M_c, V_c, th_c, v_c))
        theta0, v0 = P.polyval(1.0, th_c), P.polyval(1.0, v_c)

    # --- impose v(x_A) = v(x_B) = 0 by adding the rigid-body field C2 + C1 x -----------
    tmp = PlaneSolution(pieces, R_A, R_B, x_A, x_B)
    vA, vB = float(tmp.v(x_A)), float(tmp.v(x_B))
    C1 = -(vB - vA) / (x_B - x_A)
    C2 = -vA - C1 * x_A
    for p in pieces:
        h = p.b - p.a
        p.theta = P.polyadd(p.theta, [C1])
        p.v = P.polyadd(p.v, [C2 + C1 * p.a, C1 * h])
    return PlaneSolution(pieces, R_A, R_B, x_A, x_B)


# =============================================================================
# Whole shaft
# =============================================================================

@dataclass
class ShaftSolution:
    """Closed-form solution of one shaft, both bending planes.

    Attributes
    ----------
    name : str
        Shaft name.
    xy, xz : PlaneSolution
        Solutions in the XY and XZ planes; they map onto
        ``ShaftResults.v_xy`` / ``v_xz`` etc.
    """

    name: str
    xy: PlaneSolution
    xz: PlaneSolution

    def v(self, x) -> np.ndarray:
        """Resultant deflection sqrt(v_xy^2 + v_xz^2) [mm]."""
        return np.hypot(self.xy.v(x), self.xz.v(x))

    def M(self, x) -> np.ndarray:
        """Resultant bending moment [N.mm]."""
        return np.hypot(self.xy.M(x), self.xz.M(x))


def solve_shaft(shaft_system, segments: Sequence, kappa: float | None,
                length: float) -> ShaftSolution:
    """Closed-form solution for one resolved AxisForge shaft.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Resolved shaft; its ``radial_loads`` and ``distributed_radial_loads``
        are read as they are given to the FEM solver.
    segments : sequence
        Section properties (see :func:`solve_plane`).
    kappa : float or None
        Shear correction factor; None for Euler-Bernoulli.
    length : float
        Shaft length [mm].

    Returns
    -------
    ShaftSolution
    """
    x_A, x_B = sorted(b.position for b in shaft_system.bearings)

    def plane(pl):
        pts = [(ld.position, float(ld.component(pl))) for ld in shaft_system.radial_loads]
        pts = [(x, F) for x, F in pts if F != 0.0]
        dist = [DistributedLoad(ld.x_lo, ld.x_hi,
                                lambda x, _ld=ld, _pl=pl: float(_ld.component_intensity(x, _pl)))
                for ld in shaft_system.distributed_radial_loads]
        return solve_plane(pts, dist, x_A, x_B, segments, kappa, length)

    return ShaftSolution(shaft_system.name, plane(LoadPlane.XY), plane(LoadPlane.XZ))
