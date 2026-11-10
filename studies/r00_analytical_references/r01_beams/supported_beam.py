"""
Statically indeterminate beam: any number of supports, each rigid or a linear radial spring, with an
optional rotational spring and a prescribed settlement and rotation (r01 extension, SW-03).

Method (force method, Maxwell-Mohr compatibility)
-------------------------------------------------
The beam is reduced to the statically determinate beam of ``stepped_beam`` on two pins at the first and
the last support (the *base*). The other unknowns are the redundant reactions: the force ``R_i`` of every
interior support and the couple ``C_j`` of every support that has a rotational spring. Each redundant is
applied to the base as a point force or a point couple; the base is linear, so the deflection is the
superposition

.. math::

    v(x) = v_0(x) + \\sum_i R_i\\, v_{R_i}(x) + \\sum_j C_j\\, v_{C_j}(x) + v_A + (v_B - v_A)\\,
           \\frac{x - x_A}{x_B - x_A},

with ``v_0`` the base under the applied loads, ``v_{R_i}``, ``v_{C_j}`` the base under a unit force / couple,
and the last two terms the rigid-body motion of the base when its own two supports move (their springs and
settlements). The unknowns ``R_i``, ``C_j``, ``v_A``, ``v_B`` follow from one compatibility equation per
support condition:

* radial: the support force on the beam is ``R = -k_r (v(x_s) - s)``, i.e. ``v(x_s) - s + R / k_r = 0``
  (``k_r = inf``: ``v(x_s) = s``, a rigid support settled by ``s``);
* rotational: ``C = -k_theta (theta(x_s) - phi)``, i.e. ``theta(x_s) - phi + C / k_theta = 0``
  (``k_theta = inf``: a clamp at the rotation ``phi``; ``k_theta = 0``: no couple, not an unknown).

The base reactions ``R_A``, ``R_B`` are the statics of the base under all loads and redundants, so the
equations of the two base supports close the system (size = number of redundants + 2). Every field of the
base is exact to round-off (``stepped_beam``), hence so is the superposition. For rigid supports and a
uniform Euler-Bernoulli beam the result is the solution of Clapeyron's three-moment equation; for one span
with prescribed end deflections and slopes it is Harris's slope-deflection Eqs 9.24-9.27 (Harris2007_vol2, PDF
p. 281; Harris closes them with the nonlinear bearing law separately, sec. 9.3.2, Eqs 9.45-9.47), for two spans
Eqs 9.30-9.35 and 9.42-9.44 (PDF p. 282-285; Eq. 9.30 prints 3 (delta_1 - delta_2)/l where Eq. 9.24 has 2).

Sign convention: that of ``uniform_beam`` (forces, deflections, settlements along +v; couples, rotations
and theta positive from +x toward +v). A spring couple is a reaction couple like ``C_A`` of a fixed end.

Assumptions: those of ``stepped_beam`` (linear elastic, small slopes, piecewise-constant section, one
bending plane) plus linear springs. Units: mm, N, MPa, N.mm, rad; ``k_r`` [N/mm], ``k_theta`` [N.mm/rad].
Imports NumPy only (an r imports no AxisForge, D-0036 rule 13).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stepped_beam import DistributedLoad, Segment, solve_plane

# OPEN R5-140: v_prerequisites asks a closed form "by the three-moment equation"; the force method on the exact
# determinate solver is used instead (any number of supports, springs, settlements, stepped EI, Timoshenko) and
# Clapeyron's equation is a test (C18).
# Harris2007_vol2 ch. 9 senses (Eqs 9.6-9.10, Fig. 9.4, PDF p. 278-279; Fig. 9.10, p. 285): y and delta_r positive
# down (along P), theta = dy/dx, F upward on the shaft, M_1 CCW, M_2 CW. Map: delta = -settlement,
# theta_H = -theta, F = R, M_1 = C[0], M_2 = -C[1] (signed check C24; R5-141 answered)

__all__ = ["Support", "SupportedSolution", "solve_supported"]


@dataclass(frozen=True)
class Support:
    """One support of the beam.

    Attributes
    ----------
    x : float
        Position [mm].
    k_r : float
        Radial stiffness [N/mm], > 0; ``math.inf`` (default) for a rigid support.
    k_theta : float
        Rotational stiffness [N.mm/rad], >= 0; 0 (default) for a pin, ``math.inf`` for a clamp. It acts on the
        cross-section rotation ``theta`` (with shear, Timoshenko, not on dv/dx: the rotational DOF of a
        Timoshenko FEM node is the same quantity).
    settlement : float
        Prescribed displacement of the support along +v [mm] (of the spring's base when ``k_r`` is finite).
    rotation : float
        Prescribed rotation of the support [rad] (of the rotational spring's base), positive from +x toward +v.
    """

    x: float
    k_r: float = math.inf
    k_theta: float = 0.0
    settlement: float = 0.0
    rotation: float = 0.0


def _flex(k: float) -> float:
    """Compliance 1/k of a spring [mm/N or rad/(N.mm)]; 0 for a rigid one."""
    return 0.0 if math.isinf(k) else 1.0 / k


class SupportedSolution:
    """Superposition of base solutions; build it with :func:`solve_supported`.

    Attributes
    ----------
    supports : list of Support
        As given, sorted by position.
    R : ndarray
        Support forces on the beam [N], one per support, along +v.
    C : ndarray
        Support couples on the beam [N.mm], one per support (0 where ``k_theta = 0``).
    """

    def __init__(self, *, base, units, weights, vA, vB, xA, xB, supports, R, C):
        self._base, self._units, self._w = base, units, np.asarray(weights, dtype=float)
        self._vA, self._vB, self._xA, self._xB = vA, vB, xA, xB
        self.supports, self.R, self.C = supports, R, C

    def _sum(self, field: str, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = getattr(self._base, field)(x)
        for w, u in zip(self._w, self._units):
            out = out + w * getattr(u, field)(x)
        return out

    def v(self, x) -> np.ndarray:
        """Transverse deflection [mm]."""
        x = np.asarray(x, dtype=float)
        return self._sum("v", x) + self._vA + (self._vB - self._vA) * (x - self._xA) / (self._xB - self._xA)

    def theta(self, x) -> np.ndarray:
        """Cross-section rotation [rad] (with the rigid-body rotation of the base)."""
        return self._sum("theta", x) + (self._vB - self._vA) / (self._xB - self._xA)

    def M(self, x) -> np.ndarray:
        """Bending moment [N.mm] (the rigid-body motion adds none)."""
        return self._sum("M", x)

    def V(self, x) -> np.ndarray:
        """Shear force [N]."""
        return self._sum("V", x)


def solve_supported(point_loads: Sequence[tuple[float, float]], distributed: Sequence[DistributedLoad],
                    supports: Sequence[Support], segments: Sequence[Segment], length: float, *,
                    couples: Sequence[tuple[float, float]] = ()) -> SupportedSolution:
    """Solve one bending plane of a stepped beam on any number of rigid or elastic supports.

    Parameters
    ----------
    point_loads, distributed, segments, length, couples
        As :func:`stepped_beam.solve_plane` (forces [N], positions [mm], q [N/mm], couples [N.mm]); a change
        of section at a support gives a per-span EI there.
    supports : sequence of Support
        At least two, at distinct positions inside [0, length].

    Returns
    -------
    SupportedSolution
        ``v``, ``theta``, ``M``, ``V`` callables and the support reactions ``R`` [N], ``C`` [N.mm].

    Raises
    ------
    ValueError
        Fewer than two supports, two at one position, a support outside the beam, ``k_r <= 0`` or
        ``k_theta < 0``.

    Notes
    -----
    With at least two supports and every ``k_r > 0`` the beam is never a mechanism, so the compatibility
    system is not singular; very soft springs (compliance 1/k_r some 1e12 times the beam's unit deflections)
    make it ill-conditioned: check ``np.linalg.cond`` before trusting such a limit. ``M`` and ``V`` return the
    right-hand side of a jump at a support or a couple (compare at ``x_s +/- eps``).

    Examples
    --------
    Two equal spans L = 100 mm under uniform q = 1 N/mm, rigid supports: end reactions 3/8 q L, middle
    5/4 q L (EI cancels).

    >>> sol = solve_supported([], [DistributedLoad(0.0, 200.0, lambda x: -1.0)],
    ...                       [Support(0.0), Support(100.0), Support(200.0)], [Segment(0.0, 200.0, 1e9)], 200.0)
    >>> [round(float(r), 9) for r in sol.R]
    [37.5, 125.0, 37.5]
    """
    sup = sorted(supports, key=lambda s: s.x)
    xs = [s.x for s in sup]
    if len(sup) < 2:
        raise ValueError(f"need at least two supports, got {len(sup)}")
    if len(set(xs)) != len(xs) or not (0.0 <= xs[0] and xs[-1] <= length):
        raise ValueError(f"supports must be distinct and inside [0, {length}], got {xs}")
    bad = [s for s in sup if not (s.k_r > 0.0 and s.k_theta >= 0.0)]
    if bad:
        raise ValueError(f"need k_r > 0 (inf for rigid) and k_theta >= 0 (inf for a clamp), got {bad}")
    xA, xB = xs[0], xs[-1]

    base = solve_plane(point_loads, distributed, xA, xB, segments, length, couples=couples)
    # redundants: the force of each interior support, the couple of each support with a rotational spring
    red = [("R", i) for i in range(1, len(sup) - 1)] + [("C", i) for i, s in enumerate(sup) if s.k_theta > 0.0]
    units = [solve_plane([(xs[i], 1.0)] if kind == "R" else [], (), xA, xB, segments, length,
                         couples=[] if kind == "R" else [(xs[i], 1.0)]) for kind, i in red]
    n = len(red)
    # unknowns z = [X_1 .. X_n, v_A, v_B]; v(x) = v0 + sum X u + vA (1 - t) + vB t, t = (x - xA)/(xB - xA)
    span = xB - xA
    A = np.zeros((n + 2, n + 2))
    b = np.zeros(n + 2)

    def v_row(x):
        t = (x - xA) / span
        return np.array([float(u.v(x)) for u in units] + [1.0 - t, t]), float(base.v(x))

    def th_row(x):
        return np.array([float(u.theta(x)) for u in units] + [-1.0 / span, 1.0 / span]), float(base.theta(x))

    for row, (kind, i) in enumerate(red):
        s = sup[i]
        if kind == "R":       # v(x_i) - s + R_i / k_r = 0
            a, c0 = v_row(s.x)
            a[row] += _flex(s.k_r)
            A[row], b[row] = a, s.settlement - c0
        else:                 # theta(x_i) - phi + C_i / k_theta = 0
            a, c0 = th_row(s.x)
            a[row] += _flex(s.k_theta)
            A[row], b[row] = a, s.rotation - c0
    # base supports: v_A - s_A + R_A / k_A = 0, with R_A (and R_B) the base reactions under loads and redundants
    for row, s, attr in ((n, sup[0], "R_A"), (n + 1, sup[-1], "R_B")):
        A[row, row] = 1.0                                  # v_A (row n) or v_B (row n + 1)
        A[row, :n] = _flex(s.k_r) * np.array([getattr(u, attr) for u in units])
        b[row] = s.settlement - _flex(s.k_r) * getattr(base, attr)
    z = np.linalg.solve(A, b)
    X, vA, vB = z[:n], z[n], z[n + 1]

    R = np.zeros(len(sup))
    C = np.zeros(len(sup))
    R[0] = base.R_A + sum(x * u.R_A for x, u in zip(X, units))
    R[-1] = base.R_B + sum(x * u.R_B for x, u in zip(X, units))
    for x, (kind, i) in zip(X, red):
        (R if kind == "R" else C)[i] = x
    return SupportedSolution(base=base, units=units, weights=X, vA=vA, vB=vB, xA=xA, xB=xB, supports=sup, R=R, C=C)
