"""Involute geometry of an external cylindrical gear pair, its path of contact and mesh forces.

Independent judge of v107 (rule 13: own formulas, NumPy / SciPy only). Ideal involutes, no flank
modifications, rigid teeth, an external pair. Units: mm, N, N.m (torque), angles in degrees at the
interface and radians inside and in the results (``_rad``). Equation numbers of ISO 21771:2007 as read
from the page images (README section 2).

References
----------
ISO 21771:2007, Gears - Cylindrical involute gears and gear pairs - Concepts and geometry, clauses 4-5.
Fernandes et al. (2015b), Table 1 (PDF p. 4); Martins et al. (2015), Tables 3-4 (PDF p. 4): the tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

__all__ = ["inv", "inv_inverse", "PairGeometry", "pair_geometry", "PathPoint", "path_point", "MeshForces",
           "mesh_forces", "tip_diameter"]


def inv(alpha: float) -> float:
    """Involute function ``inv(alpha) = tan(alpha) - alpha`` [rad]."""
    return math.tan(alpha) - alpha


def inv_inverse(value: float) -> float:
    """The angle ``alpha`` in [0, pi/2) [rad] with ``inv(alpha) = value`` (``value >= 0``), to 1e-15 rad."""
    if value < 0.0:
        raise ValueError(f"inv_inverse: the involute function is >= 0, got {value}")
    if value == 0.0:
        return 0.0
    # inv is increasing on [0, pi/2) and inv(alpha) ~ alpha^3/3 near 0: bracket from above
    return brentq(lambda a: inv(a) - value, 0.0, min(math.pi / 2 - 1e-9, (3.0 * value) ** (1 / 3) + 1.0),
                  xtol=1e-15, rtol=1e-15)


def tip_diameter(z: int, m_n: float, beta_deg: float, x: float, h_a_star: float = 1.0) -> float:
    """Tip diameter ``d + 2 m_n (h_a* + x)`` [mm], no tip alteration (ISO 21771 tip from the basic rack)."""
    return z * m_n / math.cos(math.radians(beta_deg)) + 2.0 * m_n * (h_a_star + x)


@dataclass(frozen=True)
class PairGeometry:
    """Geometry of an external pair (pinion 1, wheel 2) [mm, rad]; ``T1*`` along the line of action from T1."""

    z1: int
    z2: int
    m_n: float
    alpha_n_rad: float
    beta_rad: float
    x1: float
    x2: float
    b: float
    alpha_t_rad: float
    beta_b_rad: float
    d1: float
    d2: float
    d_b1: float
    d_b2: float
    d_a1: float
    d_a2: float
    alpha_wt_rad: float
    a: float           # reference centre distance
    a_w: float         # working centre distance, from x1 + x2
    d_w1: float
    d_w2: float
    beta_w_rad: float  # helix angle at the working pitch cylinder
    p_bt: float
    T1T2: float
    T1A: float
    T1B: float
    T1C: float
    T1D: float
    T1E: float
    g_alpha: float
    eps_alpha: float
    eps_beta: float


def pair_geometry(z1: int, z2: int, m_n: float, alpha_n_deg: float, beta_deg: float, x1: float, x2: float,
                  b: float, *, d_a1: float | None = None, d_a2: float | None = None) -> PairGeometry:
    """Geometry of an external involute pair from its basic data; tips from ``x`` when not given.

    The working pressure angle comes from the sum of the profile shifts (zero backlash),
    ``inv(alpha_wt) = inv(alpha_t) + 2 (x1 + x2) tan(alpha_n) / (z1 + z2)``, and the working centre
    distance ``a_w = a cos(alpha_t) / cos(alpha_wt)``. Points on the line of action T1-T2 (T1, T2 the
    tangent points on the base circles), from T1: ``E`` (pinion tip), ``A`` (wheel tip), ``C`` (pitch
    point), ``B = E - p_bt``, ``D = A + p_bt``. ``g_alpha = T1E - T1A``, ``eps_alpha = g_alpha / p_bt``,
    ``eps_beta = b sin(beta) / (pi m_n)``.
    """
    if z1 <= 0 or z2 <= 0:
        raise ValueError("pair_geometry: an external pair needs z1, z2 > 0")
    alpha_n, beta = math.radians(alpha_n_deg), math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    beta_b = math.atan(math.tan(beta) * math.cos(alpha_t))
    d1, d2 = z1 * m_n / math.cos(beta), z2 * m_n / math.cos(beta)
    d_b1, d_b2 = d1 * math.cos(alpha_t), d2 * math.cos(alpha_t)
    d_a1 = tip_diameter(z1, m_n, beta_deg, x1) if d_a1 is None else d_a1
    d_a2 = tip_diameter(z2, m_n, beta_deg, x2) if d_a2 is None else d_a2
    alpha_wt = inv_inverse(inv(alpha_t) + 2.0 * (x1 + x2) * math.tan(alpha_n) / (z1 + z2))
    a = (d1 + d2) / 2.0
    a_w = a * math.cos(alpha_t) / math.cos(alpha_wt)
    d_w1, d_w2 = d_b1 / math.cos(alpha_wt), d_b2 / math.cos(alpha_wt)
    beta_w = math.atan(math.tan(beta) * d_w1 / d1)
    p_bt = math.pi * m_n * math.cos(alpha_t) / math.cos(beta)
    T1T2 = a_w * math.sin(alpha_wt)
    T1E = math.sqrt((d_a1 / 2) ** 2 - (d_b1 / 2) ** 2)
    T1A = T1T2 - math.sqrt((d_a2 / 2) ** 2 - (d_b2 / 2) ** 2)
    T1C = d_b1 / 2 * math.tan(alpha_wt)
    g_alpha = T1E - T1A
    return PairGeometry(z1=z1, z2=z2, m_n=m_n, alpha_n_rad=alpha_n, beta_rad=beta, x1=x1, x2=x2, b=b,
                        alpha_t_rad=alpha_t, beta_b_rad=beta_b, d1=d1, d2=d2, d_b1=d_b1, d_b2=d_b2, d_a1=d_a1,
                        d_a2=d_a2, alpha_wt_rad=alpha_wt, a=a, a_w=a_w, d_w1=d_w1, d_w2=d_w2, beta_w_rad=beta_w,
                        p_bt=p_bt, T1T2=T1T2, T1A=T1A, T1B=T1E - p_bt, T1C=T1C, T1D=T1A + p_bt, T1E=T1E,
                        g_alpha=g_alpha, eps_alpha=g_alpha / p_bt, eps_beta=b * math.sin(beta) / (math.pi * m_n))


@dataclass(frozen=True)
class PathPoint:
    """Flank curvatures and specific sliding at a point Y of the line of action [mm, -]."""

    T1Y: float
    on_path: bool      # T1A <= T1Y <= T1E (pinion driving)
    single_pair: bool  # T1B <= T1Y <= T1D: one pair in contact (a spur pair; a helical pair shares by L)
    rho1_t: float      # pinion flank radius of curvature, transverse plane = T1Y
    rho2_t: float      # wheel flank, = T1T2 - T1Y
    rho1_n: float      # normal plane, rho_t / cos(beta_b)
    rho2_n: float
    rho_red_n: float   # reduced radius rho1_n rho2_n / (rho1_n + rho2_n)
    zeta1: float       # specific sliding of the pinion, 1 - (rho2 z1) / (rho1 z2)
    zeta2: float       # of the wheel, 1 - (rho1 z2) / (rho2 z1)


def path_point(g: PairGeometry, T1Y: float) -> PathPoint:
    """Curvatures and specific sliding at ``T1Y`` [mm] from T1, anywhere on T1-T2 (``on_path`` flags A..E).

    Pinion driving: contact runs from A to E; with the wheel driving the points are the same on the
    line of action, the traversal is reversed and the other flank is loaded. ``zeta`` is the specific
    sliding of ISO 21771, not the slide-to-roll ratio of the EHL studies (no speeds here).

    ``rho1 = T1Y``, ``rho2 = T1T2 - T1Y`` (rho_y = r_b tan(alpha_y)); the rolling speeds are
    ``omega_k rho_k`` with ``omega1 / omega2 = z2 / z1``, so the specific sliding
    ``zeta1 = (v1 - v2) / v1 = 1 - rho2 z1 / (rho1 z2)`` and ``zeta2 = 1 - rho1 z2 / (rho2 z1)``.
    """
    if not 0.0 < T1Y < g.T1T2:
        raise ValueError(f"path_point: T1Y = {T1Y} mm lies outside the line of action (0, {g.T1T2})")
    rho1, rho2 = T1Y, g.T1T2 - T1Y
    cb = math.cos(g.beta_b_rad)
    tol = 1e-9 * g.p_bt
    return PathPoint(T1Y=T1Y, on_path=g.T1A - tol <= T1Y <= g.T1E + tol,
                     single_pair=g.T1B - tol <= T1Y <= g.T1D + tol, rho1_t=rho1, rho2_t=rho2, rho1_n=rho1 / cb, rho2_n=rho2 / cb,
                     rho_red_n=rho1 * rho2 / (rho1 + rho2) / cb,
                     zeta1=1.0 - rho2 * g.z1 / (rho1 * g.z2), zeta2=1.0 - rho1 * g.z2 / (rho2 * g.z1))


@dataclass(frozen=True)
class MeshForces:
    """Mesh forces of the pair [N] and the moment of the axial force on each gear [N.m].

    ``F_a`` is parallel to the gear axis, so it has no moment about that axis: ``F_a r_w`` is the tilting
    (bending) moment about the axis perpendicular to the gear axis in the plane of the forces (the couple
    a shaft model receives; its sign is the caller's, D-0010 / CW-01).
    """

    F_t: float          # tangential at the working pitch circle
    F_r: float          # radial, F_t tan(alpha_wt)
    F_a: float          # axial, F_t tan(beta_w)
    F_bn: float         # normal, in the plane of action, F_t / (cos(alpha_wt) cos(beta_b))
    couple_1: float     # F_a r_w1 [N.m], tilting moment of the axial force on the pinion
    couple_2: float     # F_a r_w2 [N.m] on the wheel


def mesh_forces(g: PairGeometry, torque_Nm: float, *, on: int = 2) -> MeshForces:
    """Mesh forces from the torque [N.m] on gear ``on`` (1 pinion, 2 wheel), at the working pitch circle.

    ``F_t = 2000 T / d_w`` [N] (T in N.m, d_w in mm), ``F_r = F_t tan(alpha_wt)``, ``F_a = F_t tan(beta_w)``,
    ``F_bn = F_t / (cos(alpha_wt) cos(beta_b))``; magnitudes (directions are the caller's, D-0010).
    ``F_t`` is at the WORKING pitch circle (as Martins 2015 Table 4 confirms through F_r); the ISO 6336
    nominal tangential load is at the reference circle (C40: d_w/d = 1.0167): do not feed this ``F_t`` to an
    ISO 6336-2 formula. ``F_bn`` and ``F_a`` are the same either way.
    """
    if on not in (1, 2):
        raise ValueError(f"mesh_forces: on must be 1 (pinion) or 2 (wheel), got {on!r}")
    d_w = g.d_w1 if on == 1 else g.d_w2
    F_t = 2000.0 * abs(torque_Nm) / d_w
    F_a = F_t * math.tan(g.beta_w_rad)
    return MeshForces(F_t=F_t, F_r=F_t * math.tan(g.alpha_wt_rad), F_a=F_a,
                      F_bn=F_t / (math.cos(g.alpha_wt_rad) * math.cos(g.beta_b_rad)),
                      couple_1=F_a * g.d_w1 / 2000.0, couple_2=F_a * g.d_w2 / 2000.0)
