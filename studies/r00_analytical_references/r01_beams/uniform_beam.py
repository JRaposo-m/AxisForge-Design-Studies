"""
Uniform beam on two pin supports under point loads (Macaulay), and the section data both r01
beams share: EI and kappa G A from the design inputs (P-01), shear correction factors.

Euler-Bernoulli and Timoshenko (shear-corrected) deflection, bending moment and shear force in
one bending plane. For every transverse force F_k at x_k (applied loads and support reactions
alike) and Macaulay brackets <x - a>^n = (x - a)^n H(x - a):

.. math::

    M(x) = \\sum_k F_k \\langle x - x_k \\rangle, \\qquad
    V(x) = -\\frac{dM}{dx} = -\\sum_k F_k \\langle x - x_k \\rangle^0

    v(x) = \\frac{1}{EI}\\sum_k F_k \\frac{\\langle x - x_k\\rangle^3}{6}
         - \\frac{1}{\\kappa G A}\\sum_k F_k \\langle x - x_k\\rangle + C_1 x + C_2

The first deflection term is the bending (Euler-Bernoulli) part, EI v_b'' = M (Onate 2013,
Eq. (1.5)). The second is the shear part: the shear strain is gamma = v_s' = V / kGA with
V = -dM/dx, hence v_s = -M / kGA (+ linear terms) (Onate 2013, Eqs (2.1)-(2.6c)). With the
opposite sign the Timoshenko beam would come out stiffer than Euler-Bernoulli, which is
impossible. C_1, C_2 follow from v(x_A) = v(x_B) = 0 on the total deflection. Euler-Bernoulli is
the limit kGA -> infinity and is solved with ``kGA=None`` (shear term dropped).

Assumptions: linear elasticity, small slopes, uniform section, ideal pins (v = 0, rotation free),
statically determinate, point loads only, kappa constant along the beam.

Units: mm, N, MPa, N.mm. Imports NumPy only (an r imports no AxisForge, D-0036).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["kappa_cowper", "kappa_hutchinson", "KAPPA", "section_stiffness", "PlaneSolution", "solve_plane"]


# =============================================================================
# Section
# =============================================================================

def kappa_cowper(nu: float) -> float:
    """Cowper shear correction factor of a solid circular section [-].

    ``kappa = 6 (1 + nu) / (7 + 6 nu)`` (Cowper 1966, Table 1, solid circle; source not held in
    the library: README section 2).

    Examples
    --------
    >>> round(kappa_cowper(0.3), 6)
    0.886364
    """
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def kappa_hutchinson(nu: float) -> float:
    """Hutchinson shear correction factor of a solid circular section [-].

    ``kappa = 6 (1 + nu)^2 / (7 + 12 nu + 4 nu^2)`` (Hutchinson 2001, solid circle; source not
    held in the library: README section 2).

    Examples
    --------
    >>> round(kappa_hutchinson(0.3), 6)
    0.925182
    """
    return 6.0 * (1.0 + nu) ** 2 / (7.0 + 12.0 * nu + 4.0 * nu**2)


KAPPA = {"cowper": kappa_cowper, "hutchinson": kappa_hutchinson}


def section_stiffness(E: float, nu: float, d_o: float, d_i: float = 0.0,
                      kappa: str | None = "cowper") -> tuple[float, float | None]:
    """Bending and corrected shear stiffness of a circular section, from the design inputs.

    Parameters
    ----------
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio [-]; ``G = E / (2 (1 + nu))``.
    d_o, d_i : float
        Outer and inner diameter [mm]; ``d_i = 0`` for a solid section.
    kappa : {"cowper", "hutchinson"} or None
        Shear correction factor (``KAPPA``); None for Euler-Bernoulli.

    Returns
    -------
    EI : float
        Bending stiffness [N.mm^2], ``I = pi (d_o^4 - d_i^4) / 64``.
    kGA : float or None
        Corrected shear stiffness kappa G A [N], ``A = pi (d_o^2 - d_i^2) / 4``; None when
        ``kappa`` is None.

    Raises
    ------
    ValueError
        ``E <= 0``, ``nu`` outside (-1, 0.5), not ``0 <= d_i < d_o``, or an unknown ``kappa``.
    NotImplementedError
        A hollow section with a shear correction factor: the hollow Cowper and Hutchinson factors
        need sources the library does not hold (# OPEN MG-02).
    """
    if kappa is not None and kappa not in KAPPA:
        raise ValueError(f"kappa must be one of {sorted(KAPPA)} or None, got {kappa!r}")
    if not E > 0.0 or not -1.0 < nu < 0.5 or not 0.0 <= d_i < d_o:
        raise ValueError(f"expected E > 0, -1 < nu < 0.5, 0 <= d_i < d_o; got {E}, {nu}, {d_i}, {d_o}")
    EI = E * math.pi * (d_o**4 - d_i**4) / 64.0
    if kappa is None:
        return EI, None
    if d_i > 0.0:
        raise NotImplementedError("shear correction factor of a hollow section: Cowper 1966 / "
                                  "Hutchinson 2001 not in the library (OPEN MG-02)")
    G = E / (2.0 * (1.0 + nu))
    return EI, KAPPA[kappa](nu) * G * math.pi * (d_o**2 - d_i**2) / 4.0


# =============================================================================
# One bending plane
# =============================================================================

def _bracket(x: np.ndarray, a: float, n: int) -> np.ndarray:
    """Macaulay bracket <x - a>^n, vectorised; <x - a>^0 = H(x - a), H(0) = 1."""
    d = x - a
    step = (d >= 0.0).astype(float)
    return step if n == 0 else d**n * step


@dataclass
class PlaneSolution:
    """Closed-form solution in one bending plane; build it with :func:`solve_plane`.

    Attributes
    ----------
    x_A, x_B : float
        Pin support positions [mm], x_A < x_B.
    R_A, R_B : float
        Support reactions [N], same sign convention as the applied loads.
    loads : list of (float, float)
        Applied loads (position [mm], force [N]); reactions excluded.
    EI : float
        Bending stiffness [N.mm^2].
    kGA : float or None
        Corrected shear stiffness [N]; None for Euler-Bernoulli.
    """

    x_A: float
    x_B: float
    R_A: float
    R_B: float
    loads: list[tuple[float, float]]
    EI: float
    kGA: float | None
    _C: tuple[float, float] = field(default=(0.0, 0.0), repr=False)

    @property
    def forces(self) -> list[tuple[float, float]]:
        """Applied loads plus reactions, i.e. every point force [N]."""
        return [*self.loads, (self.x_A, self.R_A), (self.x_B, self.R_B)]

    def _v_particular(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for xk, Fk in self.forces:
            out += Fk * _bracket(x, xk, 3) / (6.0 * self.EI)
            if self.kGA is not None:
                out -= Fk * _bracket(x, xk, 1) / self.kGA   # v_s' = V/kGA, V = -dM/dx
        return out

    def v(self, x) -> np.ndarray:
        """Transverse deflection [mm] at positions ``x`` [mm]."""
        x = np.asarray(x, dtype=float)
        C1, C2 = self._C
        return self._v_particular(x) + C1 * x + C2

    def v_bending(self, x) -> np.ndarray:
        """Bending-only part of the deflection [mm] (same supports)."""
        return solve_plane(self.loads, self.x_A, self.x_B, self.EI, None).v(x)

    def M(self, x) -> np.ndarray:
        """Bending moment [N.mm] at positions ``x`` [mm]."""
        x = np.asarray(x, dtype=float)
        return sum((Fk * _bracket(x, xk, 1) for xk, Fk in self.forces), start=np.zeros_like(x))

    def V(self, x) -> np.ndarray:
        """Shear force V = -dM/dx [N] at positions ``x`` [mm]."""
        x = np.asarray(x, dtype=float)
        return -sum((Fk * _bracket(x, xk, 0) for xk, Fk in self.forces), start=np.zeros_like(x))


def solve_plane(loads: list[tuple[float, float]], x_A: float, x_B: float,
                EI: float, kGA: float | None) -> PlaneSolution:
    """Solve one bending plane of a uniform two-pin beam under point loads.

    Parameters
    ----------
    loads : list of (float, float)
        Applied transverse loads (position [mm], force [N]) in this plane.
    x_A, x_B : float
        Pin support positions [mm], x_A < x_B.
    EI : float
        Bending stiffness [N.mm^2] (:func:`section_stiffness`).
    kGA : float or None
        Corrected shear stiffness kappa G A [N] (:func:`section_stiffness`); None for
        Euler-Bernoulli.

    Returns
    -------
    PlaneSolution
        Reactions from static equilibrium and callables v, M, V.

    Raises
    ------
    ValueError
        If x_A >= x_B.

    Notes
    -----
    Sign convention: forces and deflections share one positive direction per plane;
    M = sum F_k <x - x_k>, V = -dM/dx. A shaft loaded in two planes is two calls; the resultant
    is ``hypot`` of the two (circular section, uncoupled planes).

    Examples
    --------
    Simply supported span L = 100 mm, central load 1000 N, EI = 1e9 N.mm^2; the bending-only
    mid-span deflection is F L^3 / (48 EI):

    >>> sol = solve_plane([(50.0, 1000.0)], 0.0, 100.0, EI=1e9, kGA=None)
    >>> round(float(sol.v(50.0)), 9), round(1000.0 * 100.0**3 / (48 * 1e9), 9)
    (0.020833333, 0.020833333)
    """
    if x_A >= x_B:
        raise ValueError(f"expected x_A < x_B, got {x_A}, {x_B}")
    sum_F = sum(F for _, F in loads)
    sum_M_A = sum(F * (x - x_A) for x, F in loads)
    R_B = -sum_M_A / (x_B - x_A)
    R_A = -sum_F - R_B
    sol = PlaneSolution(x_A, x_B, R_A, R_B, list(loads), EI, kGA)
    rhs = -sol._v_particular(np.array([x_A, x_B]))               # v(x_A) = v(x_B) = 0
    C1, C2 = np.linalg.solve(np.array([[x_A, 1.0], [x_B, 1.0]]), rhs)
    sol._C = (float(C1), float(C2))
    return sol
