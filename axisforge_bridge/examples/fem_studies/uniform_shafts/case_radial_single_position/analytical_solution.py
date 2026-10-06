"""
Closed-form reference solution -- simply supported shaft with overhangs.

Euler-Bernoulli and Timoshenko (shear-corrected) deflection, bending
moment and shear force for a uniform beam on two pin supports under
transverse point loads, by Macaulay's method. Used as the exact
reference for ``case_radial_single_position``.

Method
------
For one bending plane, with every transverse force F_k at x_k (applied
loads and support reactions alike) and Macaulay brackets
<x - a>^n = (x - a)^n H(x - a):

.. math::

    M(x) = \\sum_k F_k \\langle x - x_k \\rangle

    V(x) = -\\frac{dM}{dx} = -\\sum_k F_k \\langle x - x_k \\rangle^0

    v(x) = \\frac{1}{EI}\\sum_k F_k \\frac{\\langle x - x_k\\rangle^3}{6}
         - \\frac{1}{\\kappa G A}\\sum_k F_k \\langle x - x_k\\rangle
         + C_1 x + C_2

The first deflection term is the bending (Euler-Bernoulli) part,
EI v_b'' = M. The second is the shear part: the shear strain is
gamma = v_s' = V / kGA with V = -dM/dx, hence v_s = -M / kGA (+ linear
terms). Its sign is fixed by V = -dM/dx; with the opposite sign the
Timoshenko beam would come out STIFFER than Euler-Bernoulli, which is
physically impossible. C_1, C_2 follow from v(x_A) = v(x_B) = 0 applied to the total
deflection. Euler-Bernoulli is the limit kGA -> infinity; it is solved
with ``kGA=None`` (shear term dropped) instead of a large number.

Assumptions
-----------
* Linear elasticity, small displacements, uniform section.
* Supports are ideal pins (v = 0, rotation free) -- valid because the
  AxisForge rigid-support solver never constrains bending rotation.
* The two bending planes (XY, XZ) are uncoupled (circular section).
* Point loads only; no distributed load, no axial-bending coupling.

Scope
-----
Independent from the FEM discretisation (mesh, shape functions,
stiffness matrix), NOT from the load resolution: the applied forces are
read from the same resolved ``ShaftSystem`` that the FEM solves
(``Load.component(plane)``), so gear-force errors would go undetected
here by design.

References
----------
.. [1] Timoshenko, S. P. & Gere, J. M., *Mechanics of Materials*,
       Macaulay / singularity functions for beam deflection.
.. [2] Cowper, G. R. (1966). The shear coefficient in Timoshenko's beam
       theory. *J. Appl. Mech.* 33(2), 335-340.
.. [3] Hutchinson, J. R. (2001). Shear coefficients for Timoshenko beam
       theory. *J. Appl. Mech.* 68(1), 87-92.

Units: mm, N, MPa, N.mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from axisforge.core.loads import LoadPlane


# =============================================================================
# Shear correction factors -- solid circular section
# =============================================================================

def kappa_cowper(nu: float) -> float:
    """Cowper shear correction factor, solid circular section.

    Parameters
    ----------
    nu : float
        Poisson's ratio [-].

    Returns
    -------
    float
        kappa = 6 (1 + nu) / (7 + 6 nu) [-] (Cowper 1966; table/equation
        number to confirm).

    Examples
    --------
    >>> round(kappa_cowper(0.3), 6)
    0.886364
    """
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def kappa_hutchinson(nu: float) -> float:
    """Hutchinson shear correction factor, solid circular section.

    Parameters
    ----------
    nu : float
        Poisson's ratio [-].

    Returns
    -------
    float
        kappa = 6 (1 + nu)^2 / (7 + 12 nu + 4 nu^2) [-]
        (Hutchinson 2001, solid circle; equation number to confirm).

    Examples
    --------
    >>> round(kappa_hutchinson(0.3), 6)
    0.925182
    """
    return 6.0 * (1.0 + nu) ** 2 / (7.0 + 12.0 * nu + 4.0 * nu**2)


KAPPA: dict[str, Callable[[float], float]] = {
    "cowper": kappa_cowper,
    "hutchinson": kappa_hutchinson,
}


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
    """Closed-form solution in one bending plane.

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
        bending = PlaneSolution(self.x_A, self.x_B, self.R_A, self.R_B,
                                self.loads, self.EI, None)
        bending._C = _integration_constants(bending)
        return bending.v(x)

    def M(self, x) -> np.ndarray:
        """Bending moment [N.mm] at positions ``x`` [mm]."""
        x = np.asarray(x, dtype=float)
        return sum((Fk * _bracket(x, xk, 1) for xk, Fk in self.forces),
                   start=np.zeros_like(x))

    def V(self, x) -> np.ndarray:
        """Shear force V = -dM/dx [N] at positions ``x`` [mm]."""
        x = np.asarray(x, dtype=float)
        return -sum((Fk * _bracket(x, xk, 0) for xk, Fk in self.forces),
                    start=np.zeros_like(x))


def _integration_constants(sol: PlaneSolution) -> tuple[float, float]:
    """C1, C2 from v(x_A) = v(x_B) = 0 on the total deflection."""
    pts = np.array([sol.x_A, sol.x_B])
    rhs = -sol._v_particular(pts)
    C1, C2 = np.linalg.solve(np.array([[sol.x_A, 1.0], [sol.x_B, 1.0]]), rhs)
    return float(C1), float(C2)


def solve_plane(loads: list[tuple[float, float]], x_A: float, x_B: float,
                EI: float, kGA: float | None) -> PlaneSolution:
    """Solve one bending plane of a two-pin beam under point loads.

    Parameters
    ----------
    loads : list of (float, float)
        Applied transverse loads (position [mm], force [N]) in this plane.
    x_A, x_B : float
        Pin support positions [mm], x_A < x_B.
    EI : float
        Bending stiffness [N.mm^2].
    kGA : float or None
        Corrected shear stiffness kappa G A [N]; None for Euler-Bernoulli.

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
    Sign convention: forces and deflections share one positive direction
    per plane; M = sum F_k <x - x_k>, V = -dM/dx.

    Examples
    --------
    Simply supported span L = 100 mm, central load 1000 N, EI = 1e9 N.mm^2;
    the bending-only mid-span deflection is F L^3 / (48 EI):

    >>> sol = solve_plane([(50.0, 1000.0)], 0.0, 100.0, EI=1e9, kGA=None)
    >>> round(float(sol.v(50.0)), 9), round(1000.0 * 100.0**3 / (48 * 1e9), 9)
    (0.020833333, 0.020833333)
    >>> round(sol.R_A, 6), round(sol.R_B, 6)
    (-500.0, -500.0)
    """
    if x_A >= x_B:
        raise ValueError(f"expected x_A < x_B, got {x_A}, {x_B}")
    sum_F = sum(F for _, F in loads)
    sum_M_A = sum(F * (x - x_A) for x, F in loads)
    R_B = -sum_M_A / (x_B - x_A)
    R_A = -sum_F - R_B
    sol = PlaneSolution(x_A, x_B, R_A, R_B, list(loads), EI, kGA)
    sol._C = _integration_constants(sol)
    return sol


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
        Solutions in the XY and XZ planes; their ``v``, ``M``, ``V`` map
        onto ``ShaftResults.v_xy`` / ``v_xz`` etc.
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


def solve_shaft(shaft_system, EI: float, kGA: float | None) -> ShaftSolution:
    """Closed-form solution for one resolved AxisForge shaft.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Resolved shaft (user and gear-mesh radial loads already present).
    EI : float
        Bending stiffness [N.mm^2].
    kGA : float or None
        Corrected shear stiffness [N]; None for Euler-Bernoulli.

    Returns
    -------
    ShaftSolution

    Examples
    --------
    >>> import case_definition as cd
    >>> p = cd.section_properties()
    >>> sol = solve_shaft(cd.build_system().shafts[0], p.EI, kappa_cowper(p.nu) * p.GA)
    >>> round(float(sol.v(100.0)), 4)
    0.3763
    """
    x_A, x_B = sorted(b.position for b in shaft_system.bearings)

    def plane_loads(plane):
        out = [(ld.position, float(ld.component(plane))) for ld in shaft_system.radial_loads]
        return [(x, F) for x, F in out if F != 0.0]

    return ShaftSolution(
        name=shaft_system.name,
        xy=solve_plane(plane_loads(LoadPlane.XY), x_A, x_B, EI, kGA),
        xz=solve_plane(plane_loads(LoadPlane.XZ), x_A, x_B, EI, kGA),
    )