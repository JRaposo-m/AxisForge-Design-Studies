"""
references/beams/shear_coefficients.py -- Timoshenko shear
correction factors for a solid circular section.

Single definition, shared by every closed-form beam reference in
``references/beams/`` (it used to be copied into each case's
``analytical_solution.py``). Deliberately independent of AxisForge's own
``mesh/shaft/element_type/shear_factor.py``: a reference must not reuse
the code it checks.

References
----------
.. [1] Cowper, G. R. (1966). The shear coefficient in Timoshenko's beam
       theory. *J. Appl. Mech.* 33(2), 335-340.
.. [2] Hutchinson, J. R. (2001). Shear coefficients for Timoshenko beam
       theory. *J. Appl. Mech.* 68(1), 87-92.
"""
from __future__ import annotations

from typing import Callable

__all__ = ["kappa_cowper", "kappa_hutchinson", "KAPPA"]


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
