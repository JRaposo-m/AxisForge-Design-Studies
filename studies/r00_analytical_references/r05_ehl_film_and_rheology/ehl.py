"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

EHL film-thickness references: the published dimensionless groups and film formulas, the judge of
``slippy_bridge.lubrication.ehl.run``. NumPy only, by rule (D-0021; no body needs it yet); nothing from this repository.

Conventions (each function converts to them at its entry, nowhere else):

* Every function takes the MEAN entrainment speed ``u_mean = (u1 + u2)/2`` [m/s], the speed the
  bridge records (R-075). A formula written with the sum ``u_Sigma = u1 + u2`` gets ``2 u_mean``
  inside the function.
* Moes M and L are in the SUM convention (VennerBos1994 pp. 14-15): their speed group is
  ``U_sum = eta0 u_Sigma / (E' R_x)``.
* ``a`` is the Hertz semi-axis along rolling, ``s_x`` in the bridge (the half-width of a line
  contact, the radius of a circular one).
* Reduced modulus ``E' = 2 / [(1 - nu1^2)/E1 + (1 - nu2^2)/E2]`` (VennerBos1994 p. 14, Hamrock-Dowson
  p. 2, Canzi2010 p. 7) and ``E* = E'/2``, the contact modulus of Hertz, SlipPY ``e_star`` (R-155).
* Hamrock-Dowson is not mapped until H-D 1976 Part I is held: mean or sum speed in its U (R-092).

References
----------
VennerBos1994 pp. 14-15 (Moes groups, H scalings); Canzi2010 Eq. 3-4, 6-11, 14-18, 24 (Moes and
CELINE film formulas).
"""
from __future__ import annotations

__all__ = ["e_prime", "groups", "canzi_moes", "moes_inverse", "moes", "celine", "film_scalings"]


def e_prime(E1_Pa: float, nu1: float, E2_Pa: float, nu2: float) -> dict:
    """Reduced modulus of two elastic bodies in contact, in the EHL and the Hertz form.

    Each body deflects under the contact pressure in proportion to ``(1 - nu^2)/E``; the two
    compliances add, so the pair acts as one rigid body against an elastic half-space of modulus
    ``E*`` (Hertz) or, in the EHL literature, ``E' = 2 E*``.

    Parameters
    ----------
    E1_Pa, E2_Pa : float
        Young's moduli of body 1 and body 2 [Pa], > 0.
    nu1, nu2 : float
        Poisson's ratios [-], in (0, 0.5), as ``ContactCase``.

    Returns
    -------
    dict
        ``E_prime_Pa``: ``2 / [(1 - nu1^2)/E1 + (1 - nu2^2)/E2]`` [Pa] (EHL); ``E_star_Pa``: ``E'/2``
        [Pa] (Hertz, SlipPY ``e_star``).

    Raises
    ------
    ValueError
        If a modulus is not positive or a Poisson's ratio is outside (0, 0.5).

    References
    ----------
    VennerBos1994 p. 14; Hamrock-Dowson p. 2; Canzi2010 p. 7; Johnson 1985 ch. 4 (E*).
    """
    if not (E1_Pa > 0 and E2_Pa > 0):
        raise ValueError(f"moduli must be positive, got E1 = {E1_Pa}, E2 = {E2_Pa}")
    if not (0.0 < nu1 < 0.5 and 0.0 < nu2 < 0.5):
        raise ValueError(f"Poisson's ratios must be in (0, 0.5), got nu1 = {nu1}, nu2 = {nu2}")
    E_prime = 2.0 / ((1.0 - nu1**2) / E1_Pa + (1.0 - nu2**2) / E2_Pa)
    return {"E_prime_Pa": E_prime, "E_star_Pa": E_prime / 2.0}


def groups(*, u_mean_m_per_s: float, eta0_Pa_s: float, E_prime_Pa: float, R_x_m: float, alpha_per_Pa: float,
           load_N: float | None = None, w_N_per_m: float | None = None) -> dict:
    """Dimensionless groups of an EHL contact: speed, material, load, and Moes M and L.

    The film of a smooth, isothermal, Newtonian EHL contact depends on three groups only: a speed
    group ``U = eta0 u / (E' R_x)`` (viscous drag against elastic stiffness), a material group
    ``G = alpha E'`` (how much the viscosity rises over a pressure of order E') and a load group
    ``W``. Moes combined them into two, ``M`` (load) and ``L`` (lubricant), in the sum convention.

    Parameters
    ----------
    u_mean_m_per_s : float
        Mean entrainment speed (u1 + u2)/2 [m/s].
    eta0_Pa_s : float
        Viscosity at ambient pressure [Pa s].
    E_prime_Pa : float
        Reduced modulus E' [Pa] (:func:`e_prime`).
    R_x_m : float
        Reduced radius of curvature along rolling [m], ``1/R_x = 1/r1_x + 1/r2_x`` from the signed
        radii (concave < 0); for a gear the radii of the normal section, spur only (R-027).
    alpha_per_Pa : float
        Pressure-viscosity coefficient [1/Pa] (which alpha*: OPEN R-091).
    load_N : float, optional
        Normal load of a point contact [N].
    w_N_per_m : float, optional
        Load per unit length of a line contact [N/m]. Exactly one of ``load_N`` and ``w_N_per_m``.

    Returns
    -------
    dict
        ``U_mean`` (with u_mean), ``U_sum`` (with 2 u_mean), ``G``, ``W_point`` = ``load/(E' R_x^2)``
        or ``W_line`` = ``w/(E' R_x)``, ``M_moes``, ``L_moes`` [-] (sum convention, VennerBos1994
        pp. 14-15; the exponents to be read there: OPEN R-093).

    Raises
    ------
    ValueError
        If not exactly one of ``load_N`` and ``w_N_per_m`` is given, or the given load, the speed,
        R_x, eta0 or alpha is not > 0 (a zero or negative value has no film).
    NotImplementedError
        Always, after the checks: scaffold (OPEN R-093).
    """
    if (load_N is None) == (w_N_per_m is None):
        raise ValueError("give exactly one of load_N (point contact) and w_N_per_m (line contact)")
    for key, value in (("load_N", load_N), ("w_N_per_m", w_N_per_m), ("u_mean_m_per_s", u_mean_m_per_s),
                       ("R_x_m", R_x_m), ("eta0_Pa_s", eta0_Pa_s), ("alpha_per_Pa", alpha_per_Pa)):
        if value is not None and not value > 0:
            raise ValueError(f"{key} must be > 0, got {value}")
    # OPEN R-091: alpha is the initial slope or Blok's alpha*; named by the caller
    raise NotImplementedError("groups: scaffold, OPEN R-093")


def canzi_moes(*, alpha_bar: float, M_canzi: float, line: bool, D: float | None = None) -> dict:
    """Moes M and L from the parameters (alpha_bar, M) of Canzi's tables.

    Canzi2010 tabulates the film against a pressure-viscosity parameter ``alpha_bar`` and a load
    parameter ``M``; this puts them in the Moes sum convention so that :func:`moes` and the bridge
    record can be compared with the published rows.

    Parameters
    ----------
    alpha_bar : float
        Canzi's dimensionless pressure-viscosity parameter [-].
    M_canzi : float
        Canzi's load parameter [-].
    line : bool
        True: line contact (Canzi2010 Eq. 3); False: point contact.
    D : float, optional
        Ellipse ratio ``R_x / R_y`` [-] (Canzi's convention). None: circular (Eq. 14) or line. Given:
        Eq. (19), ``alpha_bar = (L/pi) (3 M2 kappa/2)^(1/3) ((1+D) pi/(4 E_c))^(2/3)``, with
        ``kappa = a_x/a_y`` and the elliptic integral ``E_c(kappa)``; ``M_canzi`` is then M2c, the load
        parameter at D = 1, and M2 follows from Eq. (23), ``M2 = (M2c/kappa) [4 E_c/((1+D) pi)]^2``.
        The printed upper limit 2 pi of E_c is probably a misprint (pi/2; OPEN GL-13).

    Returns
    -------
    dict
        ``M_moes``, ``L_moes`` [-], sum convention.

    Raises
    ------
    NotImplementedError
        Always: scaffold (OPEN GL-13: the mapping needs a dimensional p_h, which Canzi does not print).
    """
    raise NotImplementedError("canzi_moes: scaffold, OPEN GL-13")


def moes_inverse(*, M: float, L: float, line: bool, E_prime_Pa: float, R_x_m: float, u_mean_m_per_s: float,
                 roelands_z: float, roelands_p0_Pa: float) -> dict:
    """The dimensional inputs of an EHL solve that reproduce a published (M, L) point.

    Benchmarks give only M and L. With the bodies (E', R_x), the mean speed and the Roelands
    constants fixed, ``L = alpha E' U_sum^(1/4)`` with the initial slope
    ``alpha = (ln eta0 + 9.67) z / p0`` is one equation in eta0 (solved for it), and ``M`` then
    gives the load (exponents as in :func:`groups`, OPEN R-093). SlipPY D-H (5.9e8 Pa, LM:216) and
    Roelands (9.67, LM:103-106) are dimensional: the SI point must reproduce the printed p_h as well
    as M, L, z (OPEN GL-34).

    Parameters
    ----------
    M, L : float
        Moes load and lubricant parameters [-], sum convention.
    line : bool
        True: line contact (load per unit length); False: point contact (load).
    E_prime_Pa : float
        Reduced modulus [Pa].
    R_x_m : float
        Reduced radius along rolling [m].
    u_mean_m_per_s : float
        Mean entrainment speed [m/s].
    roelands_z : float
        Roelands index z [-].
    roelands_p0_Pa : float
        Roelands reference pressure [Pa] (no source prints it for the cases: OPEN GL-18).

    Returns
    -------
    dict
        ``eta0_Pa_s`` [Pa s], ``alpha_initial_per_Pa`` [1/Pa] (slope of ln eta at p = 0), and
        ``load_N`` (point) or ``w_N_per_m`` (line).

    Raises
    ------
    NotImplementedError
        Always: scaffold (OPEN R-091 R-114 GL-18).
    """
    raise NotImplementedError("moes_inverse: scaffold, OPEN R-091 R-114 GL-18")


def moes(*, M: float, L: float, line: bool, compressible: bool, p_h_Pa: float | None = None) -> dict:
    """Moes' curve-fit formulas for the central and minimum film, in Moes' H.

    Moes interpolated between the asymptotes of the film (rigid-isoviscous, elastic-isoviscous,
    rigid-piezoviscous, elastic-piezoviscous) with one formula in M and L, as reproduced by
    Canzi2010 (Eq. 4, 6-10 line; Eq. 15-18 circular), with a compressibility correction (Eq. 11)
    that needs the dimensional Hertz pressure.

    Parameters
    ----------
    M, L : float
        Moes parameters [-], sum convention.
    line : bool
        True: line contact; False: circular contact.
    compressible : bool
        True: Dowson-Higginson density correction (Canzi2010 Eq. 11).
    p_h_Pa : float, optional
        Maximum Hertz pressure [Pa], >= 1e6 (a smaller value was given in MPa, R-077); required when
        ``compressible``.

    Returns
    -------
    dict
        ``H_c``, ``H_m``: central and minimum film in Moes' scaling ``(h/R_x) U_sum^(-1/2)`` [-].

    Raises
    ------
    ValueError
        If ``compressible`` and ``p_h_Pa`` is None, or ``p_h_Pa`` is below 1e6 Pa.
    NotImplementedError
        Always, after the check: scaffold (OPEN GL-12: the outer exponent of Eq. 15; GL-13; R-114).
    """
    if compressible and p_h_Pa is None:
        raise ValueError("a compressible Moes film needs p_h_Pa, the dimensional Hertz pressure (Canzi2010 Eq. 11)")
    # the Dowson-Higginson constant 5.9e8 is in Pa: a p_h in MPa gives a constant density (R-077)
    if p_h_Pa is not None and not p_h_Pa >= 1e6:
        raise ValueError(f"p_h_Pa = {p_h_Pa} is below 1e6 Pa (a value in MPa?)")
    raise NotImplementedError("moes: scaffold, OPEN GL-12 GL-13 R-114")


def celine(*, M: float, L: float, D: float) -> dict:
    """CELINE film formula of an elliptical contact (Canzi2010 Eq. 24).

    Parameters
    ----------
    M, L : float
        Moes parameters [-], sum convention.
    D : float
        Ellipticity parameter of Canzi2010 Eq. 24 [-] (definition to be read on the page).

    Returns
    -------
    dict
        ``H_c``, ``H_m`` [-], Moes' scaling.

    Raises
    ------
    NotImplementedError
        Always: scaffold (OPEN GL-13).
    """
    raise NotImplementedError("celine: scaffold, OPEN GL-13")


def film_scalings(*, h_m: float, R_x_m: float, s_x_ref_m: float, U_mean: float) -> dict:
    """A film thickness in the two published dimensionless scalings.

    Parameters
    ----------
    h_m : float
        Film thickness [m] (central or minimum).
    R_x_m : float
        Reduced radius along rolling [m].
    s_x_ref_m : float
        Hertz semi-axis along rolling, ``a`` [m].
    U_mean : float
        Speed group with the mean speed, ``eta0 u_mean / (E' R_x)`` [-].

    Returns
    -------
    dict
        ``H_venner = h R_x / a^2`` (VennerBos1994) and ``H_moes = (h / R_x) U_sum^(-1/2)`` with
        ``U_sum = 2 U_mean`` [-] (OPEN R-093).

    Raises
    ------
    NotImplementedError
        Always: scaffold (OPEN R-093).
    """
    raise NotImplementedError("film_scalings: scaffold, OPEN R-093")
