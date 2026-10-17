"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

One steady, isothermal, Newtonian EHL solve through SlipPY, on the surfaces of ``slippy_bridge.contact``.

Elastohydrodynamic lubrication (EHL): the oil dragged into a loaded contact by the surface speeds
builds a pressure high enough to deform the bodies elastically, and its viscosity rises steeply with
that pressure; the film thickness ``h`` that results is a few tenths of a micrometre. SlipPY solves the
Reynolds equation for the film coupled with the elastic deflection of both bodies
(``UnifiedReynoldsSolver`` inside an ``IterSemiSystem`` step, with ``no_time=True``); the template
of the call is SlipPY ``slippy/contact/tests/test_mixed_lubrication.py:9-74``.

Conventions fixed here (as ``smooth.py`` fixes the dry ones):

* SI units everywhere; the unit is in every keyword and record key.
* Entrainment speed: the MEAN speed ``(u1 + u2)/2`` (SlipPY ``rolling_speed``, R-075), along
  +axis 0, the bridge rolling direction ``x`` (R-073). Published conventions (sum speeds, Moes M
  and L, H_venner) live only in ``references.contact.ehl``; this module records no dimensionless H.
* Load control only (``normal_load``): SlipPY's EHL step ignores a prescribed interference (R-081).
"""
from __future__ import annotations

import math

from ..contact.base import Contact, ContactCase
from ..contact.smooth import Grid

__all__ = ["run", "FILM_KEYS"]

# Roelands writes ln(eta/eta0) = (ln(eta0) + 9.67) [(1 + p/p0)^z - 1] (SlipPY lubricant_models.py:103):
# the law is a viscosity RISE only if ln(eta0) + 9.67 > 0, i.e. eta0 > exp(-9.67) Pa s. The upper bound
# 10 Pa s is above any gear or bearing oil at its working temperature and catches an eta0 given in mPa s.
_ETA0_RANGE_PA_S = (math.exp(-9.67), 10.0)
# Roelands pressure-viscosity index z [-]: 0 makes the law constant (exclusive); the bound 1.5 is above the
# usual oil values (GL-17; source to find, Roelands 1966).
_ROELANDS_Z_RANGE = (0.0, 1.5)
# Roelands reference pressure p0 [Pa]: below 1e6 it was given in MPa.
_ROELANDS_P0_MIN_PA = 1e6  # OPEN GL-18: the plausible band [1e8, 1e9] Pa; no source prints p0 for the cases
# Barus pressure-viscosity coefficient alpha [1/Pa]: oils are of order 1e-8;
# a value in 1/MPa or 1/GPa falls outside this band by three decades.
_BARUS_ALPHA_RANGE_PER_PA = (1e-9, 1e-7)
_DENSITY_MODELS = ("dowson_higginson", "incompressible")

# The EHL-only keys of the record; the record is the kept static_step.run keys (see run, Notes 6) + these.
# OPEN GL-20: which static keys are carried
# Viscosity fields of the model not used are nan. roelands_p0_Pa is the Roelands constant, not the
# measured maximum pressure p0_Pa nor the Hertz p0_ref_Pa (GL-07).
FILM_KEYS = ("u_mean_m_per_s", "w_N_per_m", "eta0_Pa_s", "rho0_kg_per_m3", "viscosity_model", "roelands_z",
             "roelands_p0_Pa", "barus_alpha_per_Pa", "alpha_initial_per_Pa", "density_model", "R_x_m",
             "rtol_pressure", "h_c_m", "h_min_m", "x_h_min_m", "y_h_min_m")


def run(contact: Contact | ContactCase, grid: Grid, output_dir, *, surfaces=None, eta0_Pa_s: float,
        rho0_kg_per_m3: float, viscosity: tuple, density: str = "dowson_higginson",
        u_mean_m_per_s: float | None = None, normal_load: float | None = None, name: str | None = None,
        **step_kwargs):
    """Solve one steady EHL contact with SlipPY and measure its film.

    Physically: two smooth (or rough) elastic bodies of the case, lubricated by a Newtonian oil at
    one uniform temperature, rolling at the mean speed ``u_mean_m_per_s`` along +x, pressed by a
    constant load. The result is the steady pressure field and the film thickness (gap) between the
    deformed surfaces: the central film ``h_c`` and the minimum film ``h_min`` in the outlet
    constriction.

    Parameters
    ----------
    contact : Contact or ContactCase
        The contact, as ``static_step.run`` accepts it; a ``Contact`` may carry
        ``kinematics["u_e_m_per_s"]``. A bare ``ContactCase`` has no kinematics, so it needs
        ``u_mean_m_per_s``. SI units (``ContactCase``).
    grid : Grid
        Grid of the solve (``smooth.Grid.around``): spacing ``h`` [m], ``X`` rolling along axis 0.
    output_dir : path
        Where SlipPY writes its files (required: SlipPY otherwise writes to its global OUTPUT_DIR or
        the cwd, and a repeated model name overwrites, R-088).
    surfaces : (surface_1, surface_2), optional
        SlipPY surfaces on ``grid`` (``smooth.pair``, smooth or rough). None: the smooth composite pair
        is built from ``contact.case`` and ``grid``, as ``smooth.pair(case, grid)`` (GL-21).
    eta0_Pa_s : float
        Viscosity of the oil at ambient pressure and the working temperature [Pa s].
    rho0_kg_per_m3 : float
        Density of the oil at ambient pressure [kg/m^3].
    viscosity : tuple
        Pressure-viscosity law: ``("roelands", z, p0_Pa)`` (index z [-], reference pressure p0 [Pa];
        never an alpha in place of p0, R-079) or ``("barus", alpha_per_Pa)``, ``eta = eta0 exp(alpha p)``.
    density : {"dowson_higginson", "incompressible"}
        Pressure-density law: Dowson-Higginson, ``rho/rho0 = (5.9e8 + 1.34 p)/(5.9e8 + p)`` with p in Pa
        (SlipPY ``nd_dowson_higginson``), or constant ``rho0``.
    u_mean_m_per_s : float, optional
        Mean entrainment speed (u1 + u2)/2 [m/s], > 0. None: ``contact.kinematics["u_e_m_per_s"]``.
    normal_load : float, optional
        TOTAL load on the domain [N]. None: ``case.load`` for a point contact; for a line contact the
        load per unit length times the periodic length, ``w_N_per_m * n_y * h`` (as ``static_step.run``).
    name : str, optional
        Name of the SlipPY ``ContactModel``. Default: the case label and a counter (unique, R-088).
    **step_kwargs
        Passed to ``slippy.contact.IterSemiSystem`` (e.g. ``rtol_pressure``, ``relaxation_factor``).

    Returns
    -------
    record : dict
        ``Contact.record()`` (or the case identification) + the kept ``static_step.run`` keys (Notes 6)
        + ``FILM_KEYS``.
    p_Pa : ndarray
        Pressure field [Pa].
    gap_m : ndarray
        Film thickness field [m].

    Raises
    ------
    ValueError
        In this order: ``output_dir`` None (R-088); ``viscosity`` not a known law tuple (R-079); unknown
        ``density`` (R-078); eta0, z, p0 or alpha outside the ranges of this module (unit slips, R-077);
        no mean speed, or not > 0 (R-073, GL-19); ``interference`` given (load control only, R-081); a
        finite ``max_load`` on a surface material (it caps the pressure, R-082; can fire only with
        ``surfaces`` given: the smooth pair built here keeps the default inf); a transient request,
        ``time_step != 0``, ``number_of_steps != 1`` or ``no_time=False`` (steady only, R-083, R-087);
        a line contact with ``periodic_geometry`` or ``periodic_axes`` other than ``True``,
        ``(False, True)`` (R-084);
        with ``surfaces`` given, a surface whose ``grid_spacing`` is not ``grid.h`` (R-085).
    NotImplementedError
        Always, after the checks: the solve is a scaffold.

    Notes
    -----
    Planned body (OPEN R-076, R-086):

    1. ``p0_ref_Pa``, ``s_x_ref_m`` and ``R_x_m`` come ONCE from ``contact.reference.hertz(case)`` and
       feed the nd laws, the solver and the step: SlipPY takes them in three places and never compares
       them (R-076). The laws are called by keyword: ``nd_roelands(eta_0=, pressure_0=,
       pressure_hertzian=, z=)`` or ``nd_barus(pressure_hertzian=, alpha=)``, and
       ``nd_dowson_higginson(pressure_hertzian=)``; incompressible is ``Lubricant(name,
       constant_density=rho0)`` (SlipPY lubricant.py:61-63, R-078).
    2. A NEW ``UnifiedReynoldsSolver`` on every call (it caches the row order and keeps class-level
       lists, R-080), with ``time_step=0`` set by hand (``IterSemiSystem`` does not pass it, R-087),
       ``grid_spacing=grid.h`` and ``periodic=True`` for a line (R-084). The step gets ``no_time=True``
       explicitly: SlipPY's default is ``no_time=False`` (lubrication_steps.py:197).
    3. Initial guess: the Hertz pressure on the bridge coordinates, centred on the node of ``X == 0``,
       and the dimensional Hertz approach (R-086).
    4. ``alpha_initial_per_Pa`` is the slope of ln(eta) at p = 0: ``alpha`` for Barus,
       ``(ln eta0 + 9.67) z / p0`` for Roelands (R-091).
    5. ``load_ok = abs(err_load) <= rtol_pressure``, the tolerance SlipPY's step itself uses
       (default 2e-6, lubrication_steps.py:518-519), recorded (GL-08); the film keys come from
       ``outputs.measure(p, h, line, gap=, X=, Y=)`` (R-033).
    6. The record (OPEN GL-20): the identification, ``h_m``, ``n_x``, ``n_y``, ``load_N_applied``, the
       ``measure`` keys, ``p0_ref_Pa``, ``s_x_ref_m``, ``err_load``, ``converged``, ``load_ok``,
       ``solve_time_s`` of the ``static_step.run`` record, then ``FILM_KEYS``. Not carried, because their
       meaning is dry Hertz: ``within_reference``, ``err_p0``, ``err_s_x``, ``err_delta``, ``margin_ok``.

    Limits (R-089, R-090): isothermal, Newtonian, cavitation by clipping p >= 0, no asperity contact
    model and no traction. For a line contact the elastic kernel is a finite periodic strip
    (GL-16). Only feasibility-tested in SlipPY (R-115): the ``venner_bos_circular`` benchmark gates
    every other EHL result.

    References
    ----------
    VennerBos1994 (dimensionless film and benchmarks, pp. 14-15).
    Roelands (1966) (viscosity law; source to get). Dowson & Higginson (1966) (density law).
    SlipPY ``slippy/contact/tests/test_mixed_lubrication.py:9-74``.
    """
    # OPEN R-073: u_mean is placed along +axis 0; to test on an ellipse.
    # OPEN R-076: p_h, s_x and R_x from one reference.hertz(case) into laws, solver and step.
    # OPEN R-086: Hertz initial guess on bridge coordinates, centred, dimensional approach.
    # OPEN R-115: EHL is feasibility-tested only in SlipPY; venner_bos_circular gates the results.
    # OPEN R-089 R-090: isothermal, Newtonian, clipped cavitation, no traction.
    # OPEN GL-07 GL-08 GL-16 GL-20: key names, load_ok tolerance, finite-strip kernel, record layout.
    case = contact.case if isinstance(contact, Contact) else contact
    # 1. SlipPY writes .sar/.sdb/.log; never let it fall back to its global directory (R-088)
    if output_dir is None:
        raise ValueError("output_dir is required (SlipPY otherwise writes to its global OUTPUT_DIR; R-088)")
    # 2. the law by name and its constants by position, so that an alpha never lands in p0 (R-079)
    if not (isinstance(viscosity, (tuple, list)) and len(viscosity) >= 1
            and ((viscosity[0] == "roelands" and len(viscosity) == 3)
                 or (viscosity[0] == "barus" and len(viscosity) == 2))):
        raise ValueError(f"viscosity must be ('roelands', z, p0_Pa) or ('barus', alpha_per_Pa), got {viscosity!r}")
    # 3.
    if density not in _DENSITY_MODELS:
        raise ValueError(f"density must be one of {_DENSITY_MODELS}, got {density!r}")
    # 4. SlipPY's laws hold SI literals (9.67, 5.9e8 Pa): a value in mPa s or MPa silently gives a wrong law (R-077)
    if not _ETA0_RANGE_PA_S[0] < eta0_Pa_s <= _ETA0_RANGE_PA_S[1]:
        raise ValueError(f"eta0_Pa_s = {eta0_Pa_s} is outside ({_ETA0_RANGE_PA_S[0]:.3g}, "
                         f"{_ETA0_RANGE_PA_S[1]}] Pa s (a value in mPa s?)")
    if viscosity[0] == "roelands":
        _, z, p0_Pa = viscosity
        if not _ROELANDS_Z_RANGE[0] < z <= _ROELANDS_Z_RANGE[1]:
            raise ValueError(f"Roelands z = {z} is outside ({_ROELANDS_Z_RANGE[0]}, {_ROELANDS_Z_RANGE[1]}]")
        if not p0_Pa >= _ROELANDS_P0_MIN_PA:
            raise ValueError(f"Roelands p0 = {p0_Pa} Pa is below {_ROELANDS_P0_MIN_PA:.0e} Pa (a value in MPa?)")
    else:
        alpha = viscosity[1]
        if not _BARUS_ALPHA_RANGE_PER_PA[0] <= alpha <= _BARUS_ALPHA_RANGE_PER_PA[1]:
            raise ValueError(f"Barus alpha = {alpha} 1/Pa is outside {_BARUS_ALPHA_RANGE_PER_PA} (a value in 1/MPa?)")
    # 5. the keyword first, then the kinematics of the contact; checked > 0 only after the fallback (GL-19)
    u_mean = u_mean_m_per_s
    if u_mean is None and isinstance(contact, Contact):
        u_mean = contact.kinematics.get("u_e_m_per_s")
    if u_mean is None:
        raise ValueError("no mean entrainment speed: give u_mean_m_per_s or a Contact with kinematics['u_e_m_per_s']")
    if not u_mean > 0:
        raise ValueError(f"the mean entrainment speed must be > 0 m/s (along +x), got {u_mean}")
    # 6. IterSemiSystem ignores a prescribed interference: load control only (R-081)
    if "interference" in step_kwargs:
        raise ValueError("interference is not accepted: SlipPY's EHL step is load-controlled only (R-081)")
    # 7. a finite max_load crops the pressure and changes the solver (R-082)
    if surfaces is not None:
        for k, s in enumerate(surfaces):
            max_load = getattr(getattr(s, "material", None), "max_load", math.inf)
            if math.isfinite(max_load):
                raise ValueError(f"surfaces[{k}] material has max_load = {max_load}; EHL needs inf (R-082)")
    # 8. steady only: SlipPY resets the previous gap every sub-step and never passes its time step (R-083, R-087)
    # time_step belongs to the URS, number_of_steps and no_time to IterSemiSystem (LS:194-211)
    if (step_kwargs.get("time_step", 0) != 0 or step_kwargs.get("number_of_steps", 1) != 1
            or not step_kwargs.get("no_time", True)):
        raise ValueError("only steady EHL is supported: time_step 0, number_of_steps 1, no_time True (R-083, R-087)")
    # 9. a line contact is periodic along the roller axis (axis 1) only (R-084)
    if case.line and (step_kwargs.get("periodic_geometry", True) is not True
                      or tuple(step_kwargs.get("periodic_axes", (False, True))) != (False, True)):
        raise ValueError("a line contact needs periodic_geometry=True, periodic_axes=(False, True) (R-084)")
    # 10. the solver takes grid.h; surfaces sampled at another spacing would be solved without warning (R-085)
    if surfaces is not None:
        for k, s in enumerate(surfaces):
            if s.grid_spacing != grid.h:
                raise ValueError(f"surfaces[{k}].grid_spacing = {s.grid_spacing} is not grid.h = {grid.h} (R-085)")
    raise NotImplementedError("ehl.run: scaffold, OPEN R-076 R-086 R-115 R-033 R-048")
