"""Result of one static (dry, normal) BEM solve of a contact: data only (filled by ``solvers.solve``).

One flat dataclass with its fields grouped as D-0034 item 7 lists them (contact, grid, profile,
load, solver, fields, measured, Hertz reference, errors, checks, provenance), and
:meth:`StaticResult.record`, one flat dict of every scalar for a CSV row (written through
``axisforge.outputs._format``, D-0008; a later ``outputs/`` may write the CSV).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.contact import DC
from .hertz import HertzResult

if TYPE_CHECKING:
    from ..core.contact import Contact
    from ..mesh.grid import Grid

__all__ = ["StaticResult"]

# fields of a Contact that record() writes in its own groups, not as identification
_CONTACT_GROUPED = {"body1", "body2", "load", "angle", "label", "kappa_max", "kappa_min", "psi", "E_star"}


@dataclass(**DC)
class StaticResult:
    """Static BEM solution of one contact (SlipPY ``StaticStep``) with its Hertz reference and checks.

    Parameters
    ----------
    contact : Contact
        The contact solved.
    grid : Grid
        The grid of the solve (bridge axes, rolling = array axis 0).
    build : str
        How the surfaces were built: ``"two_profiles"``, each body its exact profile.
    z1, z2 : ndarray
        Profiles of body 1 and body 2 on the grid [m] (SlipPY sign: ``-sag``, apex 0).
    gap0 : ndarray
        Initial gap of the undeformed surfaces [m]: SlipPY ``just_touching_gap``, ``-(z1 + z2)``
        shifted to a minimum of 0, so >= 0.
    load_applied : float
        Load given to the step [N] (a line: load per unit length x period).
    period : float
        Period of a line contact along bridge y [m]; ``nan`` for a point contact.
    use_frequency_domain : bool
        How the SlipPY influence matrix is defined (``Elastic(use_frequency_domain=...)``): True in
        the frequency domain, False in space. Periodicity comes from ``periodic_axes`` of the step.
    method : str
        SlipPY solution method of the step.
    tolerance : float
        Convergence tolerance given to the step [-].
    converged : bool
        SlipPY's convergence flag of the step.
    solve_time : float
        Wall time of the step [s].
    p : ndarray
        Contact pressure on the grid [Pa] (SlipPY ``loads_z``).
    total_displacement_z, surface_1_displacement_z, surface_2_displacement_z : ndarray or None
        Normal surface displacements [m], SlipPY solution keys of the same names (always returned
        with ``Elastic`` materials; None only if a key is absent).
    delta : float
        Approach of the two bodies [m]: SlipPY ``interference``, the mean of
        ``just_touching_gap + total_displacement_z`` over the loaded nodes. Its absolute level
        depends on the kernel's zero-frequency term and on the domain, so ``err_delta`` is
        meaningful for a point contact on a large enough domain; ``nan`` for a line (not defined
        locally).
    p0 : float
        Measured maximum pressure [Pa].
    a, b : float
        Measured semi-axes of the loaded region [m]: ``b`` along ``psi``, ``a`` across (``inf``
        for a line); the edge is extrapolated below one node.
    load : float
        Measured load [N]: SlipPY ``total_normal_load``, ``sum p h^2`` over the grid.
    apex_offset_nodes : float
        Distance of the pressure maximum from the grid centre node [nodes].
    hertz : HertzResult
        Hertz reference of the contact.
    err_p0, err_a, err_b, err_delta, err_load : float
        Relative errors ``(measured - reference)/reference`` [-] against Hertz (``err_load``
        against ``load_applied``); ``nan`` when the reference is not finite or zero.
    load_ok, edge_clear, margin_ok, resolution_ok, within_reference : bool
        Checks of the solve: load balance; no pressure on the domain edge; domain margin around
        the contact; nodes across the contact; errors within the reference tolerance. With
        ``converged`` and :attr:`hertz_valid` they are the checks of the record.
    commits : dict
        Provenance: ``{"axisforge", "slippy", "design_studies"} -> commit hash`` or ``"unknown"``.
    """

    # contact
    contact: Contact
    # grid
    grid: Grid
    # profile
    z1: np.ndarray
    z2: np.ndarray
    gap0: np.ndarray
    build: str = "two_profiles"
    # load
    load_applied: float
    period: float
    # solver
    use_frequency_domain: bool
    method: str
    tolerance: float
    converged: bool
    solve_time: float
    # fields
    p: np.ndarray
    delta: float
    total_displacement_z: np.ndarray | None = None
    surface_1_displacement_z: np.ndarray | None = None
    surface_2_displacement_z: np.ndarray | None = None
    # measured
    p0: float
    a: float
    b: float
    load: float
    apex_offset_nodes: float
    # Hertz reference
    hertz: HertzResult
    # errors
    err_p0: float
    err_a: float
    err_b: float
    err_delta: float
    err_load: float
    # checks
    load_ok: bool
    edge_clear: bool
    margin_ok: bool
    resolution_ok: bool
    within_reference: bool
    # provenance
    commits: dict = field(default_factory=dict)

    @property
    def hertz_valid(self) -> bool:
        """Hertz validity of the reference (``hertz.valid``), a check of the record."""
        return self.hertz.valid

    def record(self) -> dict:
        """One flat dict of every scalar of the result (no arrays), the unit in the key (OPEN SB-10).

        Groups, in order:

        * identification: ``contact`` (class name), ``label``, ``line`` and the contact's own scalar
          fields under their names (bearing: ``race``, ``alpha_rad``, ``j``, ``phi_rad``, ...);
        * geometry: ``r1_x_m``, ``r1_y_m``, ``r2_x_m``, ``r2_y_m`` (signed body radii in the body
          frames), ``E1_Pa``, ``E2_Pa``, ``nu1``, ``nu2``, ``angle_rad``, ``kappa_max_per_m``,
          ``kappa_min_per_m``, ``psi_rad``, ``R_x_m = 1/kappa_max`` and ``R_y_m = 1/kappa_min``
          (principal relative radii along ``psi`` and across it; they are along bridge x and y
          only when ``psi = 0``; ``R_y_m = inf`` for a line), ``E_star_Pa``;
        * grid: ``h_m``, ``n_x``, ``n_y``, ``x_min_m``, ``x_max_m``, ``y_min_m``, ``y_max_m``,
          ``span_x`` and ``span_y`` (the shorter side of the domain in half-extents of the Hertz
          ellipse on the bridge axes, as ``Grid.around``; ``span_y`` nan for a line);
        * profile ``build``; load: ``Q_N`` (the load of a point contact, nan for a line) and
          ``w_N_per_m`` (of a line contact, nan for a point), ``load_applied_N``, ``period_m``;
        * solver: ``use_frequency_domain``, ``method``, ``tolerance``, ``converged``,
          ``solve_time_s``;
        * measured: ``p0_Pa``, ``a_m``, ``b_m``, ``load_N``, ``apex_offset_nodes``, ``delta_m``;
        * Hertz reference: :meth:`HertzResult.record` with ``tag="_ref"`` (``p0_ref_Pa``, ``a_ref_m``,
          ``b_ref_m``, ``delta_ref_m``, ``b_over_a_ref``, ``size_ratio_ref``), without the copies of
          the contact (psi, E_star, line) and ``valid`` (the check ``hertz_valid``);
        * errors ``err_*``, checks, and ``commit_axisforge``, ``commit_slippy``,
          ``commit_design_studies``.
        """
        c, g, hz = self.contact, self.grid, self.hertz
        ident = {"contact": type(c).__name__, "label": c.label, "line": c.line}
        for f in dataclasses.fields(c):
            value = getattr(c, f.name)
            if f.name not in _CONTACT_GROUPED and (value is None or isinstance(value, (bool, int, float, str))):
                ident[f.name] = value
        (r1x, r1y), (r2x, r2y) = c.body1.r, c.body2.r
        geometry = {"r1_x_m": r1x, "r1_y_m": r1y, "r2_x_m": r2x, "r2_y_m": r2y,
                    "E1_Pa": c.body1.E, "E2_Pa": c.body2.E, "nu1": c.body1.nu, "nu2": c.body2.nu,
                    "angle_rad": c.angle, "kappa_max_per_m": c.kappa_max, "kappa_min_per_m": c.kappa_min,
                    "psi_rad": c.psi, "R_x_m": 1.0 / c.kappa_max,
                    "R_y_m": math.inf if c.kappa_min == 0 else 1.0 / c.kappa_min, "E_star_Pa": c.E_star}
        X_half, Y_half = hz.half_extents()                  # as Grid.around sizes the domain
        x, y = g.X[:, 0], g.Y[0, :]
        grid = {"h_m": g.h, "n_x": g.X.shape[0], "n_y": g.X.shape[1],
                "x_min_m": float(x.min()), "x_max_m": float(x.max()),
                "y_min_m": float(y.min()), "y_max_m": float(y.max()),
                "span_x": min(-x.min(), x.max()) / X_half,
                "span_y": math.nan if c.line else min(-y.min(), y.max()) / Y_half}
        ref = {k: v for k, v in hz.record("_ref").items()
               if k not in ("psi_ref_rad", "E_star_ref_Pa", "line_ref", "valid_ref")}
        return {**ident, **geometry, **grid, "build": self.build,
                "Q_N": math.nan if c.line else c.load, "w_N_per_m": c.load if c.line else math.nan,
                "load_applied_N": self.load_applied, "period_m": self.period,
                "use_frequency_domain": self.use_frequency_domain, "method": self.method,
                "tolerance": self.tolerance, "converged": self.converged, "solve_time_s": self.solve_time,
                "p0_Pa": self.p0, "a_m": self.a, "b_m": self.b, "load_N": self.load,
                "apex_offset_nodes": self.apex_offset_nodes, "delta_m": self.delta,
                **ref,
                "err_p0": self.err_p0, "err_a": self.err_a, "err_b": self.err_b, "err_delta": self.err_delta,
                "err_load": self.err_load,
                "load_ok": self.load_ok, "edge_clear": self.edge_clear, "margin_ok": self.margin_ok,
                "resolution_ok": self.resolution_ok, "within_reference": self.within_reference,
                "hertz_valid": self.hertz_valid,
                **{f"commit_{k}": self.commits.get(k, "unknown") for k in ("axisforge", "slippy", "design_studies")}}
