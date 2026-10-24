"""SlipPY static (dry, normal) contact on a surface pair, with measurements and checks.

:func:`run` takes the contact and its grid as ``lubrication.ehl.run`` does (the surfaces built
on the grid, or given and checked), keeps the argument names of ``slippy.contact.StaticStep``
(``normal_load``, ``method``, ``tolerance`` and any other keyword), builds the ``ContactModel``
and solves it, then returns one
record (units in the keys) with the identification of the contact, the measured values, the
Hertz reference and boolean checks:

* ``converged`` - SlipPY's own flag;
* ``load_ok`` - the pressure integrates to the applied load (``rtol 1e-3``);
* ``edge_clear`` - the pressure is zero on the boundary of the domain (necessary, not sufficient);
* ``margin_ok`` - the domain extends at least ``span_x_min * s_x`` on each side of the origin
  along rolling. Empirical (2026-10-10, DGBB inner contact): p0 vs Hertz was -3.2 % at +-2,
  -0.5 % at +-4, +0.05 % at +-8, with zero pressure on the boundary in every case. The cause is
  the periodic image loads of SlipPY's default frequency-domain kernel
  (``Elastic(use_frequency_domain=True)``), not a property of the elastic half-space. A study
  still checks domain convergence;
* ``resolution_ok`` - at least ``min_nodes`` nodes across the smaller Hertz semi-axis;
* ``within_reference`` - p0, s_x, s_y (and the approach for a point contact) within
  ``rtol_reference`` of Hertz (only meaningful for smooth surfaces).

The approach ``delta_m`` is recorded for a point contact only: in a line contact the approach of
two bodies is not defined by the contact alone, it needs a datum inside each body (Johnson1985
§4.2(c), Eq. (4.37)).
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from .base import Contact, ContactCase
from .measures import measure, rel_err
from .reference import hertz
from .surfaces import SPAN_X_0, Grid, _case_and_surfaces

__all__ = ["run"]

_ids = itertools.count()


def run(contact: Contact | ContactCase, grid: Grid, output_dir, *, surfaces=None,
        build: str = "composite", rough=(None, None), use_frequency_domain: bool = True,
        normal_load: float | None = None, method: str = "pk", tolerance: float = 1e-9,
        span_x_min: float = SPAN_X_0, min_nodes: float = 8.0, rtol_reference: float = 0.03,
        name: str | None = None, **step_kwargs):
    """Solve one static contact and measure it.

    Parameters
    ----------
    contact : Contact or ContactCase
    grid : Grid
        The grid of the solve (``surfaces.Grid.around``): its rolling coordinate ``X`` [m] and
        spacing ``h`` [m] are used.
    output_dir : path
        Where SlipPY writes its .sar/.sdb/.log files (required: SlipPY otherwise writes to cwd).
    surfaces : (surface_1, surface_2), optional
        SlipPY surfaces on ``grid``, checked against it; recorded as ``build = "given"``. None:
        built here by ``surfaces.pair(case, grid, build, rough, use_frequency_domain=...)``.
    build, rough, use_frequency_domain
        Arguments of ``surfaces.pair`` when ``surfaces`` is None; refused with ``surfaces``.
    normal_load : float, optional
        Total load [N]; default from ``case.load`` (see ``Grid.step_args``).
    method, tolerance, **step_kwargs
        Passed to ``slippy.contact.StaticStep``, with the periodic set-up of a line contact
        (``Grid.step_args``).
    span_x_min : float
        Minimum domain half-length on each side of the origin along rolling, in units of the
        Hertz ``s_x`` [-]; limit of the check ``margin_ok``. Same quantity as ``span_x`` of
        ``Grid.around``.
    min_nodes : float
        Minimum number of nodes across the smaller Hertz semi-axis [-]; limit of the check
        ``resolution_ok``.
    rtol_reference : float
        Relative tolerance of p0, s_x, s_y and the approach against Hertz [-]; limit of the check
        ``within_reference``.
    name : str, optional
        Name of the SlipPY ``ContactModel`` (and of its output files). Default: the case label
        and a counter.

    Returns
    -------
    record : dict
        Identification, ``build`` (or ``"given"``), ``use_frequency_domain``, ``method``,
        ``tolerance``, grid (``h_m``, ``n_x``, ``n_y``, ``length_m`` = periodic length
        ``Grid.length_y_m`` [m] of a line contact, nan for a point), measured values
        (``delta_m`` nan for a line), the relative radii of the case ``R_x_m``, ``R_y_m`` [m]
        (geometry, ``1/R = 1/r1 + 1/r2`` per bridge axis), Hertz reference, relative errors, checks,
        ``solve_time_s``.
    pressure : ndarray
        Pressure field [Pa] (for plots; not stored in the record).

    Raises
    ------
    ValueError
        From ``surfaces._case_and_surfaces`` (given surfaces not on ``grid``, or given with
        ``build``/``rough``/``use_frequency_domain``) and ``Grid.step_args`` (a line contact
        made non-periodic).
    """
    import slippy.contact as c
    case, (s1, s2), surface_keys = _case_and_surfaces(contact, grid, surfaces, build, rough,
                                                      use_frequency_domain)
    ident = contact.record() if isinstance(contact, Contact) else {"label": case.label,
                                                                   "case_hash": case.case_hash(),
                                                                   "line": case.line}
    X, h = grid.X, grid.h
    args = grid.step_args(case, normal_load, **step_kwargs)
    load = args["normal_load"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = c.ContactModel(name or f"{case.label}_{next(_ids)}", s1, s2, output_dir=str(out))
    model.add_step(c.StaticStep("static", method=method, tolerance=tolerance, **args))
    t0 = time.perf_counter()
    sol = model.solve()
    dt = time.perf_counter() - t0

    p = np.asarray(sol["loads_z"], dtype=float)          # SlipPY a4fbb44: a pressure [Pa]
    m = measure(p, h, line=case.line)
    ref = hertz(case)
    delta = math.nan if case.line else float(sol.get("interference", math.nan))
    edge = np.concatenate([p[0, :], p[-1, :]] + ([] if case.line else [p[:, 0], p[:, -1]]))
    small = min(ref["s_x_m"], ref["s_y_m"])
    err = {"err_p0": rel_err(m["p0_Pa"], ref["p0_Pa"]), "err_s_x": rel_err(m["s_x_m"], ref["s_x_m"]),
           "err_s_y": rel_err(m["s_y_m"], ref["s_y_m"]),
           "err_delta": rel_err(delta, ref["delta_m"]), "err_load": rel_err(m["load_N"], load)}
    refs = {"err_p0": ref["p0_Pa"], "err_s_x": ref["s_x_m"], "err_s_y": ref["s_y_m"],
            "err_delta": ref["delta_m"]}            # a missing measurement fails a finite reference
    checks = {
        "converged": bool(sol.get("converged", False)),
        "load_ok": abs(err["err_load"]) <= 1e-3,
        "edge_clear": bool(edge.max(initial=0.0) <= 1e-6 * m["p0_Pa"]),
        "margin_ok": bool(min(-X.min(), X.max()) >= span_x_min * ref["s_x_m"]),
        "resolution_ok": bool(small / h >= min_nodes),
        "within_reference": all(math.isfinite(err[k]) and abs(err[k]) <= rtol_reference
                                for k, r in refs.items() if math.isfinite(r)),
    }
    record = {**ident, **surface_keys, "method": method, "tolerance": tolerance,
              "h_m": h, "n_x": X.shape[0], "n_y": X.shape[1],
              "length_m": grid.length_y_m if case.line else math.nan, "load_N_applied": load,
              **m, "delta_m": delta,
              "p0_ref_Pa": ref["p0_Pa"], "s_x_ref_m": ref["s_x_m"], "s_y_ref_m": ref["s_y_m"],
              "delta_ref_m": ref["delta_m"], "R_x_m": ref["R_x_m"], "R_y_m": ref["R_y_m"],
              **err, **checks, "solve_time_s": dt}
    return record, p
