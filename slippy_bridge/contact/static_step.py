"""SlipPY static (dry, normal) contact on a surface pair, with measurements and checks.

:func:`run` keeps the argument names of ``slippy.contact.StaticStep`` (``normal_load``, ``method``,
``tolerance`` and any other keyword), builds the ``ContactModel`` and solves it, then returns one
record (units in the keys) with the identification of the contact, the measured values, the
Hertz reference and boolean checks:

* ``converged`` - SlipPY's own flag;
* ``load_ok`` - the pressure integrates to the applied load (``rtol 1e-3``);
* ``edge_clear`` - the pressure is zero on the boundary of the domain (necessary, not sufficient);
* ``margin_ok`` - the domain spans at least ``margin_x`` Hertz semi-axes along rolling. Empirical
  (2026-10-10, DGBB inner contact): p0 vs Hertz was -3.2 % at +-2, -0.5 % at +-4, +0.05 % at +-8,
  with zero pressure on the boundary in every case. A study still checks domain convergence;
* ``resolution_ok`` - at least ``min_nodes`` nodes across the smaller Hertz semi-axis;
* ``within_reference`` - p0, s_x, s_y (and the approach for a point contact) within
  ``rtol_reference`` of Hertz (only meaningful for smooth surfaces).
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from .base import Contact, ContactCase
from .outputs import measure, rel_err
from .reference import hertz

__all__ = ["run"]

_ids = itertools.count()


def run(contact: Contact | ContactCase, pair, X: np.ndarray, h: float, output_dir, *,
        normal_load: float | None = None, method: str = "pk", tolerance: float = 1e-9,
        margin_x: float = 4.0, min_nodes: float = 8.0, rtol_reference: float = 0.03,
        name: str | None = None, **step_kwargs):
    """Solve one static contact and measure it.

    Parameters
    ----------
    contact : Contact or ContactCase
    pair : (surface_1, surface_2)
        From ``smooth`` (optionally through ``rough.add_roughness``).
    X : ndarray
        Rolling coordinate of the grid (from ``smooth.grid``) [m].
    h : float
        Grid spacing [m].
    output_dir : path
        Where SlipPY writes its .sar/.sdb/.log files (required: SlipPY otherwise writes to cwd).
    normal_load : float, optional
        Total load [N]. Default: ``case.load`` for a point contact; ``case.load`` times the length
        of the periodic domain for a line contact.
    method, tolerance, **step_kwargs
        Passed to ``slippy.contact.StaticStep``. A line contact gets
        ``periodic_geometry=True, periodic_axes=(False, True)`` unless given.

    Returns
    -------
    record : dict
        Identification, measured values, reference, relative errors, checks, ``solve_time_s``.
    pressure : ndarray
        Pressure field [Pa] (for plots; not stored in the record).
    """
    import slippy.contact as c
    case = contact.case if isinstance(contact, Contact) else contact
    ident = contact.record() if isinstance(contact, Contact) else {"label": case.label,
                                                                   "case_hash": case.case_hash(),
                                                                   "line": case.line}
    s1, s2 = pair
    if case.line:
        step_kwargs.setdefault("periodic_geometry", True)
        step_kwargs.setdefault("periodic_axes", (False, True))
        load = case.load * X.shape[1] * h if normal_load is None else normal_load
    else:
        load = case.load if normal_load is None else normal_load

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = c.ContactModel(name or f"{case.label}_{next(_ids)}", s1, s2, output_dir=str(out))
    model.add_step(c.StaticStep("static", normal_load=load, method=method, tolerance=tolerance,
                                **step_kwargs))
    t0 = time.perf_counter()
    sol = model.solve()
    dt = time.perf_counter() - t0

    p = np.asarray(sol["loads_z"], dtype=float)          # SlipPY a4fbb44: a pressure [Pa]
    m = measure(p, h, line=case.line)
    ref = hertz(case)
    delta = float(sol.get("interference", math.nan))
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
        "margin_ok": bool(min(-X.min(), X.max()) >= margin_x * ref["s_x_m"]),
        "resolution_ok": bool(small / h >= min_nodes),
        "within_reference": all(math.isfinite(err[k]) and abs(err[k]) <= rtol_reference
                                for k, r in refs.items() if math.isfinite(r)),
    }
    record = {**ident, "h_m": h, "n_x": X.shape[0], "n_y": X.shape[1], "load_N_applied": load,
              **m, "delta_m": delta,
              "p0_ref_Pa": ref["p0_Pa"], "s_x_ref_m": ref["s_x_m"], "s_y_ref_m": ref["s_y_m"],
              "delta_ref_m": ref["delta_m"], **err, **checks, "solve_time_s": dt}
    return record, p
