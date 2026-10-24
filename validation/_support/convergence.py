"""
validation/_support/convergence.py -- convergence helpers shared by the
validation cases.

* ``ResultantView``: adapter for AxisForge's mesh convergence machinery
  (``MeshConvergenceStudy``, ``SubmodelSolver``). Extracted on 2026-10-06 from
  the two ``02_verification_*`` notebooks.
* ``sweep`` and ``RECORD_COLUMNS``: mesh and domain sweep of a smooth SlipPY
  static contact through ``slippy_bridge.contact`` (SI units). Moved on
  2026-10-16 from the study ``hertz_vs_bem`` (``_common/sweep.py``) for the
  three ``validation/contact`` cases.

Only what the validation cases repeated verbatim lives here. The grade loop
itself stays in each notebook on purpose: it is the documented procedure of
that study (which quantities, which thresholds, when to stop), not plumbing.
Promote a loop here when two validation cases need the same one.
"""
from __future__ import annotations

import numpy as np

from slippy_bridge.contact import Grid, static_step
from slippy_bridge.contact.surfaces import SPAN_X_0

__all__ = ["ResultantView", "sweep", "RECORD_COLUMNS"]


class ResultantView:
    """Read-only view of a ``SubmodelResult`` adding the resultant fields.

    ``SubmodelResult`` (AxisForge) stores the two bending planes separately;
    ``MeshConvergenceStudy`` evaluates named fields. This view exposes

    * ``v_res = hypot(v_xz, v_xy)`` -- resultant deflection [mm];
    * ``M_res = hypot(M_xz, M_xy)`` -- resultant bending moment [N.mm];

    and forwards every other attribute to the wrapped result. Nothing in
    AxisForge is modified.

    Parameters
    ----------
    result : SubmodelResult
        As returned by ``SubmodelSolver.solve``.

    Notes
    -----
    The resultant is computed node by node; its maximum is not the maximum
    of either plane, and it is non-smooth where one plane changes sign.
    """

    def __init__(self, result):
        self._r = result
        self.x_nodes = result.x_nodes
        self.v_res = np.hypot(result.v_xz, result.v_xy)
        self.M_res = np.hypot(result.M_xz, result.M_xy)

    def __getattr__(self, name):
        return getattr(self._r, name)


# Console tables (``axisforge.outputs._format.format_table``): key, header, format, scale.
RECORD_COLUMNS = (("build", "build", "s"), ("sweep", "sweep", "s"), ("span_x", "span_x [s_x]", "g"),
                  ("h_m", "h [um]", ".3f", 1e6), ("n_x", "n_x", ".0f"), ("n_y", "n_y", ".0f"),
                  ("p0_Pa", "p0 [MPa]", ".2f", 1e-6), ("err_p0", "err p0 [%]", "+.3f", 100.0),
                  ("err_s_x", "err s_x [%]", "+.3f", 100.0), ("err_s_y", "err s_y [%]", "+.3f", 100.0),
                  ("err_delta", "err delta [%]", "+.3f", 100.0), ("solve_time_s", "t [s]", ".2f"))


def sweep(contact, builds, h_levels, span_x_levels, output_dir, *, span_x_0=SPAN_X_0,
          use_frequency_domain=True) -> list[dict]:
    """Solve one smooth static contact on a series of grids, for each build.

    Two sweeps around the default grid of the bridge (spacing ``h0`` = largest ``2**k`` not above
    ``s_x / 10``, half-length ``span_x_0 * s_x`` along rolling):

    * ``"mesh"``: spacing ``f * h0`` for each ``f`` in ``h_levels``, half-length ``span_x_0 * s_x``.
      Shows the discretisation error.
    * ``"domain"``: half-length ``span_x * s_x`` for each ``span_x`` in ``span_x_levels``, spacing ``h0``.
      Shows the dependence on the domain. A half-space BEM should have none once the contact is
      inside; SlipPY's default frequency-domain kernel repeats the load with period twice the
      domain, and this sweep measures that image error.

    Parameters
    ----------
    contact : Contact or ContactCase
        The contact (SI units, ``x`` = rolling direction).
    builds : sequence of {"composite", "two_profiles"}
        Surface builds of ``slippy_bridge.contact.pair``.
    h_levels : sequence of float
        Grid spacings in units of ``h0`` [-]; powers of two keep the spacing dyadic.
    span_x_levels : sequence of float
        Half-lengths of the domain along rolling in units of the Hertz semi-axis ``s_x`` [-].
    output_dir : path
        Folder for SlipPY's .sar/.sdb/.log files.
    span_x_0 : float, optional
        Half-length of the mesh-sweep grids along rolling in units of ``s_x`` [-]; the default 4.0
        is the default grid of ``Grid.around``.
    use_frequency_domain : bool, optional
        Elastic kernel of both bodies, passed to ``slippy_bridge.contact.pair``: True is SlipPY's
        frequency-domain kernel (periodic images of the load), False the spatial-domain kernel
        (no images on a non-periodic axis). Not available for a line contact:
        ``slippy_bridge.contact.surfaces.pair`` raises ``ValueError``.

    Returns
    -------
    list of dict
        One record of ``static_step.run`` per solve (measurements, Hertz reference, relative
        errors as fractions, checks, ``build``, ``use_frequency_domain``), plus ``sweep``
        (``"mesh"`` or ``"domain"``) and ``span_x`` [-]. The default grid appears once in each sweep.
    """
    case = getattr(contact, "case", contact)
    h0 = Grid.around(case).h
    grids = [("mesh", f * h0, span_x_0) for f in h_levels] + [("domain", h0, span_x) for span_x in span_x_levels]
    records = []
    for build in builds:
        for i, (kind, h, span_x) in enumerate(grids):
            grid = Grid.around(case, h=h, span_x=span_x)
            record, _ = static_step.run(contact, grid, output_dir, build=build,
                                        use_frequency_domain=use_frequency_domain,
                                        name=f"{case.label}_{build}_{kind}_{i}")
            records.append({"sweep": kind, "span_x": span_x, **record})
    return records
