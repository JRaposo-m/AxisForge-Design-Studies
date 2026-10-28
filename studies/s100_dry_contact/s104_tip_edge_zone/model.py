"""Model of s104: integral measures of a BEM line contact next to a flank end, and their grid convergence.

The BEM pressure of a line contact on a periodic strip is uniform along the line (bridge y); its profile across the
line, ``p(x)`` (the mean over y), carries the load per unit length ``w = int p dx``. Next to a sharp flank end the
pressure has an integrable ``r^(-1/2)`` singularity at the edge (a punch corner: Johnson1985 Sec. 2.8, Eq. (2.64),
PDF p. 47; the half-space cannot represent the corner's own compliance, ibid. PDF p. 46), so the peak ``p0`` grows
without bound with the grid (it is recorded, flagged, and never a result) while the integral measures converge:

* centroid offset ``x_c = int p x dx / w``, given over the Hertz half-width ``b`` and signed positive AWAY from the
  edge (the load moves away from the edge): 0 for Hertz;
* equivalent width ``b_eq = 2 sqrt(int p (x - x_c)^2 dx / w)``: the half-width of the semi-ellipse with the same
  second moment, ``b_eq = b`` for Hertz (``int x^2 sqrt(1 - x^2/b^2) dx / int sqrt(1 - x^2/b^2) dx = b^2/4``);
  ``sqrt(2) a`` for a flat punch of half-width ``a`` (Johnson Eq. (2.64), the test of the node sums on a singular field);
* mean pressure ``p_m = w / (2 b_eq)`` over Hertz's ``w / (2 b)``: ``b / b_eq``;
* edge-zone load share: the part of ``w`` within ``b`` of the edge, against the Hertz value of the same window
  (:func:`hertz_measures`, the semi-ellipse as if there were no edge).

Discrete integrals are node sums times the spacing ``h`` (piecewise-constant pressure per node, SlipPY's BEM). The
material edge of that field lies at the last on-flank node + h/2, not at ``x_edge_m``: the edge window and the
``singular`` flag jitter by one node with h (the oscillatory Richardson judgements of the edge share). The grid study
(:func:`grid_series`) halves ``h`` twice with the domain fixed and judges the measures with
``numerical_solvers.richardson`` (``flat_atol = 0``, ``max_p = 2``, the order of the collocation for smooth fields;
a singular field converges slower, which the observed order shows).
"""
from __future__ import annotations

import math

import numpy as np
from slippy_bridge.contact import Grid, hertz, solve

from numerical_solvers import RichardsonOptions, richardson

__all__ = ["measures", "hertz_measures", "grid_series", "span_for", "B_OVER_H", "SPAN_X_MIN", "SPAN_RULE_MARGIN",
           "N_Y", "METRICS", "MAX_P"]

B_OVER_H = (16.0, 32.0, 64.0)   # Hertz half-width over grid spacing asked: h, h/2, h/4 (v104 rule b/h >= 16)
# Domain half-length in half-widths, the same at every level: the R2-08 line rule >= 0.23 M^0.75 at the M ACTUALLY
# solved on the finest grid (M = 2 b/h; h is floored to a power of two, so M is 1-2x the asked one: R5-118), and at
# least SPAN_X_MIN. A shorter domain leaves the frequency kernel's image error 0.0514/span^2 in p0 (v103 README 8).
# At the rule itself that error EQUALS v103's C3 tolerance M^-1.5 and any discretisation error tips C3 over (first
# re-run: E 4.4e-4 vs 4.3e-4); SPAN_RULE_MARGIN 2 makes the image error a quarter of the tolerance.
# OPEN R5-111: the rule was set on smooth contacts; the margin 2 is this case's choice
SPAN_X_MIN = 10.0
SPAN_RULE_MARGIN = 2.0
N_Y = 16                        # nodes along the periodic axis (the pressure is uniform along it)
METRICS = ("x_c_over_b", "b_eq_over_b", "edge_share")   # the judged integral measures [-]
MAX_P = 2.0                     # formal order of the piecewise-constant collocation (smooth field)


def _hertz_cdf(u):
    """Share of a semi-elliptic load ``sqrt(1 - x^2)`` on ``[-1, u]``, ``u`` clipped to [-1, 1]."""
    u = np.clip(u, -1.0, 1.0)
    return (u * np.sqrt(1.0 - u * u) + np.arcsin(u)) / math.pi + 0.5


def hertz_measures(x_edge_m: float, b_m: float) -> dict:
    """The integral measures of the Hertz pressure (no edge) on the same windows: the limit far from the edge.

    ``edge_share`` is the part of the semi-elliptic load on ``|x - x_edge| <= b``; the rest are 0, 1, 1.
    """
    lo, hi = (x_edge_m - b_m) / b_m, (x_edge_m + b_m) / b_m
    return {"x_c_over_b": 0.0, "b_eq_over_b": 1.0, "p_m_over_p_m_H": 1.0,
            "edge_share": float(_hertz_cdf(hi) - _hertz_cdf(lo))}


def measures(result, x_edge_m: float, b_m: float) -> dict:
    """Integral measures of one solved line contact next to an edge at ``x_edge_m`` (bridge x) [m].

    Parameters
    ----------
    result : StaticResult
        A line contact solved on a periodic strip (``slippy_bridge.contact.solve``), axis 0 = bridge x.
    x_edge_m : float
        Signed position of the flank end on bridge x, the contact point at 0 [m]; ``-0.0`` / ``+0.0`` for a
        point on the end (the sign says on which side the flank stops, construction ``edge_of``).
    b_m : float
        Hertz half-width of the contact (the normaliser) [m].

    Returns
    -------
    dict
        ``w_N_per_m`` (load per unit length carried), ``x_c_over_b`` (signed, + away from the edge),
        ``b_eq_over_b`` (``b_eq / b``), ``p_m_over_p_m_H`` (``b / b_eq``), ``edge_share`` (load within ``b`` of the
        edge over ``w``), ``p0_over_p0_H`` (peak over Hertz ``p0``; grid-dependent), ``singular`` (the peak is on
        the node next to the edge), ``d_end_nodes`` (edge distance in nodes).
    """
    grid = result.grid
    x = np.asarray(grid.X)[:, 0]
    p = np.asarray(result.p).mean(axis=1)                    # [Pa] uniform along the line
    h = grid.h
    w_N_per_m = float(p.sum() * h)
    x_c_m = float((p * x).sum() * h / w_N_per_m)
    b_eq = 2.0 * math.sqrt(float((p * (x - x_c_m) ** 2).sum() * h / w_N_per_m))
    near = np.abs(x - x_edge_m) <= b_m
    away = -math.copysign(1.0, x_edge_m)          # signed zero kept: an edge on the contact point has a side
    i_max = int(np.argmax(p))
    return {"w_N_per_m": w_N_per_m, "x_c_over_b": away * x_c_m / b_m, "b_eq_over_b": b_eq / b_m,
            "p_m_over_p_m_H": b_m / b_eq, "edge_share": float(p[near].sum() * h / w_N_per_m),
            "p0_over_p0_H": float(p[i_max]) / result.hertz.p0,
            "singular": bool(abs(x[i_max] - x_edge_m) <= 1.5 * h), "d_end_nodes": abs(x_edge_m) / h}


def span_for(b_m: float, b_over_h_finest: float) -> float:
    """Domain half-length [half-widths] meeting the R2-08 line rule at the finest grid ``Grid.around`` will build:
    ``max(SPAN_X_MIN, SPAN_RULE_MARGIN 0.23 M^0.75)`` rounded up to 0.1, ``M = 2 b / h``, ``h`` the dyadic floor of
    ``b / b_over_h_finest`` (as ``Grid.around``: never coarser than asked)."""
    h = 2.0 ** math.floor(math.log2(b_m / b_over_h_finest))
    return max(SPAN_X_MIN, math.ceil(SPAN_RULE_MARGIN * 0.23 * (2.0 * b_m / h) ** 0.75 * 10.0) / 10.0)


def grid_series(contact, x_edge_m: float, output_dir, *, b_over_h=B_OVER_H, span_x: float | None = None,
                label: str = "") -> tuple[list[dict], dict, object]:
    """Solve ``contact`` on the grids ``b/h`` (frequency kernel, periodic strip, load control) and judge the measures.

    ``span_x`` [half-widths] is the same at every level; None: :func:`span_for` at the finest level. Returns the
    per-level records (measures, grid, solver flags), the Richardson summary per metric (status, observed order
    ``p``, extrapolated value, ``GCI_fine``, the fine - medium change) and the finest result.
    """
    hertz_ref = hertz(contact)
    span_x = span_for(hertz_ref.b, max(b_over_h)) if span_x is None else span_x
    levels, results = [], {}
    for b_over_h_asked in b_over_h:
        grid = Grid.around(contact, hertz_ref, h=hertz_ref.b / b_over_h_asked, span_x=span_x, n_y_line=N_Y)
        levels.append(grid.h)
        results[grid.h] = solve(contact, grid, output_dir, hertz_result=hertz_ref, use_frequency_domain=True)

    def evaluate(h):
        return {k: measures(results[h], x_edge_m, hertz_ref.b)[k] for k in METRICS}

    richardson_result = richardson(evaluate, levels, options=RichardsonOptions(kind="size", flat_atol=0.0,
                                                                               max_p=MAX_P, stop_when_converged=False),
                                   name="s104_grid", label=label)
    rows = []
    for b_over_h_asked, h in zip(b_over_h, levels):
        r = results[h]
        M = 2.0 * hertz_ref.b / h
        rows.append({"label": label, "b_over_h_asked": b_over_h_asked, "h_m": h, "b_over_h": hertz_ref.b / h, "M": M,
                     "n_x": r.grid.X.shape[0], "span_x": span_x, "span_rule_R2_08": 0.23 * M ** 0.75,
                     "converged": r.converged, "load_ok": r.load_ok, "edge_clear": r.edge_clear,
                     "solve_time_s": r.solve_time, **measures(r, x_edge_m, hertz_ref.b)})
    fine, medium = rows[-1], rows[-2]
    summary = {}
    for k in METRICS:
        f = richardson_result.final[k]
        summary[k] = {"status": f.status, "p": f.p, "extrapolated": f.f_extrapolated, "gci_fine": f.gci_fine_medium,
                      "change_fine_medium": fine[k] - medium[k]}
    return rows, summary, results[levels[-1]]
