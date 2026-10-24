"""Measurements on a contact solution, as records with the unit in the key.

Records are plain dicts; write and read them with ``axisforge.outputs._format.write_records_csv``
/ ``read_records_csv`` (D-0008: AxisForge owns the record format). Generated files go where the
study says (its ``results/`` folder, D-0020), never inside this package.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["measure", "rel_err"]


def _semi_axis(p_cut: np.ndarray, h: float, i0: int, thr: float) -> float:
    """Semi-axis of the loaded region on one grid line ``p_cut`` through the maximum ``i0`` [m].

    It is half the length of the connected loaded segment through the maximum: the Hertz semi-axis
    of a smooth contact, but only one contact patch of a rough one.

    Each edge is placed where ``p**2``, linear in position near a Hertz-like edge, reaches zero,
    extrapolated from the last two loaded nodes; this resolves the edge below one node, unlike a
    count of loaded nodes.
    """
    edges = []
    for step in (1, -1):
        k = i0
        while 0 <= k + step < p_cut.size and p_cut[k + step] > thr:
            k += step
        p2_k, p2_in = p_cut[k] ** 2, p_cut[k - step] ** 2 if k != i0 else p_cut[k] ** 2
        frac = p2_k / (p2_in - p2_k) if p2_in > p2_k else 0.5
        edges.append((k - i0) * h + step * min(frac, 1.0) * h)
    return (edges[0] - edges[1]) / 2.0


def measure(p: np.ndarray, h: float, line: bool = False, *, gap: np.ndarray | None = None,
            X: np.ndarray | None = None, Y: np.ndarray | None = None) -> dict:
    """Measure a pressure field (and, for a lubricated contact, its film) on the bridge grid.

    Parameters
    ----------
    p : ndarray
        Pressure [Pa].
    h : float
        Grid spacing [m].
    line : bool
        Line contact (periodic along axis 1): ``s_y_m`` is ``inf``.
    gap : ndarray, optional
        Film thickness (the gap between the deformed surfaces) on the same nodes [m], from an EHL
        solve. None (dry contact): only the pressure keys are returned, exactly as before.
    X, Y : ndarray, optional
        Node coordinates of the grid [m] (``Grid.X``, ``Grid.Y``); with ``gap``, they locate the
        contact centre and the minimum film.

    Returns
    -------
    dict
        ``p0_Pa``, ``s_x_m`` and ``s_y_m`` (semi-axes along and across rolling), ``load_N``,
        ``apex_offset_nodes`` (pressure maximum vs grid centre). With ``gap`` (planned, R-033):
        ``h_c_m``, the central film at the node where ``X == 0`` and ``Y == 0`` (raise if there is
        none: an asymmetric inlet/outlet domain no longer has its centre at node n//2, GL-06);
        ``h_min_m = gap.min()``, the minimum film (in the outlet constriction of an EHL contact);
        ``x_h_min_m``, ``y_h_min_m``, its position (``y_h_min_m`` is ``nan`` for a line).

    Raises
    ------
    NotImplementedError
        If ``gap`` is given: scaffold (OPEN R-033 R-048 GL-06).
    """
    if gap is not None:
        raise NotImplementedError("measure(gap=...): scaffold, OPEN R-033 R-048 GL-06")
    p = np.asarray(p, dtype=float)
    i0, j0 = np.unravel_index(int(np.argmax(p)), p.shape)
    thr = 1e-6 * p[i0, j0]
    s_x = _semi_axis(p[:, j0], h, i0, thr)
    s_y = math.inf if line else _semi_axis(p[i0, :], h, j0, thr)
    # OPEN R-048 GL-06: the centre is node n//2 only for a symmetric span_x (Grid.around); not for (inlet, outlet)
    c0, c1 = p.shape[0] // 2, p.shape[1] // 2
    offset = abs(i0 - c0) if line else math.hypot(i0 - c0, j0 - c1)
    return {"p0_Pa": float(p[i0, j0]), "s_x_m": s_x, "s_y_m": s_y, "load_N": float(p.sum() * h**2),
            "apex_offset_nodes": float(offset)}


def rel_err(num: float, ref: float) -> float:
    """``(num - ref) / ref``; ``nan`` when the reference is missing, infinite or zero."""
    return math.nan if ref is None or not math.isfinite(ref) or ref == 0 else (num - ref) / ref
