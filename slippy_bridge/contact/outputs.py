"""Measurements on a contact solution, as records with the unit in the key.

Records are plain dicts; write and read them with ``axisforge.outputs._format.write_records_csv``
/ ``read_records_csv`` (D-0008: AxisForge owns the record format). Generated files go where the
study says (``artifacts/``), never inside this package.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["measure", "rel_err"]


def _half_extent(line_p: np.ndarray, h: float, i0: int, thr: float) -> float:
    """Half-extent of the loaded region on one grid line through the maximum ``i0`` [m].

    Each edge is placed where ``p**2``, linear in position near a Hertz-like edge, reaches zero,
    extrapolated from the last two loaded nodes; this resolves the edge below one node, unlike a
    count of loaded nodes.
    """
    edges = []
    for step in (1, -1):
        k = i0
        while 0 <= k + step < line_p.size and line_p[k + step] > thr:
            k += step
        p_k, p_in = line_p[k] ** 2, line_p[k - step] ** 2 if k != i0 else line_p[k] ** 2
        frac = p_k / (p_in - p_k) if p_in > p_k else 0.5
        edges.append((k - i0) * h + step * min(frac, 1.0) * h)
    return (edges[0] - edges[1]) / 2.0


def measure(p: np.ndarray, h: float, line: bool = False) -> dict:
    """Measure a pressure field on the bridge grid (rolling along axis 0).

    Parameters
    ----------
    p : ndarray
        Pressure [Pa].
    h : float
        Grid spacing [m].
    line : bool
        Line contact (periodic along axis 1): ``s_y_m`` is ``inf``.

    Returns
    -------
    dict
        ``p0_Pa``, ``s_x_m`` and ``s_y_m`` (semi-axes along and across rolling), ``load_N``,
        ``apex_offset_nodes`` (pressure maximum vs grid centre).
    """
    p = np.asarray(p, dtype=float)
    i0, j0 = np.unravel_index(int(np.argmax(p)), p.shape)
    thr = 1e-6 * p[i0, j0]
    s_x = _half_extent(p[:, j0], h, i0, thr)
    s_y = math.inf if line else _half_extent(p[i0, :], h, j0, thr)
    c0, c1 = p.shape[0] // 2, p.shape[1] // 2
    offset = abs(i0 - c0) if line else math.hypot(i0 - c0, j0 - c1)
    return {"p0_Pa": float(p[i0, j0]), "s_x_m": s_x, "s_y_m": s_y, "load_N": float(p.sum() * h**2),
            "apex_offset_nodes": float(offset)}


def rel_err(num: float, ref: float) -> float:
    """``(num - ref) / ref``; ``nan`` when the reference is missing, infinite or zero."""
    return math.nan if ref is None or not math.isfinite(ref) or ref == 0 else (num - ref) / ref
