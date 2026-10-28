"""Measures of a rough contact on a pressure field the caller gives (no solver here).

The pressure field and the grid come from the caller's solve (SlipPY ``loads_z`` of a
``StaticStep`` on the surfaces of :func:`surfaces.rough_surface_pair`); the bridge's ``solve`` does
not take roughness yet. Its ``_measure`` assumes one Hertz-like patch (semi-axes through the peak);
these measures do not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from ...core.contact import DC
from ...solvers.static import _EDGE

if TYPE_CHECKING:
    from ...mesh.grid import Grid
    from .surfaces import RoughHeights

__all__ = ["RoughMeasures", "rough_measures"]

# [-] a node is in contact when p > THRESHOLD_REL p_max: the numerical zero of the bridge's own measure
# (solvers.static._EDGE, relative to the peak there too; one source); probe C2: p > 0 and p > 1e-3 p0 gave the
# same area to 0.1 %
THRESHOLD_REL = _EDGE


@dataclass(**DC)
class RoughMeasures:
    """Real contact area, contact patches and pressures of one pressure field.

    Every measure depends on the grid: the area is a node count (its convergence with ``h`` is the
    grid rule, R2-20); the patch count was never checked for convergence (OPEN R4-10). For a line
    contact ``load`` is that of the strip (one period).

    Parameters
    ----------
    area_real : float
        Real contact area [m^2]: number of nodes with ``p > threshold`` times ``h^2``.
    area_reference : float
        The reference area given [m^2] (e.g. the Hertz area ``pi a b`` of the smooth contact, or the
        nominal area of a patch).
    area_ratio : float
        ``area_real / area_reference`` [-].
    n_patches : int
        Connected components of the contact nodes (``connectivity``).
    p_max : float
        Largest node pressure [Pa].
    p_mean_real : float
        Load over the real area, ``sum(p) h^2 / area_real`` [Pa] (nan without contact).
    load : float
        ``sum(p) h^2`` [N].
    threshold : float
        The contact threshold used [Pa].
    connectivity : {4, 8}
        4: nodes sharing an edge belong to one patch; 8: also nodes sharing a corner. A periodic
        field is not wrapped: a patch across the boundary counts twice.
    h : float
        Grid spacing [m].
    surfaces : tuple of (RoughHeights or None)
        The record of the surfaces of the solve (what was solved: Rq, lambda_s, beta*, seed, h).
    """

    area_real: float
    area_reference: float
    area_ratio: float
    n_patches: int
    p_max: float
    p_mean_real: float
    load: float
    threshold: float
    connectivity: int
    h: float
    surfaces: tuple = ()

    def record(self) -> dict:
        """One flat dict for a CSV row (unit suffixes); each surface's record prefixed ``s1_`` / ``s2_``."""
        units = {"area_real": "_m2", "area_reference": "_m2", "area_ratio": "", "n_patches": "", "p_max": "_Pa",
                 "p_mean_real": "_Pa", "load": "_N", "threshold": "_Pa", "connectivity": "", "h": "_m"}
        out = {k + u: getattr(self, k) for k, u in units.items()}
        for k, rh in enumerate(self.surfaces, start=1):
            if rh is not None:
                out.update({f"s{k}_{key}": v for key, v in rh.record().items()})
        return out


def rough_measures(p: np.ndarray, grid: Grid, *, area_reference: float, threshold: float | None = None,
                   connectivity: int = 4, surfaces: tuple = ()) -> RoughMeasures:
    """Measures of the pressure field ``p`` [Pa] on ``grid`` (see :class:`RoughMeasures`).

    Parameters
    ----------
    p : ndarray
        Node pressures [Pa], the grid's shape.
    grid : Grid
    area_reference : float
        Reference area [m^2], > 0.
    threshold : float, optional
        Contact threshold [Pa]; default ``THRESHOLD_REL * p.max()``.
    connectivity : {4, 8}
        Patch connectivity (``scipy.ndimage.label`` with the cross or the full 3x3 structure).
    surfaces : tuple of (RoughHeights or None)
        Recorded as given (``rough_surface_pair``'s last two returns).

    Raises
    ------
    ValueError
        If ``p`` is not of the grid's shape, ``area_reference`` not positive, ``connectivity`` not 4
        or 8.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != grid.X.shape:
        raise ValueError(f"rough_measures: p has shape {p.shape}, the grid {grid.X.shape}")
    if not (area_reference > 0 and math.isfinite(area_reference)):
        raise ValueError(f"rough_measures: area_reference must be positive and finite, got {area_reference}")
    if connectivity not in (4, 8):
        raise ValueError(f"rough_measures: connectivity must be 4 or 8, got {connectivity}")
    p_max = float(p.max())
    thr = THRESHOLD_REL * p_max if threshold is None else float(threshold)
    loaded = p > thr
    _, n = ndimage.label(loaded, structure=ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2))
    h2 = grid.h**2
    area = float(loaded.sum() * h2)
    load = float(p.sum() * h2)
    return RoughMeasures(area_real=area, area_reference=float(area_reference),
                                area_ratio=area / area_reference, n_patches=int(n), p_max=p_max,
                                p_mean_real=load / area if area > 0 else math.nan, load=load, threshold=thr,
                                connectivity=connectivity, h=grid.h, surfaces=tuple(surfaces))
