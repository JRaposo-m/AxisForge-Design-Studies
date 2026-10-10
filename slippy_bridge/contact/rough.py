"""Rough surfaces = smooth surfaces + roughness.

Roughness is a height array [m] on the same grid as the smooth surfaces: generated here
(:func:`fractal`) or measured and resampled by the caller. :func:`add_roughness` adds it to a
smooth pair from ``smooth.py``:

* composite build: the roughness of both bodies is summed on the equivalent body (surface 1).
  Valid for static contact and steady lubrication without sliding;
* two_profiles build: each body gets its own roughness (needed once the surfaces slide or move
  at different speeds).

SlipPY's ``Surface + Surface`` would crop silently to the smaller shape and drop the material;
here the shapes must match and each surface keeps its material.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fractal", "add_roughness"]


def fractal(shape: tuple[int, int], h: float, sigma: float, hurst: float, roll_off: float = 0.0,
            seed: int | None = None) -> np.ndarray:
    """Self-affine (Hurst fractal) roughness, zero mean, RMS ``sigma`` [m], via SlipPY.

    Parameters
    ----------
    shape : (int, int)
        Grid shape, as the smooth surfaces (``X.shape``).
    h : float
        Grid spacing [m].
    sigma : float
        RMS roughness Rq [m].
    hurst : float
        Hurst exponent, in [0, 1].
    roll_off : float
        Roll-off frequency of the PSD (SlipPY ``roll_off_frequency``).
    seed : int, optional
        Seed of NumPy's global generator before the realisation (SlipPY draws the phases from it),
        so a seed reproduces the surface.
    """
    import slippy.surface as s
    if seed is not None:
        np.random.seed(seed)
    n = max(shape)                     # SlipPY generates square fractal surfaces: make one and crop
    surf = s.HurstFractalSurface(sigma, hurst, roll_off, grid_spacing=h, shape=(n, n), generate=True)
    z = np.asarray(surf.profile, dtype=float)[: shape[0], : shape[1]]
    z = z - z.mean()
    return z * (sigma / z.std())


def add_roughness(pair, rough_1: np.ndarray | None = None, rough_2: np.ndarray | None = None,
                  composite: bool = True):
    """Add roughness to a smooth pair ``(surface_1, surface_2)``; returns a new pair.

    Parameters
    ----------
    pair : tuple of slippy Surface
        From ``smooth.composite`` or ``smooth.two_profiles``.
    rough_1, rough_2 : ndarray, optional
        Roughness heights [m] of body 1 and body 2 (``None`` = smooth body).
    composite : bool
        True: both roughnesses on surface 1 (composite build). False: each on its own surface.

    Raises
    ------
    ValueError
        If a roughness array does not have the shape of the surfaces.
    """
    import slippy.surface as s
    s1, s2 = pair
    z1, z2 = np.array(s1.profile, dtype=float), np.array(s2.profile, dtype=float)
    for name, r in (("rough_1", rough_1), ("rough_2", rough_2)):
        if r is not None and np.shape(r) != z1.shape:
            raise ValueError(f"{name} has shape {np.shape(r)}, the surfaces {z1.shape}")
    r1 = 0.0 if rough_1 is None else rough_1
    r2 = 0.0 if rough_2 is None else rough_2
    if composite:
        z1 = z1 + r1 + r2
    else:
        z1, z2 = z1 + r1, z2 + r2
    out = []
    for z, old in ((z1, s1), (z2, s2)):
        new = s.Surface(profile=z, grid_spacing=old.grid_spacing)
        new.material = old.material
        out.append(new)
    return tuple(out)
