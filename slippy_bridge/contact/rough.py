"""Roughness generation.

Roughness is a height array [m] on the same grid as the surfaces: generated here
(:func:`fractal`) or measured and resampled by the caller. ``smooth.pair`` adds it to the
profiles (argument ``rough``); how it is shared between the two bodies in the composite and
two_profiles builds is explained there.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fractal"]


def fractal(shape: tuple[int, int], h: float, sigma: float, hurst: float,
            roll_off_frequency: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Self-affine (Hurst fractal) roughness, zero mean, RMS ``sigma`` [m], via SlipPY.

    Parameters
    ----------
    shape : (int, int)
        Grid shape, as the surfaces (``grid.X.shape``).
    h : float
        Grid spacing [m].
    sigma : float
        RMS roughness Rq [m].
    hurst : float
        Hurst exponent, in [0, 1] (SlipPY ``hurst_exponent``).
    roll_off_frequency : float
        Roll-off wavevector of the PSD [rad/m] (SlipPY ``roll_off_frequency``): the PSD is flat
        below it. Default 0: no roll-off.
    seed : int, optional
        Seed of NumPy's global generator before the realisation (SlipPY draws the phases from it),
        so a seed reproduces the surface.
    """
    import slippy.surface as s
    if seed is not None:
        np.random.seed(seed)
    n = max(shape)                     # SlipPY generates square fractal surfaces: make one and crop
    surf = s.HurstFractalSurface(sigma, hurst, roll_off_frequency, grid_spacing=h, shape=(n, n), generate=True)
    z = np.asarray(surf.profile, dtype=float)[: shape[0], : shape[1]]
    z = z - z.mean()
    return z * (sigma / z.std())
