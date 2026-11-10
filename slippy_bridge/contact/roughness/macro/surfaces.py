"""Rough heights on a grid (:func:`rough_heights`), the ``heights`` that ``solvers.surface_pair`` and
``solvers.solve`` add to the body profiles.

Generation through SlipPY (D-0018), with the reason wherever SlipPY is not used (SlipPY
local-fixes df5916c):

* ``"acf"``: ``slippy.surface.RandomFilterSurface`` with ``linear_transform`` (Patir 1978; Watson et
  al. 2020), Gaussian white noise convolved with the fitted filter, ``discretise(periodic=True)``
  (wrap-padded noise and a ``valid`` convolution: periodic for any filter size, ``Random.py:820,
  824``). ``fir_filter`` is refused (R2-17): it samples the ACF on one quadrant and takes
  ``sqrt(fft2(acf))``, complex and never transformed back, as a spatial filter
  (``slippy/surface/Random.py:752-757``, convolved at 824): measured ACF 0.65 against 0.11 at 10
  nodes, Rq 78 for a unit target (probe C2). ``linear_transform`` computes ``max_it`` and never
  passes it to scipy ``minimize`` (``Random.py:518-519, 536-541``): not given here.
* ``"psd"``: ``slippy.surface.HurstFractalSurface`` (random phases, amplitudes from the PSD,
  periodic by construction) on a square lattice: for a non-square shape it returns the transposed
  array (``meshgrid(qx, qy)``, ``rand(n, m)``, ``FFTBased.py:318, 333, 350``). A non-square or
  odd lattice gets the smallest even square, band-limited there and cropped: periodic then only on
  an even square lattice.
* ``"measured"``: heights given or read by ``slippy.surface.Surface(file_name=...)``, levelled by
  ``slippy.surface.subtract_polynomial(z, 1)`` (it subtracts the bilinear ``a + b x + c y + d xy``,
  ``roughness_funcs.py:848-851``, not the least-squares plane of ISO 25178, OPEN R4-08), resampled
  by a bicubic spline (``scipy.interpolate.RectBivariateSpline``). Not SlipPY
  ``Surface.interpolate``/``resample``: they build their nodes with ``np.arange(0, extent, h)``
  (one node too many for some ``h``, R-049) from the corner, not on given node coordinates, and
  ``interpolate`` reads its two coordinate arguments on swapped axes between ``mode='nearest'`` and
  ``'linear'/'cubic'`` (``Surface_class.py:1205-1224``; ``resample`` 751-752).
* Band limit (S- and L-filter): a numpy FFT mask, as the probe's ``band_limit``. Not SlipPY
  ``low_pass_filter`` (``roughness_funcs.py:745-754``): its Bessel kernel is centred on the corner
  node (``meshgrid(arange(1, n + 1))``), so it shifts the surface, is transposed for a non-square
  profile and is truncated to the profile size.

The random surface is drawn periodic on its generation lattice (``lattice``), so the FFT band
limit has no edge; ``Rq`` is then set exactly (``z <- Rq (z - mean) / std`` on the lattice nodes)
after the band limit, which lowers it. The filter also changes the ACF: ``beta*`` is that of the
unfiltered surface, and the realised 10 % length (``RoughHeights.beta_star_realised``) is larger
when ``lambda_s`` is near ``beta*`` (1.20 beta* at lambda_s = 0.8 beta*, physics review R4a). A grid
with a spacing a power-of-two multiple of the lattice's samples the lattice at its own nodes: the
same surface on every grid level (probe C2).
"""
from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import slippy.surface as s
from scipy.interpolate import RectBivariateSpline

from ...core.contact import DC
from .spec import RoughnessSpec

if TYPE_CHECKING:
    from ...mesh.grid import Grid

__all__ = ["RoughHeights", "rough_heights", "band_limit", "LAMBDA_S_OVER_H_MIN", "BETA_OVER_H_MIN"]

# Grid rule of a rough contact (R2-20, OPEN R4-04): a proposal from one case (probe C2: v101 ball, Rq 0.2 um,
# lambda_s 16 um, three seeds), area changes -4.8 % and -1.6 % between lambda_s/h 2.1, 4.2 and 8.4; "within
# about 2 % of the next level" is extrapolated. In the probe beta*/h and lambda_s/h changed together, so the
# beta rule was never separated: here beta*/h is checked on the generation lattice (the fit of the ACF),
# lambda_s/h on the grid of the solve (its convergence).
LAMBDA_S_OVER_H_MIN = 8.0
BETA_OVER_H_MIN = 10.0


@dataclass(**DC)
class RoughHeights:
    """Heights of one rough surface on a grid and the record of how they were made.

    Parameters
    ----------
    z : ndarray
        Heights [m] on the grid nodes (grid frame, axis 0 = grid x), positive out of the body
        toward the other one: added to the body's ``Contact.profiles`` (SlipPY sign, apex 0).
    spec : RoughnessSpec
    h : float
        Grid spacing [m].
    h_lattice : float
        Spacing of the generation lattice [m] (``h`` when no lattice was given).
    rq_lattice : float
        ``Rq`` on the generation lattice [m]: the target, exactly, for a random surface.
    rq_grid : float
        ``Rq`` of ``z`` about its own mean [m]: differs from ``rq_lattice`` when the grid samples a
        part of the lattice.
    beta_star_realised : float
        10 % correlation length of the generated surface on the lattice [m]: where the mean of the
        normalised periodic ACF along both axes first falls to 0.1 (linear between nodes); nan for
        a measured or smooth surface.
    lambda_s_over_h : float
        ``lambda_s / h`` on the grid [-] (nan without ``lambda_s``).
    beta_over_h : float
        ``beta* / h_lattice`` [-] (nan for a non-ACF kind).
    grid_rule_ok : bool
        ``lambda_s / h >= LAMBDA_S_OVER_H_MIN`` and, for an ACF, ``beta* / h_lattice >=
        BETA_OVER_H_MIN`` (False for a measured surface without ``lambda_s``: not checkable).
    periodic_y : bool
        ``z`` repeats with the grid's length along grid y (``Grid.length_y_m``): a random surface
        on a lattice of that length along y (an ``"acf"`` spec, or a ``"psd"`` spec on an even
        square lattice); False for a measured surface. A line contact (periodic along y) needs it
        (``solvers.surface_pair``).
    warnings : tuple of str
        The grid-rule messages (also issued with ``warnings.warn``) and the others of the build
        (SlipPY's filter fit, a cropped PSD square, a roll-off below the lattice).
    """

    z: np.ndarray
    spec: RoughnessSpec
    h: float
    h_lattice: float
    rq_lattice: float
    rq_grid: float
    beta_star_realised: float
    lambda_s_over_h: float
    beta_over_h: float
    grid_rule_ok: bool
    periodic_y: bool
    warnings: tuple[str, ...] = ()

    def record(self) -> dict:
        """The scalars for a CSV row: the spec's (:meth:`RoughnessSpec.record`) and the grid's."""
        return {**self.spec.record(), "h_m": self.h, "h_lattice_m": self.h_lattice, "rq_lattice_m": self.rq_lattice,
                "rq_grid_m": self.rq_grid, "beta_star_realised_m": self.beta_star_realised,
                "lambda_s_over_h": self.lambda_s_over_h, "beta_over_h_lattice": self.beta_over_h,
                "grid_rule_ok": self.grid_rule_ok, "periodic_y": self.periodic_y, "warnings": " | ".join(self.warnings)}


def band_limit(z: np.ndarray, h: float, lambda_s: float | None, lambda_c: float | None = None) -> np.ndarray:
    """``z`` without its mean and without every Fourier component of wavelength below ``lambda_s``
    or above ``lambda_c`` [m] (the S- and L-filter of ``spec``, ideal cut-offs on the 2-D wave
    vector ``|k| = hypot(k_x, k_y)``, ``k`` in cycles/m). Exact for a periodic ``z``; on a
    non-periodic one (a measurement) the FFT sees the jump at the edges (leakage, OPEN R4-08).
    ``None`` leaves that end open.

    Raises
    ------
    ValueError
        If no wave vector of the lattice lies in the band (``lambda_s`` above the lattice size, or
        ``lambda_s``, ``lambda_c`` too close): the surface would be all zero.
    """
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(z.shape[0], h), np.fft.fftfreq(z.shape[1], h), indexing="ij"))
    keep = k > 0.0
    if lambda_s is not None:
        keep &= k <= 1.0 / lambda_s
    if lambda_c is not None:
        keep &= k >= 1.0 / lambda_c
    if not keep.any():
        raise ValueError(f"band_limit: no wave vector of the {z.shape} lattice (h {h} m) lies between lambda_s "
                         f"{lambda_s} m and lambda_c {lambda_c} m")
    return np.real(np.fft.ifft2(np.fft.fft2(z) * keep))


def rough_heights(spec: RoughnessSpec, grid: Grid, *, lattice: Grid | None = None) -> RoughHeights:
    """Heights of the surface ``spec`` on the nodes of ``grid`` [m], with the grid-rule check.

    The surface is made on the generation lattice ``lattice`` (default: ``grid`` itself) and
    sampled at the nodes of ``grid``; pass the same ``lattice`` (the finest grid, as large as the
    largest) to every grid of a convergence study so that each samples one surface. Random kinds:
    a unit-variance periodic realisation of ``seed`` on the lattice, band-limited
    (:func:`band_limit`), then scaled to ``Rq`` exactly on the lattice nodes. Measured: levelled,
    band-limited on its own lattice, then resampled on the lattice nodes by a bicubic spline, the
    centre node ``(n_0//2, n_1//2)`` of the measurement on the contact centre; ``Rq`` is the
    measurement's (its mean is removed over the whole measurement, not over the lattice window).
    The heights are on the grid frame (axis 0 = grid x); an isotropic random surface does not
    depend on ``grid.angle``, a measured one is taken as given on the grid axes.

    Parameters
    ----------
    spec : RoughnessSpec
        Validated here (``validate_or_raise``).
    grid : Grid
    lattice : Grid, optional
        Generation lattice: ``lattice.h`` divides ``grid.h`` by a power of two, the nodes of
        ``grid`` are nodes of ``lattice`` and ``lattice.angle == grid.angle``.

    Returns
    -------
    RoughHeights
        ``z`` all 0 when a random spec has ``Rq = 0`` (nothing is drawn).

    Raises
    ------
    ValueError
        From ``spec.validate_or_raise`` and :func:`band_limit`; if ``grid`` is not on the lattice
        nodes; if a measurement does not cover the lattice, or is resampled on a lattice coarser
        than it without ``lambda_s >= 2 h_lattice`` (the spline then samples points: aliasing).

    Warns
    -----
    UserWarning
        When the grid rule fails (``lambda_s / h < 8``, or ``beta* / h_lattice < 10`` for an ACF),
        with the numbers; never silently (also in ``RoughHeights.warnings``).
    """
    spec.validate_or_raise()
    lattice = grid if lattice is None else lattice
    i, j = _nodes_on(grid, lattice)
    notes: list[str] = []
    beta_real = math.nan
    smooth = spec.random and spec.target_rq() == 0.0
    if smooth:
        z_m = np.zeros(lattice.X.shape)
    elif spec.random:
        z_m = _random(spec, lattice, notes)
        if spec.kind == "acf":
            beta_real = _beta_star(z_m, lattice.h)
    else:
        z_m = _measured(spec, lattice)
    z = z_m[np.ix_(i, j)]

    lam_h = math.nan if spec.lambda_s is None else spec.lambda_s / grid.h
    beta_h = spec.beta_star / lattice.h if spec.kind == "acf" else math.nan
    who = spec.label or spec.kind
    rule: list[str] = []
    if smooth:
        pass                                                              # nothing to resolve
    elif spec.lambda_s is None:
        rule.append(f"{who}: no lambda_s, the grid rule cannot be checked (h = {grid.h:.4g} m)")
    else:
        if lam_h < LAMBDA_S_OVER_H_MIN:
            rule.append(f"{who}: lambda_s/h = {lam_h:.3g} < {LAMBDA_S_OVER_H_MIN:g} (lambda_s {spec.lambda_s:.4g} m, "
                        f"h {grid.h:.4g} m): area and p_max not grid-converged (R2-20)")
        if spec.kind == "acf" and beta_h < BETA_OVER_H_MIN:
            rule.append(f"{who}: beta*/h_lattice = {beta_h:.3g} < {BETA_OVER_H_MIN:g} (beta* {spec.beta_star:.4g} m, "
                        f"h_lattice {lattice.h:.4g} m): the ACF is not resolved (R2-20)")
    for msg in rule:
        warnings.warn(msg, stacklevel=2)
    # OPEN R4-19: the even-square test repeats the one of _random (same predicate on the shape)
    n0, n1 = lattice.X.shape
    periodic_lattice = smooth or spec.kind == "acf" or (spec.kind == "psd" and n0 == n1 and n0 % 2 == 0)
    periodic_y = periodic_lattice and math.isclose(lattice.length_y_m, grid.length_y_m)
    return RoughHeights(z=z, spec=spec, h=grid.h, h_lattice=lattice.h, rq_lattice=float(z_m.std()),
                        rq_grid=float(z.std()), beta_star_realised=beta_real, lambda_s_over_h=lam_h,
                        beta_over_h=beta_h, grid_rule_ok=not rule, periodic_y=periodic_y,
                        warnings=tuple(notes + rule))


def _nodes_on(grid: Grid, lattice: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Indices along axis 0 and axis 1 of ``lattice`` of the nodes of ``grid``."""
    step = grid.h / lattice.h
    if lattice.angle != grid.angle or step < 1 or not math.log2(step).is_integer():
        raise ValueError(f"rough_heights: the lattice (h {lattice.h} m, angle {lattice.angle}) must have the "
                         f"grid's angle {grid.angle} and a spacing h / 2^k of the grid's h {grid.h} m")
    idx = []
    for axis, (a, a0) in enumerate(((grid.X[:, 0], lattice.X[0, 0]), (grid.Y[0, :], lattice.Y[0, 0]))):
        f = (a - a0) / lattice.h
        k = np.rint(f).astype(int)
        if np.any(np.abs(f - k) > 1e-6) or k.min() < 0 or k.max() >= lattice.X.shape[axis]:
            raise ValueError("rough_heights: the grid nodes are not nodes of the lattice (or lie outside it)")
        idx.append(k)
    return idx[0], idx[1]


def _random(spec: RoughnessSpec, lattice: Grid, notes: list[str]) -> np.ndarray:
    """A periodic realisation of ``spec`` on the lattice nodes, band-limited and scaled to ``Rq``."""
    shape = lattice.X.shape
    state = np.random.get_state()             # SlipPY draws from numpy's global state: restore it
    try:
        np.random.seed(spec.seed)
        if spec.kind == "acf":
            gen, fit_notes = _filter(spec.acf_shape, spec.beta_star / lattice.h)
            notes.extend(fit_notes)
            # create_new: the cached generator keeps no profile
            z = np.asarray(gen.discretise(shape, periodic=True, create_new=True).profile, dtype=float)
        else:
            n = max(shape) + max(shape) % 2   # even square (HurstFractalSurface)
            if (n, n) != shape:
                notes.append(f"psd: drawn on a {n}x{n} square, band-limited and cropped to {shape}: not periodic")
            if spec.q_r and spec.q_r < 2.0 * math.pi / (n * lattice.h):
                notes.append(f"psd: q_r {spec.q_r:.4g} rad/m below the lowest wave number 2 pi / L = "
                             f"{2.0 * math.pi / (n * lattice.h):.4g} rad/m: no plateau on the lattice (pure power law)")
            gen = s.HurstFractalSurface(1.0, spec.hurst, spec.q_r, shape=(n, n), grid_spacing=lattice.h)
            with np.errstate(divide="ignore"):   # SlipPY's q = 0 term, set to 0 after the power
                gen.discretise()
            z = np.asarray(gen.profile, dtype=float)
    finally:
        np.random.set_state(state)
    z = band_limit(z, lattice.h, spec.lambda_s, spec.lambda_c)[:shape[0], :shape[1]]
    return spec.target_rq() * (z - z.mean()) / z.std()                    # Rq exact after the filter


# OPEN R4-04: filter lags to about 4 beta* (2 ceil(2 beta*/h) + 1 nodes): ACF error 0.010-0.015 at
# beta*/h 5 on 512^2 against 0.11-0.12 with lags to 2 beta* (scratch check, R4 log)
@functools.lru_cache(maxsize=8)
def _filter(acf_shape: str, beta_nodes: float):
    """SlipPY ``RandomFilterSurface`` with its ``linear_transform`` filter fitted for a unit-Rq ACF of
    correlation length ``beta_nodes`` [nodes] (the fit depends on beta*/h only), and the fit's
    warnings. Cached (the fit takes seconds to a minute); callers draw with ``create_new=True`` so
    the shared generator is not changed."""
    if acf_shape == "exponential":
        acf = s.ACF("exp", 1.0, beta_nodes, beta_nodes)                 # Rq^2 exp(-2.3 r / beta*)
    else:
        def gaussian(x, y):                                             # Rq^2 exp(-2.3 (r / beta*)^2)
            return np.exp(-2.3 * (x**2 + y**2) / beta_nodes**2)
        acf = s.ACF(gaussian)
    n = 2 * math.ceil(2.0 * beta_nodes) + 1    # odd: a centred filter, as SlipPY's docstring asks (Random.py:772-774)
    gen = s.RandomFilterSurface(target_acf=acf, grid_spacing=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.linear_transform(filter_shape=(n, n), symmetric=True)
    return gen, tuple(f"linear_transform: {w.message}" for w in caught)


def _beta_star(z: np.ndarray, h: float) -> float:
    """10 % correlation length [m] of a periodic ``z``: the normalised ACF (by FFT) averaged along
    axis 0 and axis 1, its first fall to 0.1, linear between nodes; nan if it never falls."""
    z = z - z.mean()
    acf = np.real(np.fft.ifft2(np.abs(np.fft.fft2(z)) ** 2))
    acf /= acf[0, 0]
    n = min(z.shape) // 2
    r = 0.5 * (acf[:n, 0] + acf[0, :n])
    below = np.nonzero(r <= 0.1)[0]
    if below.size == 0:
        return math.nan
    k = int(below[0])
    return float((k - 1 + (r[k - 1] - 0.1) / (r[k - 1] - r[k])) * h)


def _measured(spec: RoughnessSpec, lattice: Grid) -> np.ndarray:
    """A measured surface levelled, band-limited and resampled on the lattice nodes [m]."""
    if spec.heights is not None:
        z, h = np.asarray(spec.heights, dtype=float), float(spec.spacing)
    else:
        surface = s.Surface(file_name=spec.file, grid_spacing=spec.spacing)
        z, h = np.asarray(surface.profile, dtype=float), surface.grid_spacing
        if not (h and h > 0 and np.all(np.isfinite(z))):
            raise ValueError(f"rough_heights: {spec.file}: spacing not known or heights not finite "
                             "(give spacing; fill holes)")
    if lattice.h > h and not (spec.lambda_s is not None and spec.lambda_s >= 2.0 * lattice.h):
        raise ValueError(f"rough_heights: the measurement (spacing {h} m) is resampled on a coarser lattice "
                         f"(h {lattice.h} m): give lambda_s >= 2 h_lattice so that the spline does not alias")
    z, _ = s.subtract_polynomial(z, 1)
    z = band_limit(z, h, spec.lambda_s, spec.lambda_c)
    x = (np.arange(z.shape[0]) - z.shape[0] // 2) * h
    y = (np.arange(z.shape[1]) - z.shape[1] // 2) * h
    xm, ym = lattice.X[:, 0], lattice.Y[0, :]
    if xm.min() < x[0] or xm.max() > x[-1] or ym.min() < y[0] or ym.max() > y[-1]:
        raise ValueError(f"rough_heights: the measurement ({z.shape[0] * h:.4g} x {z.shape[1] * h:.4g} m) "
                         "does not cover the lattice")
    return RectBivariateSpline(x, y, z, kx=3, ky=3)(xm, ym)
