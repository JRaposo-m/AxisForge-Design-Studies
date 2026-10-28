"""How a rough surface is defined: :class:`RoughnessSpec`, the input of :func:`surfaces.rough_heights`.

A spec is the statistics (or the measurement) of one surface's roughness, its band limits and the
seed of the realisation, in SI units. It is not tied to a grid: the same spec gives the same
surface on every grid that :func:`surfaces.rough_heights` samples it on through one generation
lattice (its ``lattice``).

Definitions (stated once, used by every function of ``roughness``):

* ``Rq`` - root mean square of the heights about their mean plane over the generation lattice
  (``Sq`` of ISO 25178-2 for an areal surface; no ISO note is held in the library, OPEN R4-08).
* ACF ``R(r) = Rq^2 exp(-2.3 r / beta*)`` (exponential) or ``Rq^2 exp(-2.3 (r / beta*)^2)``
  (Gaussian), isotropic, ``r`` the lag length. ``beta*`` is the 10 % correlation length,
  ``R(beta*) = 0.100 Rq^2`` (2.3 is SlipPY's rounding of ln 10, ``slippy.surface.ACF('exp', ...)``;
  Patir 1978, as SlipPY cites it).
* PSD ``C(q)`` with ``Rq^2 = integral of C(q) d^2q`` over the wave-vector plane (Persson 2001,
  Eq. (20), PDF p. 8, journal p. 3846). Its shape (Muser 2017, Eq. (1), PDF p. 3), self-affine with roll-off:
  ``C = C0`` for ``q < q_r``, ``C0 (q / q_r)^(-2(1 + H))`` for ``q_r <= q <= q_s``, 0 above
  ``q_s``; ``q`` in rad/m (``q = 2 pi / lambda``), the unit of SlipPY ``HurstFractalSurface``.
* S-filter ``lambda_s`` - every Fourier component of wavelength below ``lambda_s`` is removed
  (an ideal cut-off in the 2-D spectrum: ``|k| <= 1 / lambda_s``, ``k`` in cycles/m, as the probe's
  ``band_limit``; not the Gaussian filter of ISO 16610-61, OPEN R4-08). For a PSD spec it is
  also the PSD cut-off: ``q_s = 2 pi / lambda_s`` (one source for one quantity, OPEN R4-02).
* L-filter ``lambda_c`` - every component of wavelength above ``lambda_c`` is removed
  (optional; the mean plane is always removed).
* ``rq`` is the Rq after the S- and L-filter (the surface is rescaled after them); ``beta*`` and the
  ACF are those of the unfiltered surface. The filter lengthens the realised 10 % length when
  ``lambda_s`` is near ``beta*`` (1.20 beta* at lambda_s = 0.8 beta*, the probe C2 case), recorded in
  ``RoughHeights.beta_star_realised``.

SlipPY names: ``rq`` = ``sigma`` (``ACF``, ``HurstFractalSurface``), ``beta_star`` = ``beta_x`` =
``beta_y`` of ``ACF('exp', ...)``, ``hurst`` = ``hurst_exponent``, ``q_r`` = ``roll_off_frequency`` [rad/m].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...core.contact import DC

__all__ = ["RoughnessSpec"]

_KINDS = ("acf", "psd", "measured")
_ACF_SHAPES = ("exponential", "gaussian")


# OPEN R4-05: no `body` field: the argument of rough_surface_pair says which body carries it;
# `composite` states that its heights are the composite roughness of the pair
@dataclass(**DC)
class RoughnessSpec:
    """Definition of the roughness of one surface (SI units); see the module for the definitions.

    Parameters
    ----------
    kind : {"acf", "psd", "measured"}
        ``"acf"``: Gaussian random surface with the ACF ``acf_shape``, ``rq``, ``beta_star``.
        ``"psd"``: self-affine random surface (``rq`` or ``psd_c0``, ``hurst``, ``q_r``; cut-off
        ``q_s = 2 pi / lambda_s``). ``"measured"``: the heights ``heights`` at spacing
        ``spacing``, or read from ``file`` (``slippy.surface.Surface(file_name=...)``: .txt, .csv,
        .al3d, .mat).
    rq : float, optional
        Target ``Rq`` [m] of a random surface (> 0, or 0 for a smooth surface: no heights).
        For ``"psd"`` give ``rq`` or ``psd_c0``, not both.
    acf_shape : {"exponential", "gaussian"}
        Shape of the ACF (``"acf"``). Default ``"exponential"``.
    acf_method : {"linear_transform"}
        SlipPY filter of an ACF surface (``RandomFilterSurface.linear_transform``). ``"fir_filter"``
        is refused (R2-17, see ``surfaces``). Default ``"linear_transform"``.
    beta_star : float, optional
        10 % correlation length ``beta*`` [m] (``"acf"``), of the unfiltered surface.
    hurst : float, optional
        Hurst exponent ``H`` in (0, 1] (``"psd"``).
    q_r : float, optional
        Roll-off wave number [rad/m] (``"psd"``), below ``q_s``; 0 for no roll-off (pure power
        law down to the lattice's lowest wave number).
    psd_c0 : float, optional
        PSD plateau ``C0`` [m^4] (``"psd"``, instead of ``rq``): converted to ``Rq`` by
        :meth:`target_rq` (the continuum integral, plateau down to q = 0, not from the lattice's
        lowest ``2 pi / L``): the generated surface has that ``Rq``, so its realised ``C0`` is not
        ``psd_c0`` exactly (OPEN R4-02).
    heights : ndarray, optional
        Measured heights [m], 2-D, array axis 0 along grid x (``"measured"``), no NaN (fill
        holes first, e.g. ``slippy.surface.Surface.fill_holes``).
    spacing : float, optional
        Spacing of ``heights`` [m] (``"measured"``; read from an .al3d file when None).
    file : str, optional
        File of a measured surface (``"measured"``, instead of ``heights``), heights in [m].
    lambda_s : float, optional
        S-filter cut-off [m]. Mandatory for ``"acf"`` and ``"psd"`` (R2-20: an unfiltered random
        surface does not converge with the grid). Optional for ``"measured"``.
    lambda_c : float, optional
        L-filter cut-off [m], > ``lambda_s``. Default None (only the mean plane removed).
    distribution : {"gaussian"}
        Height distribution. Only Gaussian (OPEN R4-03: non-Gaussian heights, e.g. SlipPY
        ``set_moments``, would be changed again by the band limit).
    seed : int, optional
        Seed of the realisation (random kinds, required). The numpy global state is restored
        after the draw.
    composite : bool
        True: the heights are the composite roughness of the pair (``Rq = sqrt(Rq1^2 + Rq2^2)``)
        carried by this body, the other smooth: the same gap, so the same frictionless normal
        contact of linear elastic half-spaces with small slopes (Muser 2017, section 2.1, PDF p. 3),
        not with friction, plasticity or tangential load. The two surfaces must be statistically
        independent; the composite PSD is ``C1 + C2``, so the composite ACF has the shape of
        neither unless both are equal. Default False.
    label : str
        Identifier in records and messages.
    """

    kind: str
    rq: float | None = None
    acf_shape: str = "exponential"
    acf_method: str = "linear_transform"
    beta_star: float | None = None
    hurst: float | None = None
    q_r: float | None = None
    psd_c0: float | None = None
    heights: np.ndarray | None = None
    spacing: float | None = None
    file: str | None = None
    lambda_s: float | None = None
    lambda_c: float | None = None
    distribution: str = "gaussian"
    seed: int | None = None
    composite: bool = False
    label: str = ""

    @property
    def random(self) -> bool:
        """True for the generated kinds (``"acf"``, ``"psd"``)."""
        return self.kind in ("acf", "psd")

    def target_rq(self) -> float:
        """Target ``Rq`` [m] of a random spec: ``rq``, or for ``"psd"`` with ``psd_c0`` the integral
        ``Rq^2 = 2 pi int_0^q_s C(q) q dq = 2 pi C0 q_r^2 [1/2 + (1 - (q_s/q_r)^(-2H)) / (2H)]``
        (the PSD of the module, plateau to q = 0; ``q_r = 0`` gives 0: give ``rq`` then)."""
        if self.rq is not None:
            return float(self.rq)
        q_s, q_r, H = 2.0 * math.pi / self.lambda_s, self.q_r, self.hurst
        return math.sqrt(2.0 * math.pi * self.psd_c0 * q_r**2 * (0.5 + (1.0 - (q_s / q_r) ** (-2.0 * H)) / (2.0 * H)))

    def validate(self) -> list[str]:
        """Return the list of error messages; empty if the spec is consistent."""
        e: list[str] = []
        name = f"RoughnessSpec {self.label!r}"

        def positive(key: str, required: bool = True, zero: bool = False) -> None:
            v = getattr(self, key)
            if v is None:
                if required:
                    e.append(f"{name}: {key} is required for kind {self.kind!r}")
            elif not (math.isfinite(v) and (v >= 0 if zero else v > 0)):
                e.append(f"{name}: {key} must be {'>= 0' if zero else '> 0'} and finite, got {v}")

        if self.kind not in _KINDS:
            return [f"{name}: kind must be one of {_KINDS}, got {self.kind!r}"]
        if self.distribution != "gaussian":
            e.append(f"{name}: distribution must be 'gaussian' (non-Gaussian heights: OPEN R4-03), "
                     f"got {self.distribution!r}")
        if self.random:
            # R2-20: without a short-wavelength cut-off area and p_max do not converge with h
            positive("lambda_s")
            if self.seed is None:
                e.append(f"{name}: seed is required for a random surface (reproducible realisation)")
            if any(getattr(self, k) is not None for k in ("heights", "spacing", "file")):
                e.append(f"{name}: heights, spacing and file are for kind 'measured' only")
        else:
            positive("lambda_s", required=False)
            if (self.heights is None) == (self.file is None):
                e.append(f"{name}: kind 'measured' needs exactly one of heights or file")
            if self.heights is not None:
                z = np.asarray(self.heights, dtype=float)
                if z.ndim != 2 or min(z.shape) < 2 or not np.all(np.isfinite(z)):
                    e.append(f"{name}: heights must be a finite 2-D array (fill holes first), got shape {z.shape}")
                positive("spacing")
            else:
                positive("spacing", required=False)       # an .al3d file carries its own spacing
            if any(getattr(self, k) is not None for k in ("rq", "beta_star", "hurst", "q_r", "psd_c0", "seed")):
                e.append(f"{name}: rq, beta_star, hurst, q_r, psd_c0 and seed are for the random kinds; a measured "
                         "surface keeps its own heights")
        if self.kind == "acf":
            positive("rq", zero=True)
            positive("beta_star")
            if self.acf_method == "fir_filter":
                e.append(f"{name}: acf_method 'fir_filter' refused (R2-17): SlipPY RandomFilterSurface.fir_filter "
                         "uses sqrt(fft2(acf)) of one ACF quadrant, complex and never transformed back, as the "
                         "spatial filter (slippy/surface/Random.py:752-757): ACF 0.65 for 0.11 at 10 nodes, Rq 78 "
                         "for 1 (probe C2); use 'linear_transform'")
            elif self.acf_method != "linear_transform":
                e.append(f"{name}: acf_method must be 'linear_transform', got {self.acf_method!r}")
            if self.acf_shape not in _ACF_SHAPES:
                e.append(f"{name}: acf_shape must be one of {_ACF_SHAPES}, got {self.acf_shape!r}")
            if any(getattr(self, k) is not None for k in ("hurst", "q_r", "psd_c0")):
                e.append(f"{name}: hurst, q_r and psd_c0 are for kind 'psd'")
        if self.kind == "psd":
            if (self.rq is None) == (self.psd_c0 is None):
                e.append(f"{name}: kind 'psd' needs exactly one of rq or psd_c0")
            positive("rq", required=False, zero=True)
            positive("psd_c0", required=False)
            positive("q_r", zero=True)
            if self.hurst is None or not 0.0 < self.hurst <= 1.0:
                e.append(f"{name}: hurst must be in (0, 1], got {self.hurst}")
            if self.beta_star is not None:
                e.append(f"{name}: beta_star is for kind 'acf'")
            if (self.q_r is not None and self.lambda_s is not None and self.lambda_s > 0
                    and not self.q_r < 2.0 * math.pi / self.lambda_s):
                e.append(f"{name}: q_r = {self.q_r} rad/m must be below q_s = 2 pi / lambda_s = "
                         f"{2.0 * math.pi / self.lambda_s} rad/m")
            if self.psd_c0 is not None and not self.q_r:
                e.append(f"{name}: psd_c0 needs q_r > 0 (the plateau); give rq for a pure power law")
        if self.lambda_c is not None:
            positive("lambda_c")
            if self.lambda_s is not None and not self.lambda_c > self.lambda_s:
                e.append(f"{name}: lambda_c = {self.lambda_c} m must be above lambda_s = {self.lambda_s} m")
        return e

    def validate_or_raise(self) -> None:
        """Raise ValueError with all messages of :meth:`validate` if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    def record(self) -> dict:
        """The scalars of the spec for a CSV row (``rq_target_m`` resolved; arrays and file left out
        except ``file``)."""
        return {"kind": self.kind, "label": self.label,
                "rq_target_m": self.target_rq() if self.random else math.nan,
                "acf_shape": self.acf_shape if self.kind == "acf" else "",
                **{f"{k}_m": math.nan if getattr(self, k) is None else float(getattr(self, k))
                   for k in ("beta_star", "lambda_s", "lambda_c")},
                "hurst": math.nan if self.hurst is None else float(self.hurst),
                "q_r_rad_per_m": math.nan if self.q_r is None else float(self.q_r),
                "seed": "" if self.seed is None else int(self.seed),
                "distribution": self.distribution, "composite": bool(self.composite),
                "file": self.file or ""}
