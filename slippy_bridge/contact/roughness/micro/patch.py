"""The periodic, nominally flat rough patch of the micro scale: :class:`PatchSpec` and
:func:`periodic_patch`. Constructions only: no solve here (see ``README.md`` of this folder).

The patch is a square of side ``L``, ``n`` nodes along each axis, periodic along both axes: one
cell of an infinite rough surface pressed on a flat. Its heights are those of
:func:`..macro.surfaces.rough_heights` on the patch's own grid, which draws every random surface
periodic on an even square. Both bodies are SlipPY ``Elastic`` materials with
``zero_frequency_value = 0``, what SlipPY's ``rey`` solver (the ``mean_gap`` control) requires of
both (``HeightOptimisationFunction.rey``, ``slippy/contact/_step_utils.py:352-353``; the flag is
set at 141-144 only when both are ``_IMMaterial``). The flat is elastic by choice: the frictionless
normal contact depends only on ``E*`` (``PatchSpec.e_star``). SlipPY ``Rigid`` (an ``_IMMaterial``
with zero-frequency value 0, ``slippy/core/materials.py:379, 398``) would also run under ``rey``
(slippy-expert R4b, scratch check: twice the pressure at the same mean gap, as ``E*`` predicts);
a rigid flat option is OPEN R4-11 (create ``Rigid(f"{label}_flat")``, never the shared ``rigid``). Under a non-zero mean pressure the mean displacement of a periodic half-space is not
defined: with the zero-frequency term 0 there is no absolute approach, only the mean gap (SB-61).
A later solve through ``solvers.solve`` / ``surface_pair`` would rebuild the materials with the
default zero-frequency term: the patch is solved from these surfaces (OPEN R4-11).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import slippy.contact as c
import slippy.surface as s

from ...core.contact import DC
from ...mesh.grid import Grid
from ..macro.spec import RoughnessSpec
from ..macro.surfaces import RoughHeights, rough_heights

__all__ = ["PatchSpec", "periodic_patch", "L_OVER_LAMBDA_MIN"]

# [-] RVE rule (proposal, OPEN R4-07): the patch holds at least 5 of its largest wavelength, the ratio of the
# contact-mechanics challenge, L / lambda_r = 0.1 mm / 20 um (Muser 2017, section 2, Eq. (1), PDF p. 3). At
# L / lambda = 4 the wave vectors below 2 pi / L would carry about 3 % of the variance of an exponential or a
# Gaussian ACF (physics review R4b); the Rq rescaling hides that loss.
L_OVER_LAMBDA_MIN = 5.0


@dataclass(**DC)
class PatchSpec:
    """A periodic rough patch: size, grid, roughness, materials and the nominal pressures of later use.

    Parameters
    ----------
    length : float
        Side ``L`` of the square patch [m]; ``h = L / n`` must be a power of two (``Grid``, the
        SlipPY node count), so choose ``L = n 2^k``.
    n : int
        Nodes along each axis (even).
    roughness : RoughnessSpec
        A random kind (``"acf"`` or ``"psd"``): a measured surface is not periodic.
    E_1, nu_1 : float
        Rough body: Young's modulus [Pa], Poisson's ratio [-].
    E_2, nu_2 : float, optional
        Flat counter-face; default the rough body's. Both elastic (SlipPY ``rey``, see module).
        Only ``E*`` (:attr:`e_star`) matters for the frictionless normal contact with small slopes
        (Muser 2017, section 2.1, PDF p. 3): a comparison with the challenge compares ``p / E*``.
    pressure_range : (float, float), optional
        Nominal pressures [Pa] a later contact-law run sweeps (``p_nom = load / L^2``).
    label : str
        Prefix of the SlipPY material names (process-wide: keep it unique).
    """

    length: float
    n: int
    roughness: RoughnessSpec
    E_1: float
    nu_1: float
    E_2: float | None = None
    nu_2: float | None = None
    pressure_range: tuple[float, float] | None = None
    label: str = "patch"

    @property
    def h(self) -> float:
        """Grid spacing ``L / n`` [m]."""
        return self.length / self.n

    @property
    def e_star(self) -> float:
        """Contact modulus ``1/E* = (1 - nu_1^2)/E_1 + (1 - nu_2^2)/E_2`` [Pa] (steel on steel:
        ``E / (2 (1 - nu^2))``, 115 GPa for 210 GPa, 0.3)."""
        E_2 = self.E_1 if self.E_2 is None else self.E_2
        nu_2 = self.nu_1 if self.nu_2 is None else self.nu_2
        return 1.0 / ((1.0 - self.nu_1**2) / self.E_1 + (1.0 - nu_2**2) / E_2)

    @property
    def lambda_largest(self) -> float:
        """Largest wavelength of the roughness [m]: ``lambda_c`` if given; else for ``"psd"`` the
        roll-off ``2 pi / q_r`` (``L`` with ``q_r = 0``: a pure power law always fails the RVE
        rule); for ``"acf"`` ``2 pi beta* / 2.3``: the knee ``q xi = 1`` of the PSD of the
        exponential ACF, ``C ~ [1 + (q xi)^2]^(-3/2)``, ``xi = beta*/2.3``. The Gaussian ACF has no
        plateau (``C ~ exp(-q^2 l^2/4)``, ``l = beta*/sqrt(2.3)``); the same length is kept for it,
        where it also leaves about 3 % of the variance below ``2 pi / L`` at ``L / lambda = 4``."""
        r = self.roughness
        if r.lambda_c is not None:
            return r.lambda_c
        if r.kind == "psd":
            return 2.0 * math.pi / r.q_r if r.q_r else self.length
        return 2.0 * math.pi * r.beta_star / 2.3

    @property
    def l_over_lambda(self) -> float:
        """``L / lambda_largest`` [-], checked against ``L_OVER_LAMBDA_MIN``."""
        return self.length / self.lambda_largest

    def validate(self) -> list[str]:
        """Return the list of error messages; empty if the spec is consistent (the spacing is
        checked by ``Grid``, the roughness by ``rough_heights``)."""
        e: list[str] = []
        name = f"PatchSpec {self.label!r}"
        if not (self.length > 0 and math.isfinite(self.length)):
            e.append(f"{name}: length must be positive and finite [m], got {self.length}")
        if not (isinstance(self.n, int) and self.n >= 8 and self.n % 2 == 0):
            e.append(f"{name}: n must be an even int >= 8, got {self.n!r}")
        if not self.roughness.random:
            e.append(f"{name}: the roughness must be random ('acf' or 'psd'): a measured surface is not periodic")
        for key in ("E_1", "E_2"):
            v = getattr(self, key)
            if (v is not None or key == "E_1") and not (v is not None and v > 0 and math.isfinite(v)):
                e.append(f"{name}: {key} must be positive and finite [Pa], got {v}")
        for key in ("nu_1", "nu_2"):
            v = getattr(self, key)
            if (v is not None or key == "nu_1") and not (v is not None and 0.0 < v < 0.5):
                e.append(f"{name}: {key} must be in (0, 0.5), got {v}")
        if self.pressure_range is not None:
            lo, hi = self.pressure_range
            if not 0.0 < lo < hi < math.inf:
                e.append(f"{name}: pressure_range must be 0 < p_min < p_max [Pa], got {self.pressure_range}")
        return e

    def validate_or_raise(self) -> None:
        """Raise ValueError with all messages of :meth:`validate` if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


def periodic_patch(spec: PatchSpec):
    """The two SlipPY surfaces of the periodic patch ``spec``, ready for a fully periodic step.

    For a later solve: ``StaticStep(..., periodic_geometry=True, periodic_axes=(True, True),
    method="rey", mean_gap=...)`` or a load ``p_nom L^2`` (``rey`` needs both axes periodic and
    both zero-frequency values 0, ``_step_utils.py:352-355``).

    Returns
    -------
    surface_1, surface_2 : slippy Surface
        The rough body (heights ``heights.z``) and the flat (zeros), each with an ``Elastic``
        material of ``zero_frequency_value = 0`` (frequency-domain kernel), spacing ``h``.
    heights : RoughHeights
        Heights and the grid rule (``lambda_s / h``, ``beta* / h``).

    Raises
    ------
    ValueError
        From ``spec.validate_or_raise``, ``Grid`` (``h`` not a power of two) and ``rough_heights``.

    Warns
    -----
    UserWarning
        When the patch fails the RVE rule (``L / lambda_largest < L_OVER_LAMBDA_MIN``), with the
        numbers; the grid rule of the heights warns from :func:`rough_heights`.
    """
    spec.validate_or_raise()
    n, h = spec.n, spec.h
    x = (np.arange(n) - n // 2) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    rh: RoughHeights = rough_heights(spec.roughness, Grid(X=X, Y=Y, h=h))
    if spec.l_over_lambda < L_OVER_LAMBDA_MIN:
        warnings.warn(f"{spec.label}: L / lambda_largest = {spec.l_over_lambda:.3g} < {L_OVER_LAMBDA_MIN:g} "
                      f"(L {spec.length:.4g} m, lambda {spec.lambda_largest:.4g} m): the patch is not a "
                      "representative cell (OPEN R4-07)", stacklevel=2)
    out = []
    for k, (z, E, nu) in enumerate(((rh.z, spec.E_1, spec.nu_1),
                                    (np.zeros((n, n)), spec.E_1 if spec.E_2 is None else spec.E_2,
                                     spec.nu_1 if spec.nu_2 is None else spec.nu_2)), start=1):
        surface = s.Surface(profile=z, grid_spacing=h)
        surface.material = c.Elastic(f"{spec.label}_body{k}", {"E": E, "v": nu}, use_frequency_domain=True,
                                     zero_frequency_value=0.0)
        out.append(surface)
    return out[0], out[1], rh
