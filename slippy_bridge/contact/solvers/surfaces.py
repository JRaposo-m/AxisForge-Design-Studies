"""The two SlipPY surfaces of a contact on its grid: each body's exact profile, plus its rough
heights when given, with its material.

References
----------
SlipPY ``slippy.surface.Surface`` (a discrete profile and its grid spacing) and
``slippy.contact.Elastic`` (``slippy/core/elastic_material.py``, ``slippy/core/materials.py``).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import slippy.contact as c
import slippy.surface as s

if TYPE_CHECKING:
    from ..core.contact import Contact
    from ..mesh.grid import Grid
    from ..roughness.macro.surfaces import RoughHeights

__all__ = ["surface_pair"]


# OPEN SB-08: exact profiles plus given heights (R4c); composite build and given surfaces left for José's ruling
def surface_pair(contact: Contact, grid: Grid, *, name: str, use_frequency_domain: bool = True,
                 heights: tuple[RoughHeights | None, RoughHeights | None] | None = None):
    """SlipPY surfaces of body 1 and body 2 of ``contact`` on ``grid``, ready for ``ContactModel``.

    Each body is its exact profile (``Contact.profiles``: ``-sag`` [m], apex 0, body 2 turned by
    ``contact.angle``, both turned by ``-grid.angle`` into the grid frame) on the grid nodes, plus
    the rough heights of that body when given, with an ``Elastic`` material of its own ``E`` [Pa]
    and ``nu``. Both surfaces are discrete, of the grid's shape and spacing ``h``.

    The undeformed gap is ``-(z1 + z2)``: a height adds to its body and the two roughnesses add in
    the gap. Two random specs with the same statistics and seed are the same heights on both
    bodies (fully correlated): give different seeds for independent surfaces. SlipPY shifts the gap
    to a minimum of 0 (first touch, ``_model_utils.py:173-175``): with roughness that is the first
    asperity touch, not the smooth apex (see ``solve``, ``delta_mean_planes``).

    Parameters
    ----------
    contact : Contact
    grid : Grid
        From ``Grid.around``; grid x = array axis 0 (rolling when ``grid.angle = 0``).
    name : str
        Unique in the process (SlipPY material names are process-wide); the materials are
        ``<name>_body1`` and ``<name>_body2``. ``solve`` passes its model name.
    use_frequency_domain : bool, optional
        How SlipPY defines the influence matrix of both bodies (one choice: SlipPY combines the two
        matrices). True (SlipPY default): in the frequency domain; the kernel is then periodic and
        a non-periodic contact also feels images of its load repeated every twice the domain.
        False: the half-space (Boussinesq) kernel sampled in space, zero-padded on a non-periodic
        axis (no images); on a periodic axis it spans one period only.
    heights : (RoughHeights or None, RoughHeights or None), optional
        Rough heights of body 1 and body 2 (``roughness.macro.rough_heights`` on this grid), each
        added to its body's profile; None (default) or a None entry: that body is its exact
        profile, untouched. A ``composite`` spec carries the roughness of both bodies, so the other
        entry must be None. Heights all zero (a random spec of ``Rq = 0``) leave the profile as is.

    Returns
    -------
    surface_1, surface_2 : slippy Surface

    Raises
    ------
    ValueError
        For a line contact with ``use_frequency_domain=False``: on the periodic (line) axis the
        spatial kernel spans one period, so the load is a strip ``n_y h`` long, not plane strain
        (on the 10 x 10 mm roller at Q 1000 N with n_y 16: p0 +20 %, b -15 %); and from ``Body.profile`` if the grid reaches
        beyond a body. With ``heights``: if they are not two entries, if any is not on this grid
        (shape or spacing ``h``), if a composite spec has a second one, or for a line contact if
        any is not periodic along y with the grid's period (``RoughHeights.periodic_y``: the strip
        is one period of an infinite contact).
    """
    if contact.line and not use_frequency_domain:
        raise ValueError("a line contact needs use_frequency_domain=True "
                         "(the spatial kernel is a finite strip, not plane strain)")
    heights = (None, None) if heights is None else tuple(heights)
    _check_heights(heights, contact, grid)
    z1, z2 = contact.profiles(grid)
    out = []
    for k, (z, body, rh) in enumerate(zip((z1, z2), (contact.body1, contact.body2), heights), start=1):
        if rh is not None and np.any(rh.z):
            z = z + rh.z
        surface = s.Surface(profile=z, grid_spacing=grid.h)
        surface.material = c.Elastic(f"{name}_body{k}", {"E": body.E, "v": body.nu},
                                     use_frequency_domain=use_frequency_domain)
        out.append(surface)
    return tuple(out)


def _check_heights(heights: tuple, contact: Contact, grid: Grid) -> None:
    """Refuse heights that are not of this grid or not compatible with the contact (see ``surface_pair``)."""
    if len(heights) != 2:
        raise ValueError(f"surface_pair: heights must be (heights of body 1, heights of body 2), got {len(heights)}")
    given = [rh for rh in heights if rh is not None]
    for k, rh in enumerate(heights, start=1):
        if rh is None:
            continue
        if np.shape(rh.z) != grid.X.shape or not math.isclose(rh.h, grid.h, rel_tol=1e-9):
            raise ValueError(f"surface_pair: heights of body {k} are on a grid of shape {np.shape(rh.z)} and spacing "
                             f"{rh.h} m, the solve grid is {grid.X.shape} with h {grid.h} m: make them with "
                             "rough_heights on this grid")
        if contact.line and not rh.periodic_y:
            raise ValueError(f"surface_pair: a line contact is periodic along y; the heights of body {k} are not "
                             "(an 'acf' spec, or a 'psd' spec on an even square lattice, of the grid's length along y)")
    if len(given) == 2 and any(rh.spec.composite for rh in given):
        raise ValueError("surface_pair: a composite spec carries the roughness of both bodies; "
                         "the other body must have no heights")
