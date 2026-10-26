"""The two SlipPY surfaces of a contact on its grid: each body's exact profile with its material.

References
----------
SlipPY ``slippy.surface.Surface`` (a discrete profile and its grid spacing) and
``slippy.contact.Elastic`` (``slippy/core/elastic_material.py``, ``slippy/core/materials.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import slippy.contact as c
import slippy.surface as s

if TYPE_CHECKING:
    from ..core.contact import Contact
    from ..mesh.grid import Grid

__all__ = ["surface_pair"]


# OPEN SB-08: two exact profiles only; composite build, roughness and given surfaces left for José's ruling
def surface_pair(contact: Contact, grid: Grid, *, name: str, use_frequency_domain: bool = True):
    """SlipPY surfaces of body 1 and body 2 of ``contact`` on ``grid``, ready for ``ContactModel``.

    Each body is its exact profile (``Contact.profiles``: ``-sag`` [m], apex 0, body 2 turned by
    ``contact.angle``) on the grid nodes, with an ``Elastic`` material of its own ``E`` [Pa] and
    ``nu``. Both surfaces are discrete, of the grid's shape and spacing ``h``.

    Parameters
    ----------
    contact : Contact
    grid : Grid
        From ``Grid.around``; rolling = array axis 0.
    name : str
        Unique in the process (SlipPY material names are process-wide); the materials are
        ``<name>_body1`` and ``<name>_body2``. ``solve`` passes its model name.
    use_frequency_domain : bool, optional
        How SlipPY defines the influence matrix of both bodies (one choice: SlipPY combines the two
        matrices). True (SlipPY default): in the frequency domain; the kernel is then periodic and
        a non-periodic contact also feels images of its load repeated every twice the domain.
        False: the half-space (Boussinesq) kernel sampled in space, zero-padded on a non-periodic
        axis (no images); on a periodic axis it spans one period only.

    Returns
    -------
    surface_1, surface_2 : slippy Surface

    Raises
    ------
    ValueError
        For a line contact with ``use_frequency_domain=False``: on the periodic (line) axis the
        spatial kernel spans one period, so the load is a strip ``n_y h`` long, not plane strain
        (on the 10 x 10 mm roller at Q 1000 N with n_y 16: p0 +20 %, b -15 %); and from ``Body.profile`` if the grid reaches
        beyond a body.
    """
    if contact.line and not use_frequency_domain:
        raise ValueError("a line contact needs use_frequency_domain=True "
                         "(the spatial kernel is a finite strip, not plane strain)")
    z1, z2 = contact.profiles(grid)
    out = []
    for k, (z, body) in enumerate(((z1, contact.body1), (z2, contact.body2)), start=1):
        surface = s.Surface(profile=z, grid_spacing=grid.h)
        surface.material = c.Elastic(f"{name}_body{k}", {"E": body.E, "v": body.nu},
                                     use_frequency_domain=use_frequency_domain)
        out.append(surface)
    return tuple(out)
