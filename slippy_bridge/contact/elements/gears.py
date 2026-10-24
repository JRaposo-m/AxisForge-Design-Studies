"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Gears: AxisForge cylindrical pair + ``contact_inputs`` record -> one line :class:`Contact` per path point.

The cylindrical pair (spur, helical, internal: one ``GearPair`` class, its subtype is the family,
as for a bearing) is translated by :func:`gear_contacts`. AxisForge already gives, per step of the contact point along the
path, the two equivalent cylinders, the surface speeds and the load per unit length
(``axisforge.solvers.machine_elements.gears.contact.contact_inputs``); this module is the ONLY
place where that record is converted to SI (mm -> m, N/mm -> N/m, MPa -> Pa), as the bearing
adapter does for bearings (D-0019). The input checks are real; the conversion body is a scaffold.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from axisforge.core.machine_elements.gears import GearPair
from axisforge.results._base import DC

from ..base import Contact

__all__ = ["GearLocation", "gear_contacts"]

# AxisForge ``CylindricalFamily.path_points`` gives only A..E (start, limits of single contact, pitch
# point, end); B-, B+, D-, D+ are bridge labels for the one-sided limits either side of B and D (the
# load share jumps there), built by ``construction.path_times`` (GL-09, GL-25).
_POINTS = ("A", "B-", "B+", "C", "D-", "D+", "E")


@dataclass(**DC)
class GearLocation:
    """Where one contact of a gear pair is, on the path of contact, and the load sharing there.

    Field names and units are the keys of AxisForge ``contact_inputs``, except ``driver``
    (``axisforge.solvers.machine_elements.gears.contact``): ``s_mm``, ``t_s``, ``g_y_mm`` from
    ``CylindricalFamily.path_points``; ``share``, ``l_mm``, ``l_sum_mm``, ``n_pairs`` from
    ``load_sharing``. They explain ``case.load`` (``w = share F_bn / l``).

    Parameters
    ----------
    point : str
        Label of the step: one of ``("A", "B-", "B+", "C", "D-", "D+", "E")`` or ``"s=<s_mm>"``.
    s_mm : float
        Distance travelled by the contact point from the start of meshing A along the path [mm].
    t_s : float
        Time since A [s].
    g_y_mm : float
        Position from the pitch point C on the path, positive toward the tip of gear 1 [mm].
    share : float
        Share of the normal force carried by this tooth pair [-].
    l_mm : float
        Length of the contact line of this tooth pair [mm].
    l_sum_mm : float
        Total length of the contact lines of all pairs in contact [mm].
    n_pairs : int
        Number of tooth pairs in contact [-].
    driver : str
        Driving gear, ``"gear1"`` or ``"gear2"`` (the ``driver`` argument of ``contact_inputs``).
    """

    point: str
    s_mm: float
    t_s: float
    g_y_mm: float
    share: float
    l_mm: float
    l_sum_mm: float
    n_pairs: int
    driver: str


def gear_contacts(pair: GearPair, inputs: Mapping[str, np.ndarray], *,
                  points: Sequence[str] | None = None,
                  E: tuple[float | None, float | None] | None = None,
                  nu: tuple[float | None, float | None] | None = None) -> list[Contact]:
    """One line :class:`Contact` per step of an AxisForge ``contact_inputs`` record (SI units).

    Physically, each step is the contact of the two flanks at one position of the contact point
    on the line of action, at one instant of the mesh cycle: two parallel cylinders of the local
    normal radii of curvature, pressed by the load per unit length that the tooth pair carries
    there (load sharing between the tooth pairs in contact), with the surface speeds at that
    point. The contact is quasi-steady: each step is solved as if it lasted for ever.

    Parameters
    ----------
    pair : GearPair
        AxisForge pair of the cylindrical family, the one ``inputs`` was computed for. E and nu of
        each gear come from its ``material_id`` registered in ``axisforge.core.materials``.
    inputs : mapping of str to ndarray
        The output of AxisForge ``contact_inputs(pair, driver, n1_rpm, F_bn_N, ...)``, arrays of
        equal length, one entry per step, AxisForge units: ``s_mm``, ``t_s`` [s], ``g_y_mm``,
        ``rho1_n_mm``, ``rho2_n_mm`` (signed: convex > 0, concave < 0) [mm], ``u1_m_per_s``,
        ``u2_m_per_s``, ``vg_m_per_s``, ``ue_m_per_s`` [m/s], ``share`` [-], ``l_mm``,
        ``l_sum_mm`` [mm], ``n_pairs`` [-], ``w_N_per_mm`` [N/mm].
    points : sequence of str, optional
        One label per step, unique, from ``("A", "B-", "B+", "C", "D-", "D+", "E")``. None
        labels each step ``f"s={s_mm:.6g}"`` (distance from A [mm], 6 significant digits). The
        case label is ``f"{pair.label or 'gear'}_{point}"``: it is used in file names, so no ``:``
        (an NTFS stream separator).
    E, nu : (value or None, value or None), optional
        Young's modulus [MPa] and Poisson's ratio [-] of gear 1 and gear 2, given only for a gear
        without an AxisForge material (empty ``material_id``); None for a gear that has one.

    Returns
    -------
    list of Contact
        One per step, in the order of ``inputs``, with ``element="gear"`` and
        ``subtype="CylindricalFamily"``:

        * ``case`` = ``ContactCase(label=f"{pair.label or 'gear'}_{point}",
          r1=(rho1_n_mm*_MM, inf), r2=(rho2_n_mm*_MM, inf), E=(E1*_MPA, E2*_MPA), nu=(nu1, nu2),
          load=w_N_per_mm/_MM, line=True)`` (N/mm -> N/m): radii [m] (``x`` = rolling direction =
          profile direction; the line is infinite along the face width, ``y``), E [Pa], load [N/m];
        * ``location`` = :class:`GearLocation` of the step;
        * ``kinematics`` = ``Kinematics(ue_m_per_s, vg_m_per_s, u1_m_per_s, u2_m_per_s)`` copied
          from ``inputs``: ``ue`` the MEAN entrainment speed (u1+u2)/2, ``vg`` the sliding speed
          u1 - u2, signed and referenced to gear 1 (AxisForge ``CylindricalFamily.surface_speeds``).

    Raises
    ------
    ValueError
        If ``points`` does not have one label per step, or a label is repeated or not one of
        the named points; if any ``rho1_n_mm <= 0`` (the flank of gear 1 is convex on a valid
        path; a pair assembled with ``check=False`` can give rho <= 0 at the ends, R-030); if
        the sign of any ``rho2_n_mm`` is not that of ``pair.gear2.z`` (convex external gear 2,
        concave internal ring; R-030); if a gear has an empty ``material_id`` and no ``E``,
        ``nu`` given for it (R-025), or has a material and ``E`` or ``nu`` given for it; if ``E``
        or ``nu`` is not a 2-sequence (one entry per gear). A load
        ``w <= 0`` or ``rho2 = 0`` is refused by ``ContactCase``.
    NotImplementedError
        Always, after the checks: the conversion is a scaffold (OPEN R-023, R-031).

    Notes
    -----
    Assumptions: smooth elastic half-spaces, local equivalent cylinders (no tip relief, no edge
    contact), uniform load per unit length (AxisForge rigid load sharing), steady speed, no
    friction. Helical pairs: only the mid-face transverse section is recorded (AxisForge
    ``contact_inputs`` Notes).

    References
    ----------
    Johnson, K. L. (1985). *Contact Mechanics*, ch. 4 (line contact of cylinders).
    ISO 21771:2007, 5.4.3-5.4.8 (radii of curvature along the path), 5.6 (speeds).
    """
    # OPEN R-023: convert once here (rho mm->m, w N/mm->N/m, E MPa->Pa); body not written.
    # OPEN R-024: rho2_n of a ring gear is concave < 0 (probe at C); José may close it.
    # OPEN R-027: helical pairs are mid-face slices only; EHL on beta != 0 to be refused.
    # OPEN R-031: signs of u1, u2 for internal pairs (u < 0) before forming ue and vg.
    # OPEN R-032: local equivalent cylinders, not the involute flank surfaces.
    # OPEN R-118: E, nu of the FZG gears (20MnCr5) come from get_material, not from the papers.
    # OPEN: contact_inputs does not record the driver, and gear_contacts has no driver argument
    # to fill GearLocation.driver from.
    # OPEN GL-29: an empty material_id is refused below, before get_material('') raises KeyError.
    # OPEN R-025: get_material raises KeyError for an unregistered id; the case registers steel.
    tag = pair.label or "gear"
    n_steps = len(inputs["t_s"])
    if points is not None:
        points = list(points)
        if len(points) != n_steps:
            raise ValueError(f"{tag}: points has {len(points)} labels for {n_steps} steps of inputs")
        if len(set(points)) != len(points):
            raise ValueError(f"{tag}: points labels are not unique: {points}")
        bad = [p for p in points if p not in _POINTS]
        if bad:
            raise ValueError(f"{tag}: points labels {bad} are not in {_POINTS}")
    labels = points if points is not None else [f"s={s:.6g}" for s in inputs["s_mm"]]
    # labels name ContactCase.label, which goes into file names: "_", never ":" (NTFS stream)
    # the flank of gear 1 is convex on the involute between the base-tangent points (R-030)
    for label, rho1 in zip(labels, np.asarray(inputs["rho1_n_mm"], dtype=float)):
        if not rho1 > 0:
            raise ValueError(f"{tag}_{label}: rho1_n_mm must be > 0, got {rho1} (pair assembled with check=False?)")
    # gear 2 is convex (rho2 > 0) on an external pair, z2 > 0, and concave (rho2 < 0) on an internal
    # ring, z2 < 0 (ISO 21771 sign of z); a check=False external pair can give rho2 < 0 at the ends (R-030)
    sign_z2 = np.sign(pair.gear2.z)
    for label, rho2 in zip(labels, np.asarray(inputs["rho2_n_mm"], dtype=float)):
        if np.sign(rho2) != sign_z2:
            raise ValueError(f"{tag}_{label}: rho2_n_mm = {rho2} has not the sign of gear2.z = {pair.gear2.z} "
                             f"(pair assembled with check=False? R-030)")
    # E, nu per gear: from its AxisForge material, or given here only for a gear that has none
    for key, value in (("E", E), ("nu", nu)):
        if value is not None and not (isinstance(value, (tuple, list)) and len(value) == 2):
            raise ValueError(f"{tag}: {key} must be (gear1, gear2), each a value or None, got {value!r}")
    E, nu = (None, None) if E is None else tuple(E), (None, None) if nu is None else tuple(nu)
    for i, name in enumerate(("gear1", "gear2")):
        given = E[i] is not None or nu[i] is not None
        if getattr(pair, name).material_id == "":
            if E[i] is None or nu[i] is None:
                raise ValueError(f"{tag}: {name}.material_id is empty and E, nu are not given for it; "
                                 "E and nu come from get_material (R-025)")
        elif given:
            raise ValueError(f"{tag}: {name} has material_id {getattr(pair, name).material_id!r}; "
                             "E and nu are given only for a gear without one")
    raise NotImplementedError("gear contacts: scaffold, OPEN R-023 R-031")
