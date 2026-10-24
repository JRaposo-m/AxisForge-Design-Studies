"""Gears: AxisForge cylindrical pair + ``contact_inputs`` record -> one line :class:`Contact` per path point.

The cylindrical pair (spur, helical, internal: one ``GearPair`` class, its subtype is the family,
as for a bearing) is translated by :func:`gear_contacts`. AxisForge already gives, per step of the
contact point along the path, the two equivalent cylinders, the surface speeds and the load per
unit length (``axisforge.solvers.machine_elements.gears.contact.contact_inputs``); this module is
the ONLY place where that record is converted to SI (mm -> m, N/mm -> N/m, MPa -> Pa), as the
bearing module does for bearings (D-0019).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from axisforge.core.machine_elements.gears import GearPair
from axisforge.core.materials import available_materials, get_material
from axisforge.results._base import DC

from ..base import Contact, ContactCase, Kinematics
from . import _MM, _MPA

__all__ = ["GearLocation", "gear_contacts"]

# AxisForge ``CylindricalFamily.path_points`` gives only A..E (start, limits of single contact, pitch
# point, end); B-, B+, D-, D+ are bridge labels for the one-sided limits either side of B and D (the
# load share jumps there), built by ``construction.path_times`` (GL-09, GL-25).
_POINTS = ("A", "B-", "B+", "C", "D-", "D+", "E")


@dataclass(**DC)
class GearLocation:
    """Where one contact of a gear pair is, on the path of contact, and the load sharing there.

    Field names and units are the keys of AxisForge ``contact_inputs``, except ``driver``,
    ``material1`` and ``material2`` (``axisforge.solvers.machine_elements.gears.contact``):
    ``s_mm``, ``t_s``, ``g_y_mm`` from ``CylindricalFamily.path_points``; ``share``, ``l_mm``,
    ``l_sum_mm``, ``n_pairs`` from ``load_sharing``. They explain ``case.load``
    (``w = share F_bn / l``).

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
    material1, material2 : str
        Where E and nu of gear 1 and gear 2 come from: the AxisForge ``material_id`` when it is
        registered; ``"given"`` (empty ``material_id``) or ``"given:<material_id>"`` (an id that is
        not registered, so that it stays visible) for the ``E``, ``nu`` arguments of
        :func:`gear_contacts`.
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
    material1: str
    material2: str


def gear_contacts(pair: GearPair, inputs: Mapping[str, np.ndarray], *, driver: str,
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
        each gear come from its ``material_id`` registered in ``axisforge.core.materials``
        (``get_material``), unless given below.
    inputs : mapping of str to ndarray
        The output of AxisForge ``contact_inputs(pair, driver, n1_rpm, F_bn_N, ...)``, arrays of
        equal length, one entry per step, AxisForge units: ``s_mm``, ``t_s`` [s], ``g_y_mm``,
        ``rho1_n_mm``, ``rho2_n_mm`` (signed: convex > 0, concave < 0) [mm], ``u1_m_per_s``,
        ``u2_m_per_s``, ``vg_m_per_s``, ``ue_m_per_s`` [m/s], ``share`` [-], ``l_mm``,
        ``l_sum_mm`` [mm], ``n_pairs`` [-], ``w_N_per_mm`` [N/mm].
    driver : {"gear1", "gear2"}
        The driver ``inputs`` was computed for. AxisForge ``contact_inputs`` does not return it and
        the pair does not store it, so the bridge trusts this keyword; it fixes the meaning of
        A..E. Copied into every location.
    points : sequence of str, optional
        One label per step, unique, from ``("A", "B-", "B+", "C", "D-", "D+", "E")``. None
        labels each step ``f"s={s_mm:.6g}"`` (distance from A [mm], 6 significant digits). The
        case label is ``f"{pair.label or 'gear'}_{point}"``: it is used in file names, so no ``:``
        (an NTFS stream separator).
    E, nu : (value or None, value or None), optional
        Young's modulus [MPa] and Poisson's ratio [-] of gear 1 and gear 2, given only for a gear
        without an AxisForge material (``material_id`` empty or not registered); None for a gear
        that has one. Recorded in ``material1``/``material2`` (see :class:`GearLocation`).

    Returns
    -------
    list of Contact
        One per step, in the order of ``inputs``, with ``element="gear"`` and
        ``subtype="CylindricalFamily"``:

        * ``case`` = ``ContactCase(label=f"{pair.label or 'gear'}_{point}",
          r1=(rho1_n_mm*_MM, inf), r2=(rho2_n_mm*_MM, inf), E=(E1*_MPA, E2*_MPA), nu=(nu1, nu2),
          load=w_N_per_mm/_MM, line=True)`` (N/mm -> N/m): radii [m] (``x`` = rolling direction =
          profile direction; the line is infinite along the face width, ``y``), E [Pa], load [N/m];
        * ``location`` = :class:`GearLocation` of the step (``n_pairs`` as int);
        * ``kinematics`` = ``Kinematics(ue_m_per_s, vg_m_per_s, u1_m_per_s, u2_m_per_s)`` copied
          from ``inputs``: ``ue`` the MEAN entrainment speed (u1+u2)/2, ``vg`` the sliding speed
          u1 - u2, signed and referenced to gear 1 (AxisForge ``CylindricalFamily.surface_speeds``).

    Raises
    ------
    ValueError
        If ``driver`` is not "gear1" or "gear2"; if ``points`` does not have one label per step,
        or a label is repeated or not one of the named points; if any ``rho1_n_mm <= 0`` (the
        flank of gear 1 is convex on a valid path; a pair assembled with ``check=False`` can give
        rho <= 0 at the ends, R-030); if
        the sign of any ``rho2_n_mm`` is not that of ``pair.gear2.z`` (convex external gear 2,
        concave internal ring; R-030); if a gear has no AxisForge material and no ``E``,
        ``nu`` given for it, or has one and ``E`` or ``nu`` given for it; if ``E`` or
        ``nu`` is not a 2-sequence (one entry per gear). A load ``w <= 0`` or ``rho2 = 0`` is
        refused by ``ContactCase``.

    Notes
    -----
    Units: AxisForge in (mm, N/mm, MPa, m/s), SI out (m, N/m, Pa, m/s). The speeds are copied, not
    recomputed: AxisForge gives them in m/s, signed, with the sign of a ring gear already applied.

    Load: ``case.load`` is the load per unit length w of this tooth pair. The static step
    (``surfaces.Grid.step_args``) multiplies it by the periodic strip ``n_y h`` of its grid, the
    ``length_m`` of the record; ``w * length_m`` is the load on that strip, not ``F_bn``, nor the
    load over ``l_mm`` or ``l_sum_mm``.

    Assumptions: smooth elastic half-spaces, local equivalent cylinders (no tip relief, no edge
    contact), uniform load per unit length (AxisForge rigid load sharing), steady speed, no
    friction. Helical pairs: only the mid-face transverse section is recorded (AxisForge
    ``contact_inputs`` Notes); whether that section stands for the whole contact line is an
    open question.

    References
    ----------
    Johnson1985, ch. 4 (line contact of two cylinders). The radii, speeds and load sharing are
    AxisForge's: ``CylindricalFamily.curvature_radii``, ``CylindricalFamily.surface_speeds``,
    ``load_sharing`` and ``contact_inputs`` (ISO_21771_2007 for the geometry).
    """
    # OPEN R-024: rho2_n of a ring gear is concave < 0 (probe at C); José may close it.
    # OPEN R-027: helical pairs are mid-face slices only; EHL on beta != 0 to be refused.
    # OPEN R-032: local equivalent cylinders, not the involute flank surfaces.
    # OPEN R-118: E, nu of the FZG gears (20MnCr5): no held source; the case gives them explicitly.
    tag = pair.label or "gear"
    if driver not in ("gear1", "gear2"):
        raise ValueError(f"{tag}: driver must be 'gear1' or 'gear2', got {driver!r}")
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
    E_MPa, nu_gear, source = [], [], []
    for i, name in enumerate(("gear1", "gear2")):
        material_id = getattr(pair, name).material_id
        given = E[i] is not None or nu[i] is not None
        if material_id != "" and material_id in available_materials():
            if given:
                raise ValueError(f"{tag}: {name} has material_id {material_id!r}; "
                                 "E and nu are given only for a gear without one")
            material = get_material(material_id)
            E_MPa.append(material.E)
            nu_gear.append(material.poisson_ratio)
            source.append(material_id)
        else:
            if E[i] is None or nu[i] is None:
                raise ValueError(f"{tag}: {name} has no AxisForge material (material_id {material_id!r} "
                                 "empty or not registered) and E, nu are not given for it")
            E_MPa.append(E[i])
            nu_gear.append(nu[i])
            source.append("given" if material_id == "" else f"given:{material_id}")

    out = []
    for i, label in enumerate(labels):
        case = ContactCase(
            label=f"{tag}_{label}",
            # equivalent cylinders in the normal section; x = profile direction (rolling, array axis 0),
            # the line is infinite along the face width (y). Signed as AxisForge: ring gear rho2 < 0.
            r1=(float(inputs["rho1_n_mm"][i]) * _MM, math.inf), r2=(float(inputs["rho2_n_mm"][i]) * _MM, math.inf),
            E=(E_MPa[0] * _MPA, E_MPa[1] * _MPA), nu=(nu_gear[0], nu_gear[1]),
            load=float(inputs["w_N_per_mm"][i]) / _MM,                  # [N/mm] -> [N/m]
            line=True)
        location = GearLocation(point=label, s_mm=float(inputs["s_mm"][i]), t_s=float(inputs["t_s"][i]),
                                g_y_mm=float(inputs["g_y_mm"][i]), share=float(inputs["share"][i]),
                                l_mm=float(inputs["l_mm"][i]), l_sum_mm=float(inputs["l_sum_mm"][i]),
                                n_pairs=int(inputs["n_pairs"][i]), driver=driver,
                                material1=source[0], material2=source[1])
        # AxisForge speeds, already in m/s and signed (gear 1 referenced): copied, not recomputed
        kinematics = Kinematics(ue_m_per_s=float(inputs["ue_m_per_s"][i]), vg_m_per_s=float(inputs["vg_m_per_s"][i]),
                                u1_m_per_s=float(inputs["u1_m_per_s"][i]), u2_m_per_s=float(inputs["u2_m_per_s"][i]))
        out.append(Contact(element="gear", subtype=type(pair.family).__name__, location=location, case=case,
                           kinematics=kinematics))
    return out
