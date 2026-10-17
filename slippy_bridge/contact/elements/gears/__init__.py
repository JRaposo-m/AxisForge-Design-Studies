"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Gears: AxisForge cylindrical pair + ``contact_inputs`` record -> one line :class:`Contact` per path point.

The cylindrical pair (spur, helical, internal: one ``GearPair`` class, its subtype is the family,
as for a bearing) is registered. AxisForge already gives, per step of the contact point along the
path, the two equivalent cylinders, the surface speeds and the load per unit length
(``axisforge.solvers.machine_elements.gears.contact.contact_inputs``); this module is the ONLY
place where that record is converted to SI (mm -> m, N/mm -> N/m, MPa -> Pa), as the bearing
adapter does for bearings (D-0019). The input checks are real; the conversion body is a scaffold.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from axisforge.core.machine_elements.gears import CylindricalFamily, GearPair

from ...base import Contact, ElementAdapter, register

__all__ = ["SpurHelicalContacts"]

# AxisForge ``CylindricalFamily.path_points`` gives only A..E (start, limits of single contact, pitch
# point, end); B-, B+, D-, D+ are bridge labels for the one-sided limits either side of B and D (the
# load share jumps there), built by ``construction.path_times`` (GL-09, GL-25).
_POINTS = ("A", "B-", "B+", "C", "D-", "D+", "E")


@register(CylindricalFamily)
class SpurHelicalContacts(ElementAdapter):
    """Adapter of an AxisForge ``GearPair`` of the cylindrical family (spur, helical, internal).

    Each step of the contact point along the path of contact is a line contact of two parallel
    cylinders whose radii are the local radii of curvature of the two flanks in the normal section
    (local equivalent cylinders, ISO 21771 5.4; Johnson 1985 ch. 4). See :meth:`contacts`.
    """

    element = "gear"

    @classmethod
    def contacts(cls, pair: GearPair, inputs: Mapping[str, np.ndarray], *,
                 points: Sequence[str] | None = None) -> list[Contact]:
        """One line :class:`Contact` per step of an AxisForge ``contact_inputs`` record (SI units).

        Physically, each step is the contact of the two flanks at one position of the contact point
        on the line of action, at one instant of the mesh cycle: two parallel cylinders of the local
        normal radii of curvature, pressed by the load per unit length that the tooth pair carries
        there (load sharing between the tooth pairs in contact), with the surface speeds at that
        point. The contact is quasi-steady: each step is solved as if it lasted for ever.

        Parameters
        ----------
        pair : GearPair
            AxisForge pair of the cylindrical family, the one ``inputs`` was computed for. Each gear
            needs a non-empty ``material_id`` registered in ``axisforge.core.materials`` (E, nu).
        inputs : mapping of str to ndarray
            The output of AxisForge ``contact_inputs(pair, driver, n1_rpm, F_bn_N, ...)``
            (``axisforge/solvers/machine_elements/gears/contact.py:134-143, 206-211``), arrays of
            equal length, one entry per step, AxisForge units: ``s_mm``, ``t_s`` [s], ``g_y_mm``,
            ``rho1_n_mm``, ``rho2_n_mm`` (signed: convex > 0, concave < 0) [mm], ``u1_m_per_s``,
            ``u2_m_per_s``, ``vg_m_per_s``, ``ue_m_per_s`` [m/s], ``share`` [-], ``l_sum_mm`` [mm],
            ``n_pairs`` [-], ``w_N_per_mm`` [N/mm].
        points : sequence of str, optional
            One label per step, unique, from ``("A", "B-", "B+", "C", "D-", "D+", "E")``. None
            labels each step ``f"s={s_mm:.6g}"`` (distance from A [mm], 6 significant digits). The
            case label is ``f"{pair.label or 'gear'}_{point}"``: it is used in file names, so no ``:``
            (an NTFS stream separator).

        Returns
        -------
        list of Contact
            One per step, in the order of ``inputs``, with ``element="gear"`` and
            ``subtype="CylindricalFamily"``:

            * ``case`` = ``ContactCase(label=f"{pair.label or 'gear'}_{point}",
              r1=(rho1_n_mm*_MM, inf), r2=(rho2_n_mm*_MM, inf),
              E=(get_material(id1).E*_MPA, get_material(id2).E*_MPA),
              nu=(get_material(id1).poisson_ratio, get_material(id2).poisson_ratio),
              load=w_N_per_mm/_MM, line=True)`` (N/mm -> N/m): radii [m] (``x`` = rolling direction = profile
              direction; the line is infinite along the face width, ``y``), E [Pa], load [N/m];
            * ``location`` = ``{point, s_mm, g_y_mm, t_s, share, n_pairs, l_sum_mm}``, AxisForge
              units, as in the keys;
            * ``kinematics`` = ``{u_e_m_per_s, u1_m_per_s, u2_m_per_s, u_s_m_per_s, slide_roll}``:
              ``u_e_m_per_s`` is the MEAN entrainment speed (u1+u2)/2 (AxisForge ``ue_m_per_s``,
              R-075); ``u_s_m_per_s`` the signed sliding speed u1 - u2 (AxisForge ``vg_m_per_s``);
              ``slide_roll = u_s/u_e`` [-], referenced to gear 1 (AxisForge ``cylindrical.py:1179``):
              its sign is set by g_y (+ on the addendum of gear 1), zero at C; independent of the
              driver (AxisForge ``cylindrical.py:1150-1151``); with gear 2 driving only the order
              A..E reverses.

        Raises
        ------
        ValueError
            If ``points`` does not have one label per step, or a label is repeated or not one of
            the named points; if any ``rho1_n_mm <= 0`` (the flank of gear 1 is convex on a valid
            path; a pair assembled with ``check=False`` can give rho <= 0 at the ends, R-030); if
            the sign of any ``rho2_n_mm`` is not that of ``pair.gear2.z`` (convex external gear 2,
            concave internal ring; R-030); if either gear has an empty ``material_id`` (R-025). A load ``w <= 0`` or ``rho2 = 0`` is
            refused by ``ContactCase``.
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
        # OPEN R-031: signs of u1, u2 for internal pairs (u < 0) before forming ue and slide_roll.
        # OPEN R-032: local equivalent cylinders, not the involute flank surfaces.
        # OPEN R-118: E, nu of the FZG gears (20MnCr5) come from get_material, not from the papers.
        # OPEN GL-05: location and kinematics keys above; José confirms slide_roll = u_s/u_e (gear-1-referenced).
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
        for name in ("gear1", "gear2"):
            if getattr(pair, name).material_id == "":
                raise ValueError(f"{tag}: {name}.material_id is empty; E and nu come from get_material (R-025)")
        raise NotImplementedError("gear contacts: scaffold, OPEN R-023 R-031")
