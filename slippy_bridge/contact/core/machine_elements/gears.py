"""Cylindrical gears: AxisForge pair + the diameter of the contact point on gear 1 -> tooth contacts.

A tooth contact is the contact of two flanks at one point of the path of contact. Near that point
each involute flank is replaced by a cylinder of its radius of curvature in the plane normal to
the contact line (the local equivalent cylinders), so the contact is a line contact of two
parallel cylinders: the line lies along the face width (bridge y), the profile direction is the
rolling direction (bridge x). It is quasi-steady (each point solved as if it lasted for ever), the
load sharing between the tooth pairs in contact is AxisForge's rigid model (one load per unit
length on all contact lines), and a helical pair is represented by its transverse section at mid
face width.

The input of a contact is the diameter ``dy1`` of the flank of gear 1 at the contact point
(D-0034 item 9). At a given ``dy1`` the radii, the position on the path, the load share and the
load per unit length do not depend on which gear drives; the driver only says which diameters
are the named points A..E (:func:`path_diameters`). AxisForge is read here only: the geometry
from ``CylindricalFamily.pair_values_at``, the load from ``load_sharing``. Units are converted
here, once: mm -> m (``_MM``), N/mm -> N/m, MPa -> Pa (``_MPA``).

References
----------
ISO 21771:2007, 4.3.8 Eq. (17) (radius of curvature of the involute at a diameter), 5.4.5.3
Eq. (87) (the two radii add up to T1T2), through AxisForge ``CylindricalFamily.pair_values_at``,
``path_points`` and ``curvature_radii``; AxisForge
``axisforge.solvers.machine_elements.gears.contact.load_sharing``. Johnson (1985), section 4.2(c)
(line contact of two cylinders).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from axisforge.core.machine_elements.gears.cylindrical import CylindricalFamily
from axisforge.core.materials import available_materials, get_material
from axisforge.solvers.machine_elements.gears.contact import load_sharing

from ..contact import DC, Contact, SweptCircleBody
from . import _MM, _MPA

__all__ = ["GearContact", "ToothContact", "gear_contacts", "path_diameters"]

# OPEN R-024: rho2_n of a ring gear is concave < 0 (probe at C); José may close it.
# OPEN R-027: helical pairs are mid-face slices only; EHL on beta != 0 to be refused.
# OPEN R-032: local equivalent cylinders, not the involute flank surfaces.
# OPEN R-118: E, nu of the FZG gears (20MnCr5): no held source; the case gives them explicitly.


@dataclass(**DC)
class GearContact(Contact):
    """Contact of the flanks of gear 1 (body 1) and gear 2 (body 2) at one point of the path.

    Abstract like :class:`Contact` (:class:`ToothContact` sets ``line``). Field names and units as
    AxisForge (``pair_values_at``, ``path_points``, ``load_sharing``).

    Parameters
    ----------
    point : str
        Label of the point: "A".."E" or a label given by the caller (no ``:``, it goes into file
        names).
    dy1_mm, dy2_mm : float
        Diameter of the contact point on the flank of gear 1 and of gear 2 [mm].
    g_y_mm : float
        Position on the path from the pitch point C, positive toward the tip of gear 1 [mm].
    share : float
        Share of the normal force carried by this tooth pair [-].
    l_mm : float
        Length of the contact line of this tooth pair [mm].
    l_sum_mm : float
        Total length of the contact lines of all tooth pairs in contact [mm]; ``load = F_bn / l_sum``.
    n_pairs : int
        Number of tooth pairs in contact [-].
    driver : {"gear1", "gear2"}
        The driving gear (fixes which diameters are A..E; not the geometry at ``dy1``).
    z2 : int
        Tooth number of gear 2, signed (ISO 21771: negative for an internal ring), for the sign of
        the flank of gear 2.
    material1, material2 : str
        Source of E and nu of gear 1 and gear 2: the AxisForge ``material_id``, or ``"given"`` /
        ``"given:<material_id>"`` for values given to :func:`gear_contacts`.

    Other fields as :class:`Contact`. Bridge y lies along the contact line: the face width for a
    spur pair, inclined at beta_b in the plane of action for a helical one; ``load`` is the load
    per unit length of that line [N/m].
    """

    point: str
    dy1_mm: float
    dy2_mm: float
    g_y_mm: float
    share: float
    l_mm: float
    l_sum_mm: float
    n_pairs: int
    driver: str
    z2: int                       # OPEN SB-11: z2 stored on the contact for the sign check of rho2
    material1: str                # OPEN SB-12: E, nu source recorded on the contact
    material2: str


@dataclass(**DC)
class ToothContact(GearContact):
    """Line contact of two involute flanks, replaced by their equivalent cylinders (see
    :class:`GearContact`). Body radii ``(rho_n, inf)``: the normal radius of curvature along the
    profile (bridge x), straight along the contact line (bridge y).

    The radius checks guard hand-built contacts (D-0034 item 6). Through :func:`gear_contacts`
    they cannot fail: ``pair_values_at`` gives ``rho_y1 = L_y1 >= 0`` from ``dy1`` and
    ``rho_y2 = T1T2 - rho_y1`` from the line of action, of the sign of z2 wherever the flanks are
    involutes (beyond T2 of an external pair ``rho_y2`` changes sign; for a ring T2 is not
    reachable), and a pair whose path passes a base-tangent point is refused by
    :func:`gear_contacts` (SB-20). A zero radius is refused by :class:`SweptCircleBody` before
    these checks.

    Raises
    ------
    ValueError
        If ``driver`` is not "gear1"/"gear2"; if a body is not straight along y (``r_y != inf``);
        if ``body1.r[0] <= 0`` (the flank of gear 1 is convex on the involute between the
        base-tangent points, R-030); if the sign of ``body2.r[0]`` is not that of ``z2`` (gear 2
        is convex on an external pair, z2 > 0, and concave on an internal ring, z2 < 0); and
        every check of :class:`Contact`.
    """

    line = True

    def __post_init__(self) -> None:
        name = f"ToothContact {self.label!r}"
        if self.driver not in ("gear1", "gear2"):
            raise ValueError(f"{name}: driver must be 'gear1' or 'gear2', got {self.driver!r}")
        if not all(math.isinf(body.r[1]) for body in (self.body1, self.body2)):
            raise ValueError(f"{name}: the flanks are straight along the face width (r_y = inf), got "
                             f"{self.body1.r[1]}, {self.body2.r[1]}")
        if not self.body1.r[0] > 0:
            raise ValueError(f"{name}: rho1_n must be > 0, got {self.body1.r[0]} m (pair assembled with check=False?)")
        if np.sign(self.body2.r[0]) != np.sign(self.z2):
            raise ValueError(f"{name}: rho2_n = {self.body2.r[0]} m has not the sign of z2 = {self.z2} "
                             "(convex external gear 2, concave internal ring; check=False pair? R-030)")
        super().__post_init__()


# OPEN SB-21: path_diameters gives A..E only; one-sided B-/B+/D-/D+ left to the caller
def path_diameters(pair, driver: str) -> dict[str, float]:
    """Diameters of the flank of gear 1 at the points A..E of the path, for a driving gear [mm].

    From AxisForge ``CylindricalFamily.path_points(pair, driver)`` (positions ``g_y``):
    ``dy1 = 2 sqrt((T1C + g_y)^2 + rb1^2)``, ISO 21771 Eq. (17) inverted (the radius of curvature
    ``T1C + g_y`` and the base radius ``rb1`` are the legs of a right triangle whose hypotenuse is
    the radius at the point), as ``pair_values_at`` does for ``dy2``. With gear 1 driving, A is at
    the active root ``2 rA1`` and E at the tip ``da1``; gear 2 driving swaps A <-> E and B <-> D.

    B and D are the limits of single tooth contact; evaluated exactly there, ``load_sharing``
    gives the double-contact share (R-305). A one-sided point (B+, D-, ...) is the caller's: shift
    along the path by a small distance. The labels follow the path from A to E, but C need not lie
    between B and D (e.g. an internal 24/-60 pair).

    Parameters
    ----------
    pair : GearPair
        AxisForge pair of the cylindrical family.
    driver : {"gear1", "gear2"}

    Returns
    -------
    dict
        ``{"A", "B", "C", "D", "E": dy1}`` [mm], to give to :func:`gear_contacts`.
    """
    points = CylindricalFamily.path_points(pair, driver)
    rb1 = pair.gear1.rb
    return {k: float(2.0 * math.hypot(pair.T1C + g_y, rb1)) for k, g_y in points.items()}


def gear_contacts(pair, dy1, F_bn_N: float, *, driver: str, E=None, nu=None) -> list[ToothContact]:
    """Tooth contacts of an AxisForge cylindrical pair at diameters ``dy1`` of gear 1 (SI units).

    Each contact is the two flanks at the point of the path where the flank of gear 1 has
    diameter ``dy1``: two parallel cylinders of the normal radii of curvature, pressed by the load
    per unit length the tooth pair carries there.

    Parameters
    ----------
    pair : GearPair
        AxisForge pair of the cylindrical family. E and nu of each gear come from its
        ``material_id`` registered in ``axisforge.core.materials``, unless given below.
    dy1 : float, sequence of float, or mapping of str to float
        Diameter(s) of the contact point on gear 1 [mm], on the path ``2 rA1 <= dy1 <= da1``
        (the active root diameter d_Nf1 to the tip; the same segment for either driver). A
        mapping gives the point labels (e.g. :func:`path_diameters`); otherwise each label is
        ``f"dy1={dy1:.6g}"``.
    F_bn_N : float
        Total normal force of the mesh [N] (AxisForge ``MeshForces.F_bn_N``), > 0.
    driver : {"gear1", "gear2"}
        The driving gear, recorded on each contact.
    E, nu : (value or None, value or None), optional
        Young's modulus [MPa] and Poisson's ratio [-] of gear 1 and gear 2, given only for a gear
        without a registered AxisForge material; None for a gear that has one.

    Returns
    -------
    list of ToothContact
        One per diameter, in the order given; label ``f"{pair.label or 'gear'}_{point}"``.
        ``body1 = SweptCircleBody(r=(rho1_n, inf))``, ``body2 = SweptCircleBody(r=(rho2_n, inf))`` with
        ``rho1_n``, ``rho2_n`` the normal-plane radii of ``pair_values_at`` (``curvature_radii``,
        ``rho_t / cos(beta_b)``) [m] (signed: a ring is concave), ``load = w`` [N/m] with
        ``w = F_bn / l_sum`` (AxisForge ``load_sharing``: ``share F_bn / l`` written without
        ``l``, which is zero at the ends of a helical line).

    Raises
    ------
    ValueError
        If ``F_bn_N`` is not positive; if the path of the pair passes a base-tangent point
        (``T1A < 0`` or ``sign(z2) T2E < 0``, a pair assembled with ``check=False``); if a ``dy1``
        is off the path; if a label contains ``:``;
        if a gear has no AxisForge material and no ``E``, ``nu`` given for it, or has one and
        ``E`` or ``nu`` given; if ``E`` or ``nu`` is not a 2-sequence; and every check of
        :class:`ToothContact`.
    """
    tag = pair.label or "gear"
    # kept here: Contact would report the load per unit length, not the force given
    if not (F_bn_N > 0 and math.isfinite(F_bn_N)):
        raise ValueError(f"{tag}: F_bn_N must be positive and finite, got {F_bn_N}")
    if isinstance(dy1, Mapping):
        labels, diameters = list(dy1), [float(v) for v in dy1.values()]
    else:
        diameters = [float(v) for v in np.atleast_1d(dy1)]
        labels = [f"dy1={d:.6g}" for d in diameters]
    bad = [label for label in labels if ":" in label]
    if bad:
        raise ValueError(f"{tag}: point labels go into file names, no ':' (NTFS stream): {bad}")
    # Beyond a base-tangent point there is no involute (curvature_radii: the values there are not radii
    # of a flank). The Q39 test of AxisForge validate_pair on the stored segments.
    # OPEN SB-20: pairs whose path passes a base-tangent point are refused whole
    # OPEN SB-59: since AxisForge 54e71dd rho_y2 is signed from the line of action, so the T2E < 0 branch could be left
    # to ToothContact; T1A < 0 cannot: the dy1 input folds a point beyond T1 onto its mirror (rho_y1 = L_y1 >= 0)
    if pair.T1A < 0.0 or np.sign(pair.gear2.z) * pair.T2E < 0.0:
        raise ValueError(f"{tag}: the path passes a base-tangent point (T1A = {pair.T1A:.6g} mm, "
                         f"sign(z2) T2E = {np.sign(pair.gear2.z) * pair.T2E:.6g} mm; both must be >= 0): "
                         "a pair assembled with check=False is not translated")
    E_MPa, nu_gear, source = _materials(pair, tag, E, nu)

    values = CylindricalFamily.pair_values_at(pair, np.asarray(diameters))
    g_y = values["g_y_mm"]                     # signed position on the path, + toward the tip of gear 1
    # on the path: distance from the stored A within [0, AE], with the end tolerance of load_sharing
    # (1e-9 p_bt) so that both agree at A and E; a dy1 below the base circle gives nan and fails too
    s, tol = g_y + pair.AC, 1e-9 * pair.gear1.pbt
    for label, d, s_i in zip(labels, diameters, s):
        if not -tol <= s_i <= pair.AE + tol:
            raise ValueError(f"{tag}_{label}: dy1 = {d} mm is off the path [{2.0 * pair.rA1:.6g}, "
                             f"{pair.gear1.da:.6g}] mm (2 rA1 to da1)")
    sharing = load_sharing(pair, g_y)
    z2 = int(pair.gear2.z)
    out = []
    for i, label in enumerate(labels):
        # normal-plane radii (Meusnier, rho_t / cos(beta_b)), AxisForge curvature_radii at g_y
        rho1_n_mm, rho2_n_mm = (float(values[k][i]) for k in ("rho1_n_mm", "rho2_n_mm"))
        out.append(ToothContact(
            # x = profile direction (rolling), y along the contact line; ring rho2 < 0
            body1=SweptCircleBody(r=(rho1_n_mm * _MM, math.inf), E=E_MPa[0] * _MPA, nu=nu_gear[0]),
            body2=SweptCircleBody(r=(rho2_n_mm * _MM, math.inf), E=E_MPa[1] * _MPA, nu=nu_gear[1]),
            load=F_bn_N / float(sharing["l_sum_mm"][i]) / _MM,              # [N/mm] -> [N/m]
            label=f"{tag}_{label}", point=label, dy1_mm=diameters[i], dy2_mm=float(values["dy2_mm"][i]),
            g_y_mm=float(g_y[i]), share=float(sharing["share"][i]), l_mm=float(sharing["l_mm"][i]),
            l_sum_mm=float(sharing["l_sum_mm"][i]), n_pairs=int(sharing["n_pairs"][i]), driver=driver, z2=z2, material1=source[0], material2=source[1]))
    return out


def _materials(pair, tag: str, E, nu) -> tuple[list[float], list[float], list[str]]:
    """E [MPa], nu and their source per gear: the registered AxisForge material, or the values
    given for a gear without one."""
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
            # an id that AxisForge does not have stays visible
            source.append("given" if material_id == "" else f"given:{material_id}")
    return E_MPa, nu_gear, source
