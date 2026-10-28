"""Shaft comparator of v001 and v002: a solved shaft on two pins (AxisForge ``RigidSupportFEMSolver``) against r01.

:func:`check` takes any shaft on two pins, solved, uniform or stepped (the ``ShaftSystem`` from a
construction and the ``ShaftResults`` of its solve, from v001, v002, their literature or a study),
with point forces, distributed forces, point couples (``ExternalMoment``, J-16) and prescribed pin
settlements, and returns one comparison record per quantity and bending plane. The judge is r01
(``solve_uniform`` for a uniform shaft without distributed load, else ``solve_stepped`` with one
``Segment`` per section) of the same loads, supports and sections, built from the system itself, plus
the rigid-body field of the settlements (:func:`rigid_body`). One comparator for v001, v002 and the s
cases that validate a shaft through them (v002 minimal review, R5).

Planes. AxisForge solves the planes XY (deflection v_xy along +Y, force F_y) and XZ (v_xz along +Z,
F_z) separately. The nodal rotation theta is the rotation of the cross-section in the plane, positive
from +x toward +v: dv/dx for Euler-Bernoulli, dv/dx - gamma for Timoshenko (gamma = V/kGA the shear
strain; the AxisForge docstrings that say theta = dv/dx are register R-584). r01 solves one plane
(x, v) with forces and deflections along the same +v and the same section rotation theta; forces map
as they are. Couples: :func:`plane_couples` (the CW-01 trap: a moment about +Y enters r01 as -M_y).

Why nodal values can be exact. The Euler-Bernoulli element has cubic Hermite shape functions, which
contain the exact homogeneous solution of EI v'''' = 0; with the loads on nodes (and a consistent
load vector for a distributed load) its nodal deflections and rotations are exact for any mesh. This
follows from Onate2013_vol2 section 1.5 (PDF p. 66: the element stiffness, and the nodal force vector
for some loads, are those of matrix structural analysis); Onate does not state it in these words. The
2-node Timoshenko element with one-point shear integration is not: see :func:`kga_reduced_integration`.

Units: mm, N, MPa, N.mm, rad on both sides (AxisForge and r01 share them).
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
from axisforge.core.loads import LoadPlane
from axisforge.core.materials.base import get_material

from studies.r00_analytical_references.r01_beams import (DistributedLoad, Segment, section_stiffness, solve_stepped,
                                                         solve_uniform)

__all__ = ["check", "reference", "section", "segments", "plane_couples", "rigid_body", "along_shaft",
           "kga_reduced_integration", "literature_record", "theory_name", "node_value", "richardson_verdict",
           "TOL_NODAL", "PLANES", "J16"]

# [-] nodal values vs r01, relative to the largest value of the field in the plane. Machine precision
# for a nodally exact element; the round-off of the solve grows with cond(K) ~ (L/h)^4 for the
# Euler-Bernoulli element (1e-11 at n = 128 on v001's shaft, 5e-9 at h = 0.625 mm on v002's, R5-33),
# so 1e-9 leaves two decades on v001's meshes (brief of v001, "rel 1e-9")
TOL_NODAL = 1e-9
J16 = "convention-dependent (J-16)"

# plane -> (AxisForge LoadPlane, ShaftResults deflection, rotation, reaction attribute)
PLANES = {"XZ": (LoadPlane.XZ, "v_xz", "theta_xz", "R_xz"), "XY": (LoadPlane.XY, "v_xy", "theta_xy", "R_xy")}


def section(system, settings, *, kGA_override: float | None = None) -> tuple[float, float | None]:
    """Bending and corrected shear stiffness of the (single) section of ``system`` (r01 ``section_stiffness``).

    Parameters
    ----------
    system : axisforge ShaftSystem
        A uniform shaft: one ``ShaftSection`` (diameter [mm], registered material, E [MPa], nu [-]).
    settings : axisforge BeamModelSettings
        Euler-Bernoulli gives ``kGA = None``; Timoshenko uses its ``shear_theory`` ("cowper" or
        "hutchinson", the same names and formulas in r01 ``KAPPA``).
    kGA_override : float, optional
        Shear stiffness [N] given to the solver instead of the section's (``RigidSupportFEMSolver``).

    Returns
    -------
    EI : float
        Bending stiffness [N.mm^2].
    kGA : float or None
        Corrected shear stiffness [N]; None for Euler-Bernoulli.
    """
    (seg,) = segments(system, settings, kGA_override=kGA_override)
    return seg.EI, seg.kGA


def kga_reduced_integration(kGA: float, EI: float, h: float) -> float:
    """Shear stiffness of the exact Timoshenko element equal to the one-point (reduced) integrated element [N].

    The 2-node linear Timoshenko element with one-point integration of the shear stiffness has the
    flexibility of a cantilever of length h (tip force, tip moment)

        F_RI = [[h/kGA + h^3/(4 EI), h^2/(2 EI)], [h^2/(2 EI), h/EI]]     (Onate2013_vol2 Eq. (2.40), PDF p. 84)

    while the exact Timoshenko beam has ``h/kGA + h^3/(3 EI)`` in the first entry (Eq. (2.35), PDF p.
    81) and the same other entries. The element is therefore the exact element of a beam whose shear
    flexibility is reduced by h^2/(12 EI):

        1 / kGA_eff = 1 / kGA - h^2 / (12 EI).

    On a uniform mesh (every element of length h) with POINT loads on nodes only, the assembled FEM is
    the exact Timoshenko model with ``kGA_eff``, so its nodal values equal r01 at ``kGA_eff`` to round-off,
    and its error against r01 at ``kGA`` is that of a shear flexibility reduced by h^2/(12 EI): second
    order in h, the FEM too stiff. Not for a distributed load: AxisForge's Timoshenko load vector is
    q h/2 per node with no nodal moments, while the nodally exact load of the kGA_eff element also has
    the end moments +-q h^2/12 (they cancel between interior elements, not at the ends of the load).
    For a cantilever with n elements this gives Onate's limit ``r_w = 1 - 1/(4 n^2)`` (Table 2.1).
    ``kGA_eff`` may be negative on a coarse mesh of a slender beam; it is an algebraic equivalent
    there, not a physical stiffness.

    Parameters
    ----------
    kGA : float
        Corrected shear stiffness of the section [N].
    EI : float
        Bending stiffness [N.mm^2].
    h : float
        Element length [mm].

    Returns
    -------
    float
        kGA_eff [N].
    """
    return 1.0 / (1.0 / kGA - h * h / (12.0 * EI))


def segments(system, settings, *, kGA_override: float | None = None, h_mm: float | None = None) -> list[Segment]:
    """r01 ``Segment`` list of the shaft of ``system``: one per ``ShaftSection``, EI and kGA from r01 ``section_stiffness``.

    Parameters
    ----------
    system : axisforge ShaftSystem
        Its shaft: sections of diameter [mm] (solid, or hollow for Euler-Bernoulli), registered
        material (E [MPa], nu [-]); one section for a uniform shaft.
    settings : axisforge BeamModelSettings
        Euler-Bernoulli gives ``kGA = None``; Timoshenko uses ``shear_theory`` ("cowper" or "hutchinson").
    kGA_override : float, optional
        The solver's ``kGA_override`` [N], the same for every section.
    h_mm : float, optional
        Element length of a uniform mesh [mm]: each Timoshenko segment gets
        :func:`kga_reduced_integration` of its own EI and kGA. Every element lies inside one section
        (the steps are nodes), so the assembled one-point FEM is the exact stepped beam with these
        per-segment values (C4).

    Returns
    -------
    list of Segment
        ``Segment(x_lo, x_hi, EI [N.mm^2], kGA [N] or None)`` covering [0, L].
    """
    shaft = system.shaft
    timoshenko = settings.beam_theory == "timoshenko"
    out = []
    for i, sec in enumerate(shaft.sections):
        elastic = get_material(sec.material_id).elastic
        EI, kGA = section_stiffness(elastic.E, elastic.poisson_ratio, sec.diameter, sec.inner_diameter,
                                    kappa=settings.shear_theory if timoshenko else None)
        if timoshenko and kGA_override is not None:
            kGA = kGA_override
        if timoshenko and h_mm is not None:
            kGA = kga_reduced_integration(kGA, EI, h_mm)
        out.append(Segment(shaft.axial_start(i), shaft.axial_end(i), EI, kGA))
    return out


def plane_couples(system, plane: str) -> list[tuple[float, float]]:
    """Point couples of ``system`` in one plane as r01 takes them: ``[(x [mm], C [N.mm])]``, C from +x toward +v.

    ``C = ExternalMoment.component(LoadPlane)``: AxisForge adds ``component(XY)`` (M cos theta) to the
    rotation DOF of the plane XY and ``component(XZ)`` (M sin theta) to that of XZ, in N.mm
    (``core/loads.py`` Warnings, ``vector_external_forces.py``); a nodal moment on the rotation DOF
    turns the section from +x toward +v, so it is an r01 couple of the same sign. Physically a moment
    vector (M_X, M_Y, M_Z) acts in XY as +M_Z and in XZ as -M_Y (CW-01); v002 ``gear_identities`` (I5)
    checks that the two agree for the couple of a gear mesh. ``convention-dependent (J-16)``: D-0010
    rows 21-22 are open.
    """
    lp = PLANES[plane][0]
    return [(m.position, float(m.component(lp))) for m in system.external_moments]


def rigid_body(x, x_A: float, x_B: float, delta_A: float, delta_B: float) -> tuple[np.ndarray, float]:
    """Rigid-body field of a beam on two pins whose pins are displaced by ``delta_A``, ``delta_B`` [mm] along +v.

    A statically determinate beam takes a settlement without strain: deflection
    ``delta_A + (delta_B - delta_A) (x - x_A) / (x_B - x_A)`` [mm] and the constant rotation
    ``(delta_B - delta_A) / (x_B - x_A)`` [rad], added to the field of the loads; moments, shear
    forces and reactions are unchanged (statics; r01 has no settlement argument until part G1).

    Returns
    -------
    (ndarray, float)
        Deflection at ``x`` [mm] and the rotation [rad].
    """
    slope = (delta_B - delta_A) / (x_B - x_A)
    return delta_A + slope * (np.asarray(x, dtype=float) - x_A), slope


def reference(system, settings, *, kGA_override: float | None = None, h_mm: float | None = None,
              couples: bool = True) -> dict:
    """r01 solution of each bending plane of ``system`` (two pins), from its own loads, without settlements.

    Point forces (``RadialLoad.component``), distributed forces (``component_intensity`` [N/mm], exact
    for a polynomial of degree <= 6), couples (:func:`plane_couples`, J-16) and the sections
    (:func:`segments`). A uniform shaft without distributed load goes to r01 ``solve_uniform`` (the
    independent Macaulay solver), anything else to ``solve_stepped``. Supports: the first bearing by
    position is x_A, the second x_B. ``couples=False`` leaves the couples out.

    Returns
    -------
    dict
        ``{"XZ": PlaneSolution, "XY": PlaneSolution}`` with ``v``, ``theta``, ``M``, ``V``, ``R_A``,
        ``R_B`` [mm, rad, N.mm, N].
    """
    x_A, x_B = sorted(b.position for b in system.bearings)
    segs = segments(system, settings, kGA_override=kGA_override, h_mm=h_mm)
    out = {}
    for plane, (lp, *_) in PLANES.items():
        forces = [(ld.position, float(ld.component(lp))) for ld in system.radial_loads]
        dist = [DistributedLoad(ld.x_lo, ld.x_hi, partial(ld.component_intensity, plane=lp))
                for ld in system.distributed_radial_loads]
        cpl = plane_couples(system, plane) if couples else []
        if len(segs) == 1 and not dist:
            out[plane] = solve_uniform(forces, x_A, x_B, segs[0].EI, segs[0].kGA, couples=cpl)
        else:
            out[plane] = solve_stepped(forces, dist, x_A, x_B, segs, system.shaft.total_length, couples=cpl)
    return out


def theory_name(settings) -> str:
    """Name of a beam model, the key of ``construction.THEORIES``: euler_bernoulli, timoshenko/<shear>/<integration>."""
    if settings.beam_theory == "euler_bernoulli":
        return "euler_bernoulli"
    return f"timoshenko/{settings.shear_theory}/{settings.integration_method}"


def _loaded_planes(system) -> list[str]:
    """Planes whose applied force or couple is not round-off of the other plane's (cos 270 deg ~ -1.8e-16)."""
    size = {}
    for plane, (lp, *_) in PLANES.items():
        force = (sum(abs(ld.component(lp)) for ld in system.radial_loads)
                 + sum(abs(ld.resultant_component(lp)) for ld in system.distributed_radial_loads))
        # a couple C [N.mm] over the shaft length is a force scale [N]
        size[plane] = force + sum(abs(c) for _, c in plane_couples(system, plane)) / system.shaft.total_length
    return [p for p, s in size.items() if s > 1e-9 * max(size.values())]


def _single_kga(segs) -> float:
    """kGA [N] of the reference when every segment has the same one (a uniform shaft), else nan (stepped or EB)."""
    values = {s.kGA for s in segs}
    return float(values.pop()) if len(values) == 1 and None not in values else math.nan


def check(system, result, settings, *, settlements_mm: dict | None = None, kGA_override: float | None = None,
          case: str = "v001", label: str = "") -> list[dict]:
    """Comparison records of one solved shaft on two pins (uniform or stepped) against r01, per bending plane.

    Quantities and tolerances (README section 6 of v001 and v002, fixed before the run). Field errors
    are the largest nodal deviation divided by the largest absolute r01 value of that field in the plane.

    * C1 deflection ``v`` [mm] and C2 rotation ``theta`` [rad], Euler-Bernoulli vs r01 Euler-Bernoulli
      plus the settlement field (:func:`rigid_body`): within ``TOL_NODAL``. Nodal exactness holds in
      exact arithmetic; in float64 the Euler-Bernoulli stiffness has cond ~ (L/h)^4, so round-off
      reaches 1e-9 near h = 0.6 mm on v002's shaft (R5-33);
    * C3 support reactions [N], any model vs r01 (statically determinate: equal for every model and
      every settlement): within ``TOL_NODAL`` of the largest reaction of the plane;
    * C4 ``v``, ``theta``, Timoshenko with one-point shear integration on a uniform mesh vs r01 at
      ``kga_reduced_integration`` per segment: within ``TOL_NODAL`` with point loads only (with a
      distributed load the identity does not hold: the C4 rows are RECORD, R5-25);
    * C7 axial reactions [N] vs statics, when the shaft carries an ``AxialLoad``: the locating pin
      takes ``-sum(F_x)``, the non-locating one nothing: within ``TOL_NODAL`` of the sum;
    * T1 ``v``, ``theta``, Timoshenko vs r01 at the sections' kGA: RECORD (the O(h^2) error of the
      element; judged through C4 and the Richardson extrapolation C6 of the study);
    * R1 bending moment ``M`` [N.mm] at the nodes (post-processing): RECORD (the Timoshenko nodal M
      is the constant M of one adjacent element, registers R-224, R-391; at a couple the node holds
      the left element's value while r01 is right-continuous).

    Every record of a system with a couple is tagged ``convention-dependent (J-16)`` in ``tag``.

    Parameters
    ----------
    system : axisforge ShaftSystem
        The shaft, two pins, its loads, as solved.
    result : axisforge ShaftResults
        Its solve (``RigidSupportFEMSolver``, or v002 ``study.solve`` with settlements).
    settings : axisforge BeamModelSettings
        The beam model of the solve.
    settlements_mm : dict, optional
        ``{"XY": (delta_A, delta_B), "XZ": (...)}``: prescribed displacements of the first and the
        second pin by position along +Y / +Z [mm]; None or a missing plane: rigid pins.
    kGA_override : float, optional
        The ``kGA_override`` [N] of the solve, if any.
    case, label : str
        Written into the records (an s passes its own).

    Returns
    -------
    list of dict
        Fields ``case``, ``label``, ``theory``, ``plane``, ``n_elements``, ``h_mm`` (largest element),
        ``uniform_mesh``, ``id``, ``quantity``, ``unit``, ``max_abs_diff``, ``ref_scale``, ``rel_err``,
        ``tol``, ``verdict``, ``reference``, ``kGA_N`` (of the reference; nan if stepped or EB),
        ``delta_A_mm``, ``delta_B_mm``, ``tag``.
    """
    settlements_mm = settlements_mm or {}
    x = np.asarray(result.x_nodes, dtype=float)
    lengths = np.diff(x)
    h = float(lengths.max())
    uniform = bool(np.all(np.abs(lengths - lengths[0]) <= 1e-9 * lengths[0]))
    theory = theory_name(settings)
    refs = {"r01": (reference(system, settings, kGA_override=kGA_override),
                    _single_kga(segments(system, settings, kGA_override=kGA_override)))}
    reduced = settings.beam_theory == "timoshenko" and settings.integration_method == "single_point"
    if reduced and uniform:
        refs["r01 kGA_eff"] = (reference(system, settings, kGA_override=kGA_override, h_mm=h),
                               _single_kga(segments(system, settings, kGA_override=kGA_override, h_mm=h)))
    x_A, x_B = sorted(b.position for b in system.bearings)
    support_x = [b.position for b in system.bearings]
    tag = J16 if system.external_moments else ""
    common = {"case": case, "label": label or system.name, "theory": theory, "n_elements": len(x) - 1, "h_mm": h,
              "uniform_mesh": uniform}

    def row(cid, plane, quantity, unit, fem, ref, tol, ref_name, k, dA=0.0, dB=0.0):
        diff = float(np.max(np.abs(np.asarray(fem, dtype=float) - ref)))
        scale = float(np.max(np.abs(ref)))
        err = diff / scale if scale > 0.0 else diff
        verdict = "RECORD" if tol is None else ("PASS" if err <= tol else "MISS")
        return {**common, "plane": plane, "id": cid, "quantity": quantity, "unit": unit, "max_abs_diff": diff,
                "ref_scale": scale, "rel_err": err, "tol": math.nan if tol is None else tol, "verdict": verdict,
                "reference": ref_name, "kGA_N": k, "delta_A_mm": dA, "delta_B_mm": dB, "tag": tag}

    rows = []
    for plane in _loaded_planes(system):
        _, v_key, th_key, r_key = PLANES[plane]
        dA, dB = settlements_mm.get(plane, (0.0, 0.0))
        v_rb, th_rb = rigid_body(x, x_A, x_B, dA, dB)
        sols, k = refs["r01"]
        sol = sols[plane]
        fem_v, fem_th = getattr(result, v_key), getattr(result, th_key)
        if settings.beam_theory == "euler_bernoulli":
            rows += [row("C1", plane, "v", "mm", fem_v, sol.v(x) + v_rb, TOL_NODAL, "r01", k, dA, dB),
                     row("C2", plane, "theta", "rad", fem_th, sol.theta(x) + th_rb, TOL_NODAL, "r01", k, dA, dB)]
        else:
            rows += [row("T1", plane, "v", "mm", fem_v, sol.v(x) + v_rb, None, "r01", k, dA, dB),
                     row("T1", plane, "theta", "rad", fem_th, sol.theta(x) + th_rb, None, "r01", k, dA, dB)]
            if "r01 kGA_eff" in refs:
                sols_e, k_e = refs["r01 kGA_eff"]
                tol4 = None if system.distributed_radial_loads else TOL_NODAL    # point loads only (R5-25)
                rows += [row("C4", plane, "v", "mm", fem_v, sols_e[plane].v(x) + v_rb, tol4, "r01 kGA_eff", k_e,
                             dA, dB),
                         row("C4", plane, "theta", "rad", fem_th, sols_e[plane].theta(x) + th_rb, tol4, "r01 kGA_eff",
                             k_e, dA, dB)]
        # reactions: ShaftResults.R_* follow system.bearings, forces of the support on the shaft along
        # +v, the sign of r01's R_A, R_B; the order of r01 is by position
        R_ref = {sol.x_A: sol.R_A, sol.x_B: sol.R_B}
        rows.append(row("C3", plane, "R", "N", getattr(result, r_key), np.array([R_ref[xs] for xs in support_x]),
                        TOL_NODAL, "r01", k, dA, dB))
        rows.append(row("R1", plane, "M", "N.mm", getattr(result, f"M_{plane.lower()}"), sol.M(x), None, "r01", k,
                        dA, dB))
    F_x = sum(ld.magnitude for ld in system.axial_loads)        # [N] along +X
    if F_x != 0.0:
        # u = 0 at the locating pin only (boundary_conditions.py): it takes the whole axial load, -F_x along +X
        expected = np.array([-F_x if b.arrangement == "locating" else 0.0 for b in system.bearings])
        rows.append(row("C7", "X", "R_axial", "N", result.R_axial, expected, TOL_NODAL, "statics", math.nan))
    return rows


def along_shaft(system, result, settings, *, settlements_mm: dict | None = None, kGA_override: float | None = None,
                series: str = "", n_dense: int = 401) -> list[dict]:
    """Deflection and rotation along the shaft: AxisForge nodes and the r01 curves (with the settlement field).

    Records of ``studies.outputs.l000_axisforge.plots.shaft.deflection_along_shaft``: ``plane``,
    ``series`` (the beam model, or ``series``), ``source`` ("FEM" at the nodes, "r01" on ``n_dense``
    points; for one-point Timoshenko on a uniform mesh with point loads also "r01 kGA_eff", the curve
    C4 judges), ``x_mm``, ``v_mm``, ``theta_rad``.
    """
    settlements_mm = settlements_mm or {}
    x_fem = np.asarray(result.x_nodes, dtype=float)
    lengths = np.diff(x_fem)
    curves = {"r01": reference(system, settings, kGA_override=kGA_override)}
    if (settings.integration_method == "single_point" and not system.distributed_radial_loads
            and np.allclose(lengths, lengths[0], rtol=1e-9, atol=0.0)):
        curves["r01 kGA_eff"] = reference(system, settings, kGA_override=kGA_override, h_mm=float(lengths[0]))
    x_A, x_B = sorted(b.position for b in system.bearings)
    x_ref = np.linspace(0.0, system.shaft.total_length, n_dense)
    name = series or theory_name(settings)
    out = []
    for plane in _loaded_planes(system):
        _, v_key, th_key, _ = PLANES[plane]
        v_rb, th_rb = rigid_body(x_ref, x_A, x_B, *settlements_mm.get(plane, (0.0, 0.0)))
        out += [{"plane": plane, "series": name, "source": "FEM", "x_mm": float(xi), "v_mm": float(v),
                 "theta_rad": float(t)} for xi, v, t in zip(x_fem, getattr(result, v_key), getattr(result, th_key))]
        for source, sols in curves.items():
            sol = sols[plane]
            out += [{"plane": plane, "series": name, "source": source, "x_mm": float(xi), "v_mm": float(v),
                     "theta_rad": float(t)} for xi, v, t in zip(x_ref, sol.v(x_ref) + v_rb, sol.theta(x_ref) + th_rb)]
    return out


def node_value(result, x_mm: float, key: str) -> float:
    """Nodal value of a ``ShaftResults`` field at the node at ``x_mm``; raises if ``x_mm`` is not a node (1e-9 mm rel)."""
    x = np.asarray(result.x_nodes, dtype=float)
    i = int(np.argmin(np.abs(x - x_mm)))
    if abs(x[i] - x_mm) > 1e-9 * max(1.0, abs(x_mm)):
        raise ValueError(f"x = {x_mm} mm is not a node (nearest {x[i]} mm): pass it in extra_mandatory")
    return float(np.asarray(getattr(result, key))[i])


def richardson_verdict(name: str, fin, exact: float, *, case: str, plane: str, quantity: str, n_elements: int,
                       h_mm: float, single_point: bool, flat_rtol: float, **extra) -> dict:
    """C5 / C6 record of the mesh series of one nodal value (``numerical_solvers.richardson``, final triple).

    C5 (Euler-Bernoulli): status ``flat``, i.e. level differences within ``flat_rtol`` |f| (the
    ``RichardsonOptions.flat_rtol`` of the run, recorded as ``tol``); C6 (one-point Timoshenko): status
    ``monotonic`` and the extrapolated value within ``TOL_NODAL`` of r01 (its error is exactly c h^2);
    T2 any other Timoshenko integration: RECORD.
    """
    err = fin.f_extrapolated / exact - 1.0
    if name == "euler_bernoulli":
        cid, tol, ok = "C5", flat_rtol, fin.status == "flat"
    elif single_point:
        cid, tol, ok = "C6", TOL_NODAL, fin.status == "monotonic" and abs(err) <= TOL_NODAL
    else:
        cid, tol, ok = "T2", math.nan, None
    return {"case": case, "label": "richardson", "theory": name, "plane": plane, "n_elements": n_elements,
            "h_mm": h_mm, "uniform_mesh": True, "id": cid, "quantity": quantity, "unit": "mm",
            "max_abs_diff": abs(fin.f_extrapolated - exact), "ref_scale": abs(exact), "rel_err": err, "tol": tol,
            "verdict": "RECORD" if ok is None else ("PASS" if ok else "MISS"),
            "reference": f"r01, Richardson {fin.status}, p {fin.p:.6g}, GCI_fm {fin.gci_fine_medium:.3g}",
            "kGA_N": math.nan, **extra}


def literature_record(lid: str, quantity: str, unit: str, measured: float, reference_: float, resolution: float, *,
                      source: str, judged: bool = True, case: str = "v001", **extra) -> dict:
    """One comparison with a printed value: within half a unit of its last printed digit, inclusive.

    ``tol_abs = resolution / 2`` [unit of the quantity]; the bound is inclusive and widened by 1e-12
    relative, because an exact value can sit on the rounding boundary (Onate Table 2.1 prints 0.938
    for 1 - 1/16 = 0.9375). ``judged=False`` gives RECORD (inputs not printed, or not AxisForge's element).
    """
    diff = measured - reference_
    tol = 0.5 * resolution
    ok = abs(diff) <= tol * (1.0 + 1e-12)
    return {"case": case, "id": lid, "quantity": quantity, "unit": unit, "measured": measured,
            "reference": reference_, "abs_diff": diff, "tol_abs": tol,
            "verdict": ("PASS" if ok else "MISS") if judged else "RECORD", "source": source, **extra}
