"""Comparator of v001: a solved uniform shaft (AxisForge ``RigidSupportFEMSolver``) against r01 (README 6).

:func:`check` takes any uniform shaft on two pins, solved (the ``ShaftSystem`` from a construction
and the ``ShaftResults`` of its solve, from this case, its literature or a study), and returns one
comparison record per quantity and bending plane. The judge is r01 (``solve_uniform`` for point
forces, ``solve_stepped`` with one segment when a distributed load is present) of the same loads,
supports and section, built from the system itself.

Planes. AxisForge solves the planes XY (deflection v_xy along +Y, force F_y) and XZ (v_xz along +Z,
F_z) separately. The nodal rotation theta is the rotation of the cross-section in the plane, positive
from +x toward +v: dv/dx for Euler-Bernoulli, dv/dx - gamma for Timoshenko (gamma = V/kGA the shear
strain; the AxisForge docstrings that say theta = dv/dx are register R-584). r01 solves one plane
(x, v) with forces and deflections along the same +v and the same section rotation theta; with forces only (no couple) each AxisForge plane maps to r01 as it is, no sign changes.
The couple sign trap (CW-01: a moment about +Y enters r01 as -M_y) is v002's.

Why nodal values can be exact. The Euler-Bernoulli element has cubic Hermite shape functions, which
contain the exact homogeneous solution of EI v'''' = 0; with the loads on nodes (and a consistent
load vector for a distributed load) its nodal deflections and rotations are exact for any mesh. This
follows from Onate2013_vol2 section 1.5 (PDF p. 66: the element stiffness, and the nodal force vector
for some loads, are those of matrix structural analysis); Onate does not state it in these words. The 2-node Timoshenko element with one-point shear integration is not: see
:func:`kga_reduced_integration`.

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

__all__ = ["check", "reference", "section", "along_shaft", "kga_reduced_integration", "literature_record", "theory_name",
           "TOL_NODAL", "PLANES"]

# [-] nodal values vs r01, relative to the largest value of the field in the plane. Machine precision
# for a nodally exact element; the round-off of the solve grows with cond(K) ~ n^4 (1e-11 at n = 128
# here), so 1e-9 leaves two decades (brief of v001, "rel 1e-9")
TOL_NODAL = 1e-9

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
    (sec,) = system.shaft.sections
    elastic = get_material(sec.material_id).elastic
    timoshenko = settings.beam_theory == "timoshenko"
    EI, kGA = section_stiffness(elastic.E, elastic.poisson_ratio, sec.diameter, sec.inner_diameter,
                                kappa=settings.shear_theory if timoshenko else None)
    return EI, (kGA_override if timoshenko and kGA_override is not None else kGA)


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
    the end moments +-q h^2/12 (they cancel between interior elements, not at the ends of the load). For a cantilever with n elements this gives Onate's
    limit ``r_w = 1 - 1/(4 n^2)`` (Table 2.1). ``kGA_eff`` may be negative on a coarse mesh of a
    slender beam; it is an algebraic equivalent there, not a physical stiffness.

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


def reference(system, EI: float, kGA: float | None) -> dict:
    """r01 solution of each bending plane of ``system`` (two pins), from its own loads.

    Point forces (``RadialLoad``) go to r01 ``solve_uniform``; with a ``DistributedRadialLoad`` the
    plane goes to ``solve_stepped`` with one segment (its intensity ``component_intensity`` [N/mm],
    exact for a polynomial of degree <= 4). Supports: the first bearing is x_A, the second x_B.

    Returns
    -------
    dict
        ``{"XZ": PlaneSolution, "XY": PlaneSolution}`` with ``v``, ``theta``, ``M``, ``V``, ``R_A``,
        ``R_B`` [mm, rad, N.mm, N].
    """
    x_A, x_B = sorted(b.position for b in system.bearings)
    out = {}
    for plane, (lp, *_) in PLANES.items():
        forces = [(ld.position, float(ld.component(lp))) for ld in system.radial_loads]
        dist = [DistributedLoad(ld.x_lo, ld.x_hi, partial(ld.component_intensity, plane=lp))
                for ld in system.distributed_radial_loads]
        if dist:
            out[plane] = solve_stepped(forces, dist, x_A, x_B, [Segment(0.0, system.shaft.total_length, EI, kGA)],
                                       system.shaft.total_length)
        else:
            out[plane] = solve_uniform(forces, x_A, x_B, EI, kGA)
    return out


def theory_name(settings) -> str:
    """Name of a beam model, the key of ``construction.THEORIES``: euler_bernoulli, timoshenko/<shear>/<integration>."""
    if settings.beam_theory == "euler_bernoulli":
        return "euler_bernoulli"
    return f"timoshenko/{settings.shear_theory}/{settings.integration_method}"


def _loaded_planes(system) -> list[str]:
    """Planes whose applied force is not round-off of the other plane's (cos 270 deg ~ -1.8e-16)."""
    size = {plane: sum(abs(ld.component(lp)) for ld in system.radial_loads)
            + sum(abs(ld.resultant_component(lp)) for ld in system.distributed_radial_loads)
            for plane, (lp, *_) in PLANES.items()}
    return [p for p, s in size.items() if s > 1e-9 * max(size.values())]


def check(system, result, settings, *, kGA_override: float | None = None, case: str = "v001",
          label: str = "") -> list[dict]:
    """Comparison records of one solved uniform shaft against r01, per loaded bending plane.

    Quantities and tolerances (README section 6, fixed before the run). Field errors are the largest
    nodal deviation divided by the largest absolute r01 value of that field in the plane.

    * C1 deflection ``v`` [mm] and C2 rotation ``theta`` [rad], Euler-Bernoulli vs r01
      Euler-Bernoulli: within ``TOL_NODAL`` (Hermite nodal exactness);
    * C3 support reactions [N], any model vs r01 of the same model (statically determinate: equal
      for every model): within ``TOL_NODAL`` of the largest reaction of the plane;
    * C4 ``v``, ``theta``, Timoshenko with one-point shear integration on a uniform mesh with point
      loads only vs r01 Timoshenko at ``kga_reduced_integration(kGA, EI, h)``: within ``TOL_NODAL``
      (with a distributed load the identity does not hold: the C4 rows are RECORD, R5-25);
    * T1 ``v``, ``theta``, Timoshenko vs r01 Timoshenko at the section's kGA: RECORD (the O(h^2)
      error of the element; judged through C4 and the Richardson extrapolation C6 of the study);
    * R1 bending moment ``M`` [N.mm] at the nodes (post-processing): RECORD (the Timoshenko nodal M
      is the constant M of one adjacent element, off by about V h/2: registers R-224, R-391).

    Parameters
    ----------
    system : axisforge ShaftSystem
        The uniform shaft, one section, two pins, as solved.
    result : axisforge ShaftResults
        ``RigidSupportFEMSolver(settings, kGA_override=...).solve(system, ...)``.
    settings : axisforge BeamModelSettings
        The beam model of the solve.
    kGA_override : float, optional
        The ``kGA_override`` [N] of the solve, if any.
    case, label : str
        Written into the records (an s passes its own).

    Returns
    -------
    list of dict
        Fields ``case``, ``label``, ``theory``, ``plane``, ``n_elements``, ``h_mm`` (largest element),
        ``uniform_mesh``, ``id``, ``quantity``, ``unit``, ``max_abs_diff``, ``ref_scale``, ``rel_err``,
        ``tol``, ``verdict``, ``reference``, ``kGA_N`` (of the reference).
    """
    x = np.asarray(result.x_nodes, dtype=float)
    lengths = np.diff(x)
    h = float(lengths.max())
    uniform = bool(np.all(np.abs(lengths - lengths[0]) <= 1e-9 * lengths[0]))
    theory = theory_name(settings)
    EI, kGA = section(system, settings, kGA_override=kGA_override)
    refs = {"r01": (reference(system, EI, kGA), kGA)}
    reduced = settings.beam_theory == "timoshenko" and settings.integration_method == "single_point"
    if reduced and uniform:
        k_eff = kga_reduced_integration(kGA, EI, h)
        refs["r01 kGA_eff"] = (reference(system, EI, k_eff), k_eff)
    support_x = [b.position for b in system.bearings]
    common = {"case": case, "label": label or system.name, "theory": theory, "n_elements": len(x) - 1, "h_mm": h,
              "uniform_mesh": uniform}

    def row(cid, plane, quantity, unit, fem, ref, tol, ref_name, k):
        diff = float(np.max(np.abs(np.asarray(fem, dtype=float) - ref)))
        scale = float(np.max(np.abs(ref)))
        err = diff / scale if scale > 0.0 else diff
        verdict = "RECORD" if tol is None else ("PASS" if err <= tol else "MISS")
        return {**common, "plane": plane, "id": cid, "quantity": quantity, "unit": unit, "max_abs_diff": diff,
                "ref_scale": scale, "rel_err": err, "tol": math.nan if tol is None else tol, "verdict": verdict,
                "reference": ref_name, "kGA_N": math.nan if k is None else k}

    rows = []
    for plane in _loaded_planes(system):
        _, v_key, th_key, r_key = PLANES[plane]
        sol, k = refs["r01"][0][plane], refs["r01"][1]
        fem_v, fem_th = getattr(result, v_key), getattr(result, th_key)
        if settings.beam_theory == "euler_bernoulli":
            rows += [row("C1", plane, "v", "mm", fem_v, sol.v(x), TOL_NODAL, "r01", k),
                     row("C2", plane, "theta", "rad", fem_th, sol.theta(x), TOL_NODAL, "r01", k)]
        else:
            rows += [row("T1", plane, "v", "mm", fem_v, sol.v(x), None, "r01", k),
                     row("T1", plane, "theta", "rad", fem_th, sol.theta(x), None, "r01", k)]
            if "r01 kGA_eff" in refs:
                sol_e, k_e = refs["r01 kGA_eff"][0][plane], refs["r01 kGA_eff"][1]
                tol4 = None if system.distributed_radial_loads else TOL_NODAL    # point loads only (R5-25)
                rows += [row("C4", plane, "v", "mm", fem_v, sol_e.v(x), tol4, "r01 kGA_eff", k_e),
                         row("C4", plane, "theta", "rad", fem_th, sol_e.theta(x), tol4, "r01 kGA_eff", k_e)]
        # reactions: ShaftResults.R_* follow system.bearings, forces of the support on the shaft along
        # +v, the sign of r01's R_A, R_B; the order of r01 is by position
        R_ref = {sol.x_A: sol.R_A, sol.x_B: sol.R_B}
        rows.append(row("C3", plane, "R", "N", getattr(result, r_key), np.array([R_ref[xs] for xs in support_x]),
                        TOL_NODAL, "r01", k))
        rows.append(row("R1", plane, "M", "N.mm", getattr(result, f"M_{plane.lower()}"), sol.M(x), None, "r01", k))
    return rows


def along_shaft(system, result, settings, *, kGA_override: float | None = None, series: str = "",
                n_dense: int = 401) -> list[dict]:
    """Deflection and rotation along the shaft: AxisForge nodes and the r01 curve, per loaded plane.

    Records of ``studies.outputs.l000_axisforge.plots.shaft.deflection_along_shaft``: ``plane``,
    ``series`` (the beam model, or ``series``), ``source`` ("FEM" at the nodes, "r01" on ``n_dense``
    points; for one-point Timoshenko on a uniform mesh with point loads also "r01 kGA_eff", the curve
    C4 judges), ``x_mm``, ``v_mm``, ``theta_rad``.
    """
    EI, kGA = section(system, settings, kGA_override=kGA_override)
    x_fem = np.asarray(result.x_nodes, dtype=float)
    lengths = np.diff(x_fem)
    curves = {"r01": reference(system, EI, kGA)}
    if (settings.integration_method == "single_point" and not system.distributed_radial_loads
            and np.allclose(lengths, lengths[0], rtol=1e-9, atol=0.0)):
        curves["r01 kGA_eff"] = reference(system, EI, kga_reduced_integration(kGA, EI, float(lengths[0])))
    x_ref = np.linspace(0.0, system.shaft.total_length, n_dense)
    name = series or theory_name(settings)
    out = []
    for plane in _loaded_planes(system):
        _, v_key, th_key, _ = PLANES[plane]
        out += [{"plane": plane, "series": name, "source": "FEM", "x_mm": float(xi), "v_mm": float(v),
                 "theta_rad": float(t)} for xi, v, t in zip(x_fem, getattr(result, v_key), getattr(result, th_key))]
        for source, sols in curves.items():
            sol = sols[plane]
            out += [{"plane": plane, "series": name, "source": source, "x_mm": float(xi), "v_mm": float(v),
                     "theta_rad": float(t)} for xi, v, t in zip(x_ref, sol.v(x_ref), sol.theta(x_ref))]
    return out


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
