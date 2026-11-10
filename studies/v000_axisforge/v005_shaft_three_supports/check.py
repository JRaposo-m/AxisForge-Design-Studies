"""Comparator of v005: a solved shaft on any number of supports (rigid, springs, settlements) against r01.

:func:`check` takes the ``ShaftSystem`` and the ``ShaftResults`` of a solve (``RigidSupportFEMSolver`` for
rigid supports, ``study.solve`` for springs and settlements, ``study.solve_hertz`` with the secant springs of a
converged nonlinear solve) and returns one record per quantity and bending plane. The judge is r01
``solve_supported`` (force method on the exact stepped beam, R5 G1) with the same per-section EI and kGA
(v001 ``segments``, AxisForge's kappa), the supports as point springs at the bearing centres, linear stiffness,
and the same settlements. This is the physics reviewer's list of conditions for r01 as judge (r01ext review).

Planes and signs (D-0010, v001 check): AxisForge solves XY (v along +Y, F_y) and XZ (v along +Z, F_z)
separately; r01 one plane (x, v) with forces, deflections, settlements and support forces along the same +v,
couples and theta from +x toward +v. Every quantity maps as it is, plane by plane: two planes = two r01 calls.
``R_*`` of ``ShaftResults`` is the force of the support ON THE SHAFT (``K d - f_ext``), r01's ``R``. The
couple of a rotational spring is the rotation row of ``f_*_reaction`` (``K d - f_ext`` of the beam), r01's ``C``.
A rotational spring acts on the cross-section rotation in both (the Timoshenko rotation DOF; r01 Support).

Units: mm, N, MPa, N.mm, rad on both sides.
"""
from __future__ import annotations

import math

import numpy as np

from studies.r00_analytical_references.r01_beams import Support, solve_supported
from studies.v000_axisforge.v001_shaft_uniform.check import PLANES, TOL_NODAL, segments, theory_name

__all__ = ["check", "reference", "supports", "along_shaft", "TOL_NODAL"]


def supports(system, plane: str, *, springs: dict | None = None, settlements_mm: dict | None = None) -> list[Support]:
    """r01 ``Support`` list of the bearings of ``system`` in one plane, by position.

    Parameters
    ----------
    system : axisforge ShaftSystem
        Its bearings give the positions [mm].
    plane : {"XY", "XZ"}
        Bending plane (selects the settlements).
    springs : dict, optional
        ``{x [mm]: (k_r [N/mm], k_theta [N.mm/rad])}``; ``k_r = inf`` is a rigid support. Not listed: a rigid
        pin (AxisForge's rotation is free at a bearing). The same spring in both planes (isotropic radial
        stiffness).
    settlements_mm : dict, optional
        ``{plane: {x [mm]: settlement along +v [mm]}}``.
    """
    springs = springs or {}
    settled = (settlements_mm or {}).get(plane, {})
    out = []
    for x in sorted(b.position for b in system.bearings):
        k_r, k_theta = springs.get(x, (math.inf, 0.0))
        out.append(Support(x, k_r=k_r, k_theta=k_theta, settlement=settled.get(x, 0.0)))
    return out


def reference(system, settings, *, springs: dict | None = None, settlements_mm: dict | None = None,
              h_mm: float | None = None) -> dict:
    """r01 ``solve_supported`` of each bending plane of ``system``: point forces, sections, supports.

    ``h_mm`` (uniform mesh, one-point Timoshenko): every segment gets the shear stiffness of the exact element
    equal to the one-point element (v001 ``kga_reduced_integration``), C4. No distributed loads or couples in
    v005 (``ValueError`` if present: the comparator would silently drop them).

    Returns
    -------
    dict
        ``{"XZ": SupportedSolution, "XY": SupportedSolution}``.
    """
    if system.distributed_radial_loads or system.external_moments:
        raise ValueError("v005 check takes point forces only")
    segs = segments(system, settings, h_mm=h_mm)
    out = {}
    for plane, (lp, *_) in PLANES.items():
        forces = [(ld.position, float(ld.component(lp))) for ld in system.radial_loads]
        out[plane] = solve_supported(forces, [], supports(system, plane, springs=springs, settlements_mm=settlements_mm),
                                     segs, system.shaft.total_length)
    return out


def check(system, result, settings, *, springs: dict | None = None, settlements_mm: dict | None = None,
          case: str = "v005", label: str = "", tag: str = "") -> list[dict]:
    """Comparison records of one solved shaft on several supports against r01, per loaded bending plane.

    Field errors are the largest nodal deviation over the largest absolute r01 value of that field in the
    plane; reactions over the largest absolute reaction. Tolerances (README section 6, fixed before the run):

    * C1 ``v`` [mm], C2 ``theta`` [rad], Euler-Bernoulli vs r01: ``TOL_NODAL`` (nodal exactness, v001);
    * C3 support forces ``R`` [N] and C3c spring couples ``C`` [N.mm], every model vs r01 at the model's own
      kGA (Euler-Bernoulli) or at kGA_eff (one-point Timoshenko on a uniform mesh): ``TOL_NODAL``. The beam is
      hyperstatic: the reactions depend on the stiffness, so unlike v001 they are judged against the
      reference of the same element;
    * C4 ``v``, ``theta``, one-point Timoshenko on a uniform mesh vs r01 at kGA_eff: ``TOL_NODAL``;
    * T1 ``v``, ``theta``, ``R``, Timoshenko vs r01 at the sections' kGA: RECORD (the O(h^2) element error).

    Parameters
    ----------
    system, result, settings
        The shaft, its ``ShaftResults``, the beam model of the solve.
    springs, settlements_mm
        As :func:`supports`, the inputs of the solve.
    case, label, tag : str
        Written into the records.

    Returns
    -------
    list of dict
        ``case``, ``label``, ``theory``, ``plane``, ``n_elements``, ``h_mm``, ``uniform_mesh``, ``id``,
        ``quantity``, ``unit``, ``max_abs_diff``, ``ref_scale``, ``rel_err``, ``tol``, ``verdict``,
        ``reference``, ``tag``.
    """
    x = np.asarray(result.x_nodes, dtype=float)
    lengths = np.diff(x)
    h = float(lengths.max())
    uniform = bool(np.all(np.abs(lengths - lengths[0]) <= 1e-9 * lengths[0]))
    timoshenko = settings.beam_theory == "timoshenko"
    exact_ref = not timoshenko or (settings.integration_method == "single_point" and uniform)
    refs = {"r01": reference(system, settings, springs=springs, settlements_mm=settlements_mm)}
    if timoshenko and exact_ref:
        refs["r01 kGA_eff"] = reference(system, settings, springs=springs, settlements_mm=settlements_mm, h_mm=h)
    judge = "r01 kGA_eff" if timoshenko else "r01"
    order = np.argsort([b.position for b in system.bearings])      # ShaftResults follow system.bearings
    nodes = [int(np.argmin(np.abs(x - b.position))) for b in system.bearings]
    common = {"case": case, "label": label or system.name, "theory": theory_name(settings), "n_elements": len(x) - 1,
              "h_mm": h, "uniform_mesh": uniform, "tag": tag}

    def row(cid, plane, quantity, unit, fem, ref, tol, ref_name):
        diff = float(np.max(np.abs(np.asarray(fem, dtype=float) - ref)))
        scale = float(np.max(np.abs(ref)))
        err = diff / scale if scale > 0.0 else diff
        verdict = "RECORD" if tol is None else ("PASS" if err <= tol else "MISS")
        return {**common, "plane": plane, "id": cid, "quantity": quantity, "unit": unit, "max_abs_diff": diff,
                "ref_scale": scale, "rel_err": err, "tol": math.nan if tol is None else tol, "verdict": verdict,
                "reference": ref_name}

    rows = []
    for plane, (lp, v_key, th_key, r_key) in PLANES.items():
        if not any(abs(ld.component(lp)) > 1e-9 * ld.magnitude for ld in system.radial_loads) and not any(
                (settlements_mm or {}).get(plane, {}).values()):
            continue                                                # an unloaded plane: all zeros, nothing to judge
        fem_v, fem_th = getattr(result, v_key), getattr(result, th_key)
        # FEM reactions in position order; spring couples from the rotation rows of K d - f_ext
        fem_R = np.asarray(getattr(result, r_key), dtype=float)[order]
        f_reac = np.asarray(getattr(result, f"f_{plane.lower()}_reaction"), dtype=float)
        fem_C = np.array([f_reac[3 * nodes[i] + 2] for i in order])
        for name, sols in refs.items():
            sol = sols[plane]
            tol = TOL_NODAL if (name == judge and exact_ref) else None
            if timoshenko and name == "r01":
                ids = ("T1", "T1", "T1", "T1")
            elif timoshenko:
                ids = ("C4", "C4", "C3", "C3c")
            else:
                ids = ("C1", "C2", "C3", "C3c")
            rows += [row(ids[0], plane, "v", "mm", fem_v, sol.v(x), tol, name),
                     row(ids[1], plane, "theta", "rad", fem_th, sol.theta(x), tol, name),
                     row(ids[2], plane, "R", "N", fem_R, sol.R, tol, name)]
            if np.any(sol.C != 0.0):
                rows.append(row(ids[3], plane, "C (spring couple)", "N.mm", fem_C, sol.C, tol, name))
    return rows


def along_shaft(system, result, settings, *, springs: dict | None = None, settlements_mm: dict | None = None,
                series: str = "", plane: str = "XY", n_dense: int = 401) -> list[dict]:
    """Deflection and rotation along the shaft in one plane: AxisForge nodes and the r01 curve.

    Records of ``studies.outputs.l000_axisforge.plots.shaft.deflection_along_shaft``: ``series`` (the beam
    model, or ``series``), ``source`` ("FEM" at the nodes, "r01" on ``n_dense`` points; one-point Timoshenko on
    a uniform mesh: "r01 kGA_eff", the curve C4 judges), ``x_mm``, ``v_mm``, ``theta_rad``.
    """
    _, v_key, th_key, _ = PLANES[plane]
    x_fem = np.asarray(result.x_nodes, dtype=float)
    h = float(np.diff(x_fem).max())
    one_point = settings.beam_theory == "timoshenko" and settings.integration_method == "single_point"
    sol = reference(system, settings, springs=springs, settlements_mm=settlements_mm,
                    h_mm=h if one_point else None)[plane]
    x_ref = np.linspace(0.0, system.shaft.total_length, n_dense)
    name = series or theory_name(settings)
    source = "r01 kGA_eff" if one_point else "r01"
    out = [{"series": name, "source": "FEM", "x_mm": float(xi), "v_mm": float(v), "theta_rad": float(t)}
           for xi, v, t in zip(x_fem, getattr(result, v_key), getattr(result, th_key))]
    out += [{"series": name, "source": source, "x_mm": float(xi), "v_mm": float(v), "theta_rad": float(t)}
            for xi, v, t in zip(x_ref, sol.v(x_ref), sol.theta(x_ref))]
    return out
