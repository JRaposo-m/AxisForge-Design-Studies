"""Load-distribution model of one rolling bearing (ISO/TS 16281 through AxisForge), shared by the bearing studies.

s001 owns it; s002, s003, s004 and s102 import it through ``s001.__init__`` (D-0036 amendment item 11).
Ported from ``s001_bearing_load_distribution/_common/{solve,rows}.py`` at the tag ``studies-pre-sweep``,
without the shaft FEM build, plots, reports, tilt sweep and stiffness derivatives.

The model is AxisForge's: the dispatched ``ISO16281BallSolver`` / ``ISO16281RollerSolver`` solve the
radial and axial force balance of the inner ring for the ring displacements delta_r, delta_a at a
prescribed tilt psi (ISO/TS 16281, 4.2.2-4.2.3, Eqs (12)-(18), (38)-(41)). What is added here:

* :func:`load_node`: a bearing node carrying a known load, so one bearing can be solved without a
  shaft FEM (the solver reads only ``shaft_results.bearing_nodes``);
* :func:`solve_bearing`: the solve, with the post-processing (stiffness, L10r) dropped when it fails;
* :func:`loaded_zone`: the edge of the loaded zone and Harris's load-zone factor epsilon, from the
  AxisForge kinematics (AxisForge has no epsilon field);
* :func:`records`: one case record and one record per rolling element, units in the keys.

Azimuth. ``phi_j`` is LOCAL: 0 on the line of the radial load, where the most loaded element is
(``families/family.py``, linspace(0, 2 pi, Z)); it is convention-free. Where the zone sits in the shaft
frame (``phi_j_global = phi_j + phi_Fr``) depends on the sign of the force fed to the node: AxisForge
puts the support force on the shaft there (``contact_solver.py:192-194``), so a global azimuth is
``convention-dependent (R-003)``. This model records local azimuths only.

Units: mm, N, N.mm, rad in the solver; degrees only in the records (``*_deg``).
"""
from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import numpy as np
from axisforge.outputs.solvers.bearings import bearing_analysis_result as af_analysis
from axisforge.outputs.solvers.bearings.load_distribution import load_distribution_results as af_ld
from axisforge.results.fem_results.shaft_results import BearingNodeData
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import resolve_solver_cls

# importing contact_solver registers the ball and roller solvers with the dispatcher (side effect only)
importlib.import_module("axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver")

__all__ = ["load_node", "solve_bearing", "loaded_zone", "records", "BALANCE_RTOL"]

# [-] force balance accepted as solved: |residual| <= 1e-6 max(Fr, Fa). AxisForge's ``ok`` is the root
# finder's flag and is not enforced (contact_solver.solve_contact, Warnings); 1e-6 is the r03 rule
BALANCE_RTOL = 1e-6


def load_node(bearing, Fr_N: float, Fa_N: float = 0.0, *, phi_Fr_rad: float = 0.0) -> BearingNodeData:
    """A bearing node carrying a known load, the input the AxisForge contact solver reads from a shaft FEM.

    Parameters
    ----------
    bearing : axisforge Bearing
        Its ``label`` names the node.
    Fr_N : float
        Radial load [N], >= 0, along ``phi_Fr_rad``.
    Fa_N : float
        Axial load [N]; > 0 loads an angular contact bearing in its contact direction.
    phi_Fr_rad : float
        Direction of the radial load, from +Z toward +Y [rad] (D-0010). It does not enter the
        equilibrium, only ``phi_j_global``; 0 makes the global azimuth equal the local one.

    Returns
    -------
    BearingNodeData
        ``Fr_xz = Fr cos(phi_Fr)``, ``Fr_xy = Fr sin(phi_Fr)``, ``Fa``; zero displacements and slopes
        (the solver then starts from its own guesses). With a real shaft FEM use its
        ``ShaftResults.bearing_nodes`` instead (the support force on the shaft: R-003).
    """
    return BearingNodeData(label=bearing.label, Fr_xz=Fr_N * math.cos(phi_Fr_rad), Fr_xy=Fr_N * math.sin(phi_Fr_rad),
                           Fr=Fr_N, Fa=Fa_N)


def solve_bearing(bearing, node: BearingNodeData, *, psi_rad: float | None = 0.0, postprocess: bool = True,
                  tol: float | None = None):
    """ISO/TS 16281 load distribution of ONE bearing at the loads of its node (AxisForge solver).

    Parameters
    ----------
    bearing : axisforge Bearing
        Assembled with ``contact=ContactAnalysis.ISO16281``.
    node : BearingNodeData
        From :func:`load_node` or a shaft FEM (``ShaftResults.bearing_nodes``).
    psi_rad : float or None
        Prescribed misalignment of the rings in the load plane [rad]; None keeps the node's slopes.
    postprocess : bool
        Also the secant stiffness and the ISO/TS 16281 reference life L10r. It fails on its own
        when no element is loaded; the solve is then repeated without it (old ``solve_bearing``).
    tol : float or None
        Tolerance of the solver's root find (``run_root``, a relative step for hybr) [-]; None keeps AxisForge's
        ``SOLVER_TOLERANCE`` (1e-6). s005 passes 1e-12 for a clean finite-difference Jacobian of Mz.

    Returns
    -------
    analysis : BearingAnalysisResult or None
        None if the solve itself failed.
    postprocessed : bool
        False if ``analysis`` has no stiffness / life.
    error : str
        "" or what failed.
    """
    label = bearing.label
    cls = resolve_solver_cls(bearing, label=label)
    psi_override = None if psi_rad is None else {label: psi_rad}
    nodes = SimpleNamespace(bearing_nodes=[node])            # the solver reads only bearing_nodes
    error = ""
    for post in ((True, False) if postprocess else (False,)):
        try:
            kwargs = {} if tol is None else {"tol": tol}
            analysis = cls(psi_input=psi_rad is not None, postprocess=post, **kwargs).solve(
                None, {label: bearing}, nodes, psi_override=psi_override)[label]
            return analysis, post, error
        except Exception as exc:                                     # noqa: BLE001 -- recorded, not hidden
            error += f"{'postprocess' if post else 'solve'} failed: {type(exc).__name__}: {exc}; "
    return None, False, error


def loaded_zone(bearing, delta_r_mm: float, delta_a_mm: float = 0.0) -> tuple[float, float]:
    """Half-angle of the loaded zone and Harris's load-zone factor epsilon, from the solver's kinematics.

    Ball (ISO/TS 16281 Eq. (12)): element j is in contact while
    ``sqrt(U^2 + V^2) > A`` with ``U = A cos(alpha_0) + delta_r cos(phi)``, ``V = A sin(alpha_0) + delta_a``,
    so the edge is ``cos(psi_l) = (sqrt(A^2 - V^2) - A cos(alpha_0)) / delta_r``. Roller (Eq. (38), rigid
    kinematics of the roller axis): ``delta_r cos(phi) - s/2 > 0``, ``cos(psi_l) = s / (2 delta_r)``; the
    ISO reference profile shifts the edge inward by 2 min P(x_k) / delta_r in cos(psi_l) (P(x_k) >= 2.4e-6 mm
    at the middle laminae of the NU210-like build, so < 5e-4 here; 40 laminae leave none at x = 0). Then
    ``epsilon = (1 - cos(psi_l)) / 2`` (Harris2007_vol1 Eqs 7.10-7.15, PDF p. 156-157, written with psi_l: 0.5
    at psi_l = 90 deg, 1 at 180 deg). Untilted rings (psi = 0) only.

    Parameters
    ----------
    bearing : axisforge Bearing
        Ball: ``A``, ``alpha_0``; roller: ``s`` [mm].
    delta_r_mm, delta_a_mm : float
        Ring displacements of the solve [mm].

    Returns
    -------
    psi_l_rad : float
        Zone half-angle [rad], 0 (nothing loaded) to pi (all loaded), continuous in the displacements.
    epsilon : float
        Load-zone factor [-], capped at 1 (all elements loaded); Harris's epsilon (Eq. 7.61) is > 1 there,
        so a 1 means "at least full loading", not how far past it.
    """
    if delta_r_mm <= 0.0:
        return math.nan, math.nan
    if hasattr(bearing, "A"):                                       # point contact
        A, a0 = bearing.A, bearing.alpha_0
        V = A * math.sin(a0) + delta_a_mm
        if abs(V) >= A:                                             # axial approach alone closes every contact
            return math.pi, 1.0
        c = (math.sqrt(A * A - V * V) - A * math.cos(a0)) / delta_r_mm
    else:                                                           # line contact, diametral clearance s
        c = bearing.s / (2.0 * delta_r_mm)
    psi_l = math.acos(min(1.0, max(-1.0, c)))
    return psi_l, 0.5 * (1.0 - math.cos(psi_l))


def records(bearing, analysis, *, postprocessed: bool = True, error: str = "", **ident) -> tuple[dict, list[dict]]:
    """One case record and the element records of a solved bearing (AxisForge's own record readers).

    Case fields: ``ident`` (the swept inputs, e.g. ``kind``, ``s_mm``, ``Fa_over_Fr``), ``label``, ``Z``,
    ``s_mm`` and ``alpha_0_deg`` of the bearing (for an angular contact bearing s is derived from alpha_0),
    ``Fr_N``, ``Fa_N``, ``psi_mrad`` (the imposed tilt), ``Mz_Nmm`` (the external moment on the rings: ball
    Eq. (18), roller Eq. (46), each in its own sign convention, R5-76), ``delta_r_mm``, ``delta_a_mm``,
    ``Q_max_N``, ``n_loaded``, ``psi_l_deg`` and ``epsilon`` (untilted rings only, else NaN),
    ``alpha_max_deg`` (at Q_max), ``alpha_guard`` (every loaded ball has alpha_j of the sign of the reference
    direction: alpha_0 for an angular contact bearing, whose groove carries axial load one way only; Fa for any
    other ball bearing, whose symmetric groove runs at alpha < 0 legitimately under Fa < 0; None when no axial
    load is applied: SB-45, R5-44), ``ok``, ``residual_N``, ``balanced`` (residual <= ``BALANCE_RTOL``
    max(Fr, Fa)), ``n_iter`` (residual evaluations), ``L10r_Mrev`` (ISO/TS 16281
    reference life, not ISO 281), ``postprocessed``, ``error``.
    Element fields: ``ident``, ``j``, ``phi_local_deg`` (0 on the load line), ``delta_mm``, ``alpha_deg``, ``Q_N``.
    """
    case = {"label": bearing.label, "Z": bearing.Z, "s_mm": float(getattr(bearing, "s", math.nan)),
            "alpha_0_deg": math.degrees(getattr(bearing, "alpha_0", 0.0)), "postprocessed": postprocessed,
            "error": error, "ok": False}
    case.update(ident)                      # the swept values win over the values read back
    if analysis is None:
        return case, []
    rec = af_analysis.analysis_record(analysis)
    elements = af_ld.element_records(analysis.load_distribution)          # single row: row 0
    Q = np.array([e["Q_N"] for e in elements])
    alpha = np.array([e.get("alpha_rad", 0.0) for e in elements])
    loaded = Q > 0.0
    j_max = int(np.argmax(Q))
    # the zone edge formula is for untilted rings: a tilted bearing gets no psi_l / epsilon
    psi_l, eps = (loaded_zone(bearing, rec["delta_r_mm"], rec["delta_a_mm"]) if rec["psi_rad"] == 0.0
                  else (math.nan, math.nan))
    point = hasattr(bearing, "A")
    ref = bearing.alpha_0 if bearing.family.name == "angular_contact" else rec["Fa_N"]     # R5-44
    guard = bool(np.all(alpha[loaded] * ref > 0.0)) if point and rec["Fa_N"] != 0.0 else None
    load = max(rec["Fr_N"], abs(rec["Fa_N"]))
    case.update(Fr_N=rec["Fr_N"], Fa_N=rec["Fa_N"], psi_mrad=1e3 * rec["psi_rad"], Mz_Nmm=rec["Mz_Nmm"],
                delta_r_mm=rec["delta_r_mm"], delta_a_mm=rec["delta_a_mm"],
                Q_max_N=float(Q.max()), n_loaded=int(loaded.sum()), psi_l_deg=math.degrees(psi_l), epsilon=eps,
                alpha_max_deg=math.degrees(alpha[j_max]) if point else math.nan, alpha_guard=guard,
                ok=bool(rec["ok"]), residual_N=rec["residual"],
                balanced=bool(rec["residual"] <= BALANCE_RTOL * load), n_iter=rec["n_iter"],
                L10r_Mrev=rec.get("L10r_Mrev", math.nan))
    rows = [{**ident, "label": bearing.label, "j": e["j"], "phi_local_deg": math.degrees(e["phi_rad"]),
             "delta_mm": e.get("delta_mm", math.nan), "alpha_deg": math.degrees(e.get("alpha_rad", math.nan)),
             "Q_N": e["Q_N"]} for e in elements]
    return case, rows
