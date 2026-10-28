"""Run of v005: the shaft on three supports through the AxisForge FEM, against r01 extended, Harris and itself.

``python -m studies.v000_axisforge.v005_shaft_three_supports.study`` writes into ``results/`` (git-ignored):
``v005_first_check.csv`` (fed-back bearing force sign, R-003), ``v005_check.csv`` (FEM vs r01 records of every
linear solve), ``v005_richardson.csv``, ``v005_stiffness_sweep.csv`` (reactions vs the middle spring),
``v005_literature.csv`` (Harris Eqs 9.42-9.44 and 9.48), ``v005_hertz.csv`` (the nonlinear solves: Newton,
scipy root, secant r01, fed-back check), ``v005_newton_history.csv`` and the figures of ``studies/outputs``.

Three parts (README 3):

1. Linear: AxisForge's stiffness K with springs added by its own ``StiffnessMatrixBuilder.add_nodal_stiffness``
   and settlements imposed by elimination (:func:`solve`), against r01 ``solve_supported``.
2. Stiff shaft on three Hertz bearings against Harris Eq. 9.48 (EI x scale, scale -> inf).
3. Flexible shaft on three Hertz bearings (:func:`solve_hertz`, ``numerical_solvers.newton``): self-consistency
   against ``scipy.optimize.root``, against r01 with the secant springs of the converged state, and through
   AxisForge with the middle bearing force fed back as a ``RadialLoad`` (R-003).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.core.loads import LoadPlane, RadialLoad
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.outputs._format import write_records_csv
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.build_stiffness_matrix import StiffnessMatrixBuilder
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import build_shaft_result
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (RigidSupportFEMSolver,
                                                                                               RigidSupportSolution)
from scipy.optimize import root

from numerical_solvers import NewtonOptions, RichardsonOptions, newton, richardson
from studies.outputs.common.plots.convergence import newton_history
from studies.outputs.l000_axisforge.plots import shaft as plots
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.r00_analytical_references.r01_beams import Support, solve_supported
from studies.v000_axisforge.v001_shaft_uniform.check import node_value, richardson_verdict, segments
from studies.v000_axisforge.v001_shaft_uniform.construction import uniform_nodes

from . import literature
from .check import TOL_NODAL, along_shaft, check, reference
from .construction import (K_HERTZ, K_MID_N_PER_MM, LENGTH_MM, LOAD_P, LOAD_Q, MODELS, N_ELEMENTS, N_HERTZ,
                           SETTLEMENTS_MM, SPRINGS, SUPPORTS_MM, stiff_material, three_support_shaft)

RESULTS = Path(__file__).resolve().parent / "results"
R003 = "convention-dependent (R-003)"
EB = MODELS["euler_bernoulli"]
# formal order of the nodal error, the max_p of Richardson (R3-17): Hermite cubic (nodally exact, flat);
# one-point Timoshenko O(h^2) (v001 kga_reduced_integration)
ORDER = {"euler_bernoulli": 4.0, "timoshenko/cowper/single_point": 2.0}
# OPEN R5-154: C6 (Timoshenko extrapolated within 1e-9 of r01, from v002) assumes v(h) = a + c h^2; in a
# hyperstatic beam the shear flexibility also enters the reactions, so r01 at kGA_eff itself extrapolates 3e-9 off
FLAT_RTOL = 1e-8                     # [-] Richardson "flat" (numerical_solvers default), C5
# EI x scale of the stiff-shaft series (Harris Eq. 9.45: I -> inf); judged at SCALE_JUDGED (H1)
EI_SCALES = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
SCALE_JUDGED = 1e5
TOL_RIGID = 1e-4                     # [-] H1: Harris 9.48 at EI x 1e5 (README 6)
H2_BAND = (0.8, 1.25)                # [-] H2: err x scale at 1e5 over that at 1e3, O(1/scale)
# H1 / H2 were set after a scratch prototype of the stiff series, not before any run: in this run they are RECORD
# (physics review MUST 1); the values above are the criteria fixed for the next run
CRITERIA_NEXT_RUN = ("H1", "H2")
TOL_SELF = 1e-8                      # [-] N2: Newton vs scipy root, bearing forces (README 6)
# middle spring k / K_MID, from the two-support limit (k -> 0) to the rigid one (k -> inf)
K_RATIOS = tuple(10.0 ** e for e in np.arange(-4.0, 4.01, 0.5))
# Newton stopping test on the residual scaled by P: 1e-10 = 1e-7 N. Measured floor of K_cond v at EI x 1e6:
# ~1e-12 scaled (1e-9 N); the bound eps ||K_cond|| |v| (2e-16 x 1e11 N/mm x 1e-2 mm = 2e-7 N) is pessimistic.
# 1e-10 is reached at every scale (6.7e-11 at 1e6)
NEWTON = dict(residual_abs=1e-10, step_rel=1e-13, max_iter=50)
# OPEN R5-155 (note): below the floor newton ends line_search_failed at the solution (Armijo cannot hold on
# round-off). Rerun at 1e-12 (physics review): s = 1, 1e3 converge; s = 1e6 stalls at 1.2e-12
CAPTION = (f"uniform shaft L {LENGTH_MM:g} mm, d 30 mm, supports at {'/'.join(f'{x:g}' for x in SUPPORTS_MM)} mm; "
           f"P {LOAD_P[1]:g} N at {LOAD_P[0]:g} mm (-Y), Q {LOAD_Q[1]:g} N at {LOAD_Q[0]:g} mm (+Z)")


def _shaft_result(r, system, d_xz, d_xy):
    """``ShaftResults`` of prescribed nodal displacements on AxisForge's mesh, through its own post-processing.

    ``K`` is the BEAM stiffness of the rigid solve ``r`` (no springs), so the reactions ``K d - f_ext`` are the
    forces of the supports and springs on the shaft (``build_shaft_result``: ``R_*`` at the bearing v DOFs).
    """
    K = r.K
    f_tot = {"xz": K @ d_xz, "xy": K @ d_xy}
    sol = RigidSupportSolution(x_nodes=list(r.x_nodes), elements=r.elements, K=K, free_dofs=list(r.free_dofs),
                               constrained_dofs=list(r.constrained_dofs), d_total_xz=d_xz, d_total_xy=d_xy,
                               f_xz_ext=r.f_xz_ext, f_xy_ext=r.f_xy_ext, f_xz_total=f_tot["xz"], f_xy_total=f_tot["xy"],
                               f_xz_reaction=f_tot["xz"] - r.f_xz_ext, f_xy_reaction=f_tot["xy"] - r.f_xy_ext)
    return build_shaft_result(sol, system)


def solve(system, settings, n_elements: int | None = None, *, springs: dict | None = None,
          settlements_mm: dict | None = None):
    """AxisForge shaft FEM with linear support springs and prescribed settlements, on AxisForge's own K.

    ``RigidSupportFEMSolver`` gives the mesh, the elements, the load vectors and the DOF partition of rigid
    pins; it has no spring or settlement input (``# OPEN R5-150``). The springs are added with AxisForge's
    ``StiffnessMatrixBuilder.add_nodal_stiffness`` (``K[3i + slot, 3i + slot] += k``: slot 1 the transverse
    DOF, slot 2 the cross-section rotation) on a rebuild of the same mesh and elements; the transverse DOF of a
    radial spring is freed. Settlements: a rigid support prescribes ``v = s`` (elimination, ``K_ff d_f = f_f -
    K_fc d_c``, Onate2013_vol2 1.3.4 Eq. (1.22), as v002 ``study.solve``); a spring of base settlement ``s``
    adds ``k s`` to the load of its DOF (spring force ``-k (v - s)``). Both planes share the springs (isotropic
    radial stiffness). The result goes through AxisForge's ``build_shaft_result``.

    Parameters
    ----------
    system : axisforge ShaftSystem
        Shaft, bearings (every support position), point forces.
    settings : axisforge BeamModelSettings
        Beam model.
    n_elements : int, optional
        Uniform divisions of the shaft (v001 ``uniform_nodes``); None: the mandatory nodes only.
    springs : dict, optional
        ``{x [mm]: (k_r [N/mm] or inf, k_theta [N.mm/rad])}`` (check ``supports``).
    settlements_mm : dict, optional
        ``{plane: {x [mm]: settlement along +v [mm]}}``.

    Returns
    -------
    ShaftResults
        ``R_*`` the support forces on the shaft along +v; the spring couples in the rotation rows of
        ``f_*_reaction``.
    """
    springs, settlements_mm = springs or {}, settlements_mm or {}
    extra = uniform_nodes(n_elements, system.shaft.total_length) if n_elements else []
    r = RigidSupportFEMSolver(settings).solve(system, extra_mandatory=extra)
    x = np.asarray(r.x_nodes, dtype=float)
    builder = StiffnessMatrixBuilder(Mesh1D(system, extra_mandatory=extra), r.elements, frame=True)
    node = {b.position: int(np.argmin(np.abs(x - b.position))) for b in system.bearings}
    keys = set(springs) | {xs for plane in settlements_mm.values() for xs in plane}
    if not keys <= set(node):                                       # a mistyped key would be a bare KeyError
        raise ValueError(f"spring / settlement positions {sorted(keys - set(node))} are not bearing positions "
                         f"{sorted(node)} [mm]")
    freed = []
    for xs, (k_r, k_theta) in springs.items():
        if math.isfinite(k_r):
            builder.add_nodal_stiffness(xs, np.array([[k_r]]), dof_slots=[1])
            freed.append(3 * node[xs] + 1)                      # the spring, not a constraint, holds this DOF
        if k_theta > 0.0:
            builder.add_nodal_stiffness(xs, np.array([[k_theta]]), dof_slots=[2])
    K_s = builder.build()
    fixed = [d for d in r.constrained_dofs if d not in freed]
    free = [d for d in range(3 * len(x)) if d not in fixed]
    d = {}
    for plane in ("xz", "xy"):
        f = np.array(getattr(r, f"f_{plane}_ext"), dtype=float)
        dp = np.zeros_like(f)
        for xs, s in settlements_mm.get(plane.upper(), {}).items():
            k_r = springs.get(xs, (math.inf, 0.0))[0]
            if math.isfinite(k_r):
                f[3 * node[xs] + 1] += k_r * s                  # spring base moved by s: force k (s - v) on the shaft
            else:
                dp[3 * node[xs] + 1] = s                        # rigid support: v prescribed
        dp[free] = np.linalg.solve(K_s[np.ix_(free, free)], f[free] - K_s[np.ix_(free, fixed)] @ dp[fixed])
        d[plane] = dp
    return _shaft_result(r, system, d["xz"], d["xy"])


def hertz_problem(system, settings, n_elements: int, *, K: float = K_HERTZ, n: float = N_HERTZ, plane: str = "XY"):
    """The shaft on one Hertz law per bearing, condensed to the bearing deflections: residual, Jacobian, start.

    Every bearing of ``system`` is a radial law ``F = -K sign(v) |v|^n`` on the shaft along +v (Harris2007_vol2
    Eq. 9.46 with Harris's delta = -v, signed: a radial bearing without clearance resists both ways). The beam
    stiffness K_beam and load f_ext are AxisForge's (one ``RigidSupportFEMSolver`` call; only u at the locating
    bearing stays fixed). Static condensation on the bearing transverse DOFs m (s = the other free DOFs):
    ``K_c = K_mm - K_ms K_ss^-1 K_sm``, ``f_c = f_m - K_ms K_ss^-1 f_s`` (exact: K_ss is the shaft on the
    bearings as pins, non-singular), and the equilibrium of the bearing nodes is

        g(v) = K_c v - f_c + K sign(v) |v|^n = 0      [N],     dg/dv = K_c + diag(n K |v|^(n-1)).

    The Jacobian is singular at v = 0 (zero Hertz stiffness), so the start is the solve with linear springs of
    the secant stiffness at P / (number of bearings).

    Returns
    -------
    dict
        ``residual``, ``jacobian`` (callables of v [mm]), ``x0`` [mm], ``x_scale`` [mm], ``f_scale`` [N],
        ``positions`` [mm] (bearings in ``system.bearings`` order), ``force`` (v -> bearing forces on the shaft
        [N]), ``expand`` (v -> ShaftResults of the full shaft).
    """
    extra = uniform_nodes(n_elements, system.shaft.total_length)
    r = RigidSupportFEMSolver(settings).solve(system, extra_mandatory=extra)
    x = np.asarray(r.x_nodes, dtype=float)
    p = plane.lower()
    f = np.asarray(getattr(r, f"f_{p}_ext"), dtype=float)
    positions = [b.position for b in system.bearings]
    m = [3 * int(np.argmin(np.abs(x - xb))) + 1 for xb in positions]
    axial = [dof for dof in r.constrained_dofs if dof % 3 == 0]     # u at the locating bearing
    s = [dof for dof in range(3 * len(x)) if dof not in m and dof not in axial]
    Kb = r.K
    K_ss = Kb[np.ix_(s, s)]
    S = np.linalg.solve(K_ss, np.column_stack([Kb[np.ix_(s, m)], f[s]]))
    K_c = Kb[np.ix_(m, m)] - Kb[np.ix_(m, s)] @ S[:, :-1]
    f_c = f[m] - Kb[np.ix_(m, s)] @ S[:, -1]
    # load scale [N]: v005 has forces only; with a moment load take the force DOFs (slot 1) only (N vs N.mm)
    P = float(np.sum(np.abs(f)))

    def force(v):                                                   # bearing forces on the shaft along +v [N]
        v = np.asarray(v, dtype=float)
        return -K * np.sign(v) * np.abs(v) ** n

    def residual(v):
        return K_c @ v - f_c - force(v)

    def jacobian(v):
        return K_c + np.diag(n * K * np.abs(np.asarray(v, dtype=float)) ** (n - 1.0))

    F0 = P / len(m)
    d0 = (F0 / K) ** (1.0 / n)                                      # [mm] deflection of one bearing at F0
    x0 = np.linalg.solve(K_c + (F0 / d0) * np.eye(len(m)), f_c)

    def expand(v):
        d = np.zeros(3 * len(x))
        d[m] = v
        d[s] = S[:, -1] - S[:, :-1] @ v                             # d_s = K_ss^-1 (f_s - K_sm v)
        zero = np.zeros(3 * len(x))
        return _shaft_result(r, system, *((zero, d) if p == "xy" else (d, zero)))

    return {"residual": residual, "jacobian": jacobian, "x0": x0, "x_scale": np.full(len(m), d0),
            "f_scale": np.full(len(m), P), "positions": positions, "force": force, "expand": expand}


def solve_hertz(system, settings, n_elements: int, *, K: float = K_HERTZ, n: float = N_HERTZ, label: str = ""):
    """Newton-Raphson (``numerical_solvers.newton``, analytic Jacobian, line search) on :func:`hertz_problem`.

    Returns
    -------
    (ShaftResults, NewtonResult, dict)
        The converged shaft, the Newton result (iterations, history, reason) and the problem.
    """
    prob = hertz_problem(system, settings, n_elements, K=K, n=n)
    res = newton(prob["residual"], prob["x0"], jacobian=prob["jacobian"],
                 options=NewtonOptions(x_scale=prob["x_scale"], f_scale=prob["f_scale"], **NEWTON),
                 name="v005_hertz", label=label)
    return prob["expand"](res.x), res, prob


def _rec(cid: str, quantity: str, unit: str, value, judge, scale: float, tol: float | None, *, case="v005",
         source: str = "", tag: str = "", **extra) -> dict:
    """One record of the case table: value, judge, largest abs difference over ``scale``, verdict."""
    diff = float(np.max(np.abs(np.subtract(value, judge))))
    err = diff / scale
    verdict = "RECORD" if tol is None else ("PASS" if err <= tol else "MISS")
    return {"case": case, "id": cid, "quantity": quantity, "unit": unit,
            "value": float(np.max(np.abs(value))) if np.ndim(value) else float(value),
            "judge": float(np.max(np.abs(judge))) if np.ndim(judge) else float(judge), "abs_diff": diff,
            "ref_scale": scale, "rel_err": err, "tol": math.nan if tol is None else tol, "verdict": verdict,
            "source": source, "tag": tag, **extra}


def first_check(n_elements: int = 27) -> list[dict]:
    """F0: the sign of a bearing force fed back to the shaft, two pins, one known load (``convention-dependent (R-003)``).

    Shaft on pins at 10 and 260 mm, P = 1000 N at 50 mm at theta 210 deg (from +Y toward +Z: F_y = -866 N,
    F_z = -500 N, both planes). Records:

    * F0a statics: ``R_A + R_B = -F`` per plane (``R`` = support force on the shaft): ``TOL_NODAL``;
    * F0b fed back: the same shaft with ``RadialLoad`` at pin B of the force ``R_B`` AS REPORTED (magnitude
      ``hypot(R_xy, R_xz)``, theta ``atan2(R_xz, R_xy)``): pin B then carries nothing and pin A is unchanged,
      so a bearing force fed back to the shaft is ``+R`` (AxisForge's sign, no flip): ``TOL_NODAL``;
    * F0c RECORD: fed back with the opposite sign (``-R``, the load on the bearing): R_B doubles;
    * F0d RECORD: ``phi_Fr = atan2(Fr_xy, Fr_xz)`` of ``bearing_nodes`` (from +Z toward +Y, the bearing
      solver's load direction) minus the direction of the applied load in the same measure: 180 deg. The
      bearing solver reads the support force, not the push of the shaft on the inner ring (R-003, bearing API
      cartographer, ``global_postprocessing._bearing_node_data`` Warnings).
    """
    x_A, x_B = SUPPORTS_MM[0], SUPPORTS_MM[-1]
    theta = 210.0
    load = RadialLoad(LOAD_P[0], LOAD_P[1], theta_deg=theta, label="P")

    def run(extra_loads, label):
        ss = three_support_shaft(supports_mm=(x_A, x_B), loads=[load, *extra_loads], label=label)
        return RigidSupportFEMSolver(EB).solve(ss, extra_mandatory=uniform_nodes(n_elements, LENGTH_MM))

    base = run([], "v005_f0")
    R = {"XY": np.asarray(base.R_xy), "XZ": np.asarray(base.R_xz)}
    out = [_rec("F0a", f"R_A + R_B vs -F ({p})", "N", float(R[p].sum()), -float(load.component(lp)), LOAD_P[1],
                TOL_NODAL, source="statics", tag=R003, plane=p) for p, lp in (("XY", LoadPlane.XY), ("XZ", LoadPlane.XZ))]
    RB = (float(R["XY"][1]), float(R["XZ"][1]))
    fed = run([RadialLoad(x_B, math.hypot(*RB), theta_deg=math.degrees(math.atan2(RB[1], RB[0])), label="fed+R",
                          source="bearing_reaction")], "v005_f0_fed")
    flipped = run([RadialLoad(x_B, math.hypot(*RB), theta_deg=math.degrees(math.atan2(-RB[1], -RB[0])), label="fed-R",
                              source="bearing_reaction")], "v005_f0_flipped")
    scale = math.hypot(*RB)
    out.append(_rec("F0b", "R_B after feeding back +R_B (XY, XZ)", "N", [fed.R_xy[1], fed.R_xz[1]], [0.0, 0.0], scale,
                    TOL_NODAL, source="statics", tag=R003))
    out.append(_rec("F0b", "R_A after feeding back +R_B (XY, XZ)", "N", [fed.R_xy[0], fed.R_xz[0]],
                    [R["XY"][0], R["XZ"][0]], scale, TOL_NODAL, source="statics", tag=R003))
    out.append(_rec("F0c", "R_B after feeding back -R_B (XY, XZ)", "N", [flipped.R_xy[1], flipped.R_xz[1]],
                    [2 * RB[0], 2 * RB[1]], scale, None, source="statics", tag=R003))
    node = base.bearing_nodes[1]
    phi_Fr = math.degrees(math.atan2(node.Fr_xy, node.Fr_xz))
    load_dir = math.degrees(math.atan2(load.Fy, load.Fz))           # the applied load, from +Z toward +Y
    out.append(_rec("F0d", "phi_Fr - direction of the load (from +Z toward +Y)", "deg",
                    (phi_Fr - load_dir) % 360.0, 180.0, 180.0, None, source="bearing_nodes", tag=R003))
    return out


def literature_series(n_elements: int = 27) -> list[dict]:
    """L1-L3: AxisForge on three rigid pins (EB, P only) vs Harris Eqs 9.42-9.44 (``TOL_NODAL`` of P)."""
    sp = literature.spans()
    ss = literature.harris_shaft()
    r = RigidSupportFEMSolver(EB).solve(ss, extra_mandatory=uniform_nodes(n_elements, LENGTH_MM))
    order = np.argsort([b.position for b in ss.bearings])
    F = np.asarray(r.R_xy)[order]                                   # Harris's F (up) = R_xy for P along -Y
    H = literature.harris_eqs_9_42_9_44(**sp)
    src = "Harris2007_vol2 Eqs 9.42-9.44, PDF p. 285"
    return [_rec(f"L{i + 1}", f"F_{i + 1} = R_xy", "N", F[i], H[i], sp["P"], TOL_NODAL, source=src, n_elements=n_elements)
            for i in range(3)]


def mesh_series() -> tuple[list[dict], list[dict], dict]:
    """The base linear case (springs, rotational spring, settlements, two planes) on every mesh and model.

    Returns check records, Richardson records (v_xy at the load P, ``kind="count"``, ``flat_atol=0``,
    ``max_p`` = the formal order, R3-17) and the solves on the coarsest mesh.
    """
    system = three_support_shaft()
    checks, conv, coarse = [], [], {}
    for name, settings in MODELS.items():
        def evaluate(n, settings=settings, name=name):
            r = solve(system, settings, int(n), springs=SPRINGS, settlements_mm=SETTLEMENTS_MM)
            checks.extend({**c, "sweep": "mesh"} for c in check(system, r, settings, springs=SPRINGS,
                                                               settlements_mm=SETTLEMENTS_MM))
            if int(n) == N_ELEMENTS[0]:
                coarse[name] = r
            return {"v_P_xy_mm": node_value(r, LOAD_P[0], "v_xy")}
        res = richardson(evaluate, N_ELEMENTS, options=RichardsonOptions(kind="count", flat_atol=0.0, flat_rtol=FLAT_RTOL,
                                                                         max_p=ORDER[name], stop_when_converged=False),
                         name="v005_mesh", label=name)
        exact = float(reference(system, settings, springs=SPRINGS, settlements_mm=SETTLEMENTS_MM)["XY"].v(LOAD_P[0]))
        conv += [{**rec, "theory": name, "r01_v_P_xy_mm": exact} for rec in res.records()]
        checks.append(richardson_verdict(name, res.final["v_P_xy_mm"], exact, case="v005", plane="XY",
                                         quantity="v at P extrapolated", n_elements=N_ELEMENTS[-1],
                                         h_mm=LENGTH_MM / N_ELEMENTS[-1], single_point=name != "euler_bernoulli",
                                         flat_rtol=FLAT_RTOL, sweep="mesh"))
    return checks, conv, coarse


def stiffness_sweep(n_elements: int = 27) -> tuple[list[dict], list[dict]]:
    """Middle support a radial spring k = ratio x K_MID, ends rigid, P only, EB: reactions vs k (C1-C3 per k).

    The limits: k -> 0, the two-pin shaft (statics: R_1 = P (x_3 - x_P)/(x_3 - x_1), R_2 = 0); k -> inf,
    Harris Eqs 9.42-9.44. Their deviations at the ends of the sweep are RECORD rows K0, Kinf (r01 judges every
    k; the limits are r01's C21 too). Returns (check records, curve records for ``ratio_sweep``).
    """
    ss = literature.harris_shaft()
    sp = literature.spans()
    P = sp["P"]
    order = np.argsort([b.position for b in ss.bearings])
    x1, x2, x3 = SUPPORTS_MM
    two_pin = (P * (x3 - LOAD_P[0]) / (x3 - x1), 0.0, P * (LOAD_P[0] - x1) / (x3 - x1))
    harris = literature.harris_eqs_9_42_9_44(**sp)
    segs = segments(ss, EB)
    checks, curve = [], []
    for ratio in K_RATIOS:
        springs = {x2: (ratio * K_MID_N_PER_MM, 0.0)}
        r = solve(ss, EB, n_elements, springs=springs)
        rows = check(ss, r, EB, springs=springs)
        checks += [{**c, "sweep": "k_mid", "k_ratio": ratio} for c in rows]
        F = np.asarray(r.R_xy)[order]
        err = next(c["rel_err"] for c in rows if c["id"] == "C3")
        curve += [{"k_ratio": ratio, "y": F[i] / P, "series": f"R_{i + 1}", "source": "FEM", "rel_err": err}
                  for i in range(3)]
    # the r01 curves on a dense k grid, and the two limits drawn at the ends of the sweep
    for ratio in np.logspace(-4.0, 4.0, 81):
        sol = solve_supported([(LOAD_P[0], -P)], [], [Support(x1), Support(x2, k_r=ratio * K_MID_N_PER_MM), Support(x3)],
                              segs, LENGTH_MM)
        curve += [{"k_ratio": float(ratio), "y": sol.R[i] / P, "series": f"R_{i + 1}", "source": "r01"} for i in range(3)]
    lo, hi = K_RATIOS[:2], K_RATIOS[-2:]
    for i in range(3):
        curve += [{"k_ratio": k, "y": two_pin[i] / P, "series": f"R_{i + 1}", "source": "two pins (k -> 0)"} for k in lo]
        curve += [{"k_ratio": k, "y": harris[i] / P, "series": f"R_{i + 1}", "source": "Harris 9.42-9.44 (k -> inf)"}
                  for k in hi]
    F_lo = [c["y"] * P for c in curve if c["source"] == "FEM" and c["k_ratio"] == K_RATIOS[0]]
    F_hi = [c["y"] * P for c in curve if c["source"] == "FEM" and c["k_ratio"] == K_RATIOS[-1]]
    checks.append(_rec("K0", f"R at k = {K_RATIOS[0]:g} K_MID vs two pins", "N", F_lo, two_pin, P, None,
                       source="statics", sweep="k_mid", k_ratio=K_RATIOS[0]))
    checks.append(_rec("Kinf", f"R at k = {K_RATIOS[-1]:g} K_MID vs Harris 9.42-9.44", "N", F_hi, harris, P, None,
                       source="Harris2007_vol2 Eqs 9.42-9.44, PDF p. 285", sweep="k_mid", k_ratio=K_RATIOS[-1]))
    return checks, curve


def hertz_flexible(n_elements: int = 54) -> tuple[list[dict], dict]:
    """Part 3: the flexible shaft (S355) on three Hertz bearings, P only, EB; Newton and its four checks.

    * N1 Newton converged (RECORD: iterations, reason);
    * N2 Newton vs ``scipy.optimize.root`` (``hybr``, same residual and Jacobian, from the same start): bearing
      forces within ``TOL_SELF`` of P;
    * N3 the converged shaft vs r01 with the SECANT springs ``k_i = K |v_i|^(n-1)`` (the linear law through
      the converged point): C1-C3 at ``TOL_NODAL`` (the physics reviewer's condition: never the nonlinear law
      in r01);
    * N4 fed back: pins at 10 and 260 settled to the converged v_1, v_3 and the middle bearing replaced by a
      ``RadialLoad`` of its converged force on the shaft (+F, F0b): v at 110 and R_1, R_3 reproduce the
      converged state within ``TOL_NODAL`` (``convention-dependent (R-003)``);
    * N5 equilibrium: sum F = P, sum F (x - x_1) = P a: ``TOL_NODAL`` of P, P l.
    """
    ss = literature.harris_shaft("v005_hertz")
    result, res, prob = solve_hertz(ss, EB, n_elements, label="flexible, EI x 1")
    sci = root(prob["residual"], prob["x0"], jac=prob["jacobian"], method="hybr", options={"xtol": 1e-14})
    v = res.x
    F = prob["force"](v)
    order = np.argsort(prob["positions"])
    xs = np.asarray(prob["positions"])[order]
    F, v = F[order], v[order]
    F_sci = prob["force"](sci.x)[order]
    P = LOAD_P[1]
    rows = [_rec("N1", "Newton converged; iterations", "-", res.n_iter, res.n_iter, 1.0, None, source=res.reason,
                 converged=res.converged, residual_norm=res.residual_norm),
            _rec("N2", "bearing forces, Newton vs scipy root (hybr)", "N", F, F_sci, P, TOL_SELF,
                 source=f"scipy root success={sci.success} nfev={sci.nfev}")]
    secant = {float(x): (float(K_HERTZ * abs(vi) ** (N_HERTZ - 1.0)), 0.0) for x, vi in zip(xs, v)}
    rows += [{**c, "id": f"N3/{c['id']}"} for c in check(ss, result, EB, springs=secant, label="secant springs")]
    # fed back (R-003): pins 1, 3 settled to the converged deflections, bearing 2 a RadialLoad of +F_2 (along +Y)
    fed = three_support_shaft(supports_mm=(xs[0], xs[2]),
                              loads=[RadialLoad(LOAD_P[0], P, theta_deg=LOAD_P[2], label="P"),
                                     RadialLoad(xs[1], abs(F[1]), theta_deg=0.0 if F[1] >= 0.0 else 180.0,
                                                label="F_2 fed back", source="bearing_reaction")],
                              label="v005_fed_back")
    r_fed = solve(fed, EB, n_elements, settlements_mm={"XY": {xs[0]: v[0], xs[2]: v[2]}})
    rows.append(_rec("N4", "v at bearing 2, F_2 fed back as RadialLoad", "mm", node_value(r_fed, xs[1], "v_xy"), v[1],
                     float(np.max(np.abs(v))), TOL_NODAL, tag=R003, source="AxisForge, RadialLoad"))
    rows.append(_rec("N4", "R_1, R_3 with F_2 fed back", "N", np.asarray(r_fed.R_xy)[np.argsort([b.position for b in fed.bearings])],
                     [F[0], F[2]], P, TOL_NODAL, tag=R003, source="AxisForge, RadialLoad"))
    rows.append(_rec("N5", "sum F", "N", float(F.sum()), P, P, TOL_NODAL, source="statics"))
    rows.append(_rec("N5", "sum F (x - x_1)", "N.mm", float(F @ (xs - xs[0])), P * (LOAD_P[0] - xs[0]),
                     P * (xs[-1] - xs[0]), TOL_NODAL, source="statics"))
    rows.append(_rec("N6", "flexible shaft vs Harris 9.48 (rigid shaft)", "N", F,
                     literature.harris_eq_9_48(**literature.spans()), P, None,
                     source="Harris2007_vol2 Eq. 9.48, PDF p. 286", ei_scale=1.0))
    return rows, {"result": result, "newton": res, "system": ss, "secant": secant, "forces_N": F}


def rigid_limit(n_elements: int = 54) -> tuple[list[dict], list[dict], list[dict]]:
    """Part 2: EI x scale on three Hertz bearings vs Harris Eq. 9.48 (scale -> inf), EB, P only.

    H1 at ``SCALE_JUDGED``: forces within ``TOL_RIGID`` of P. H2: the error falls as 1/scale (first order in the
    shaft compliance): ``err(1e5) x 1e5 / (err(1e3) x 1e3)`` within ``H2_BAND``. Both were set after a prototype
    of this series, so in this run they are RECORD (``CRITERIA_NEXT_RUN``; ``verdict_next_run`` holds what the
    criterion would say). Every other scale: RECORD (H0), with its Newton summary. Returns (records, Newton
    history records, sweep records for the figures).
    """
    sp = literature.spans()
    P = sp["P"]
    harris = np.array(literature.harris_eq_9_48(**sp))
    rows, hist, sweep = [], [], []
    errs = {}
    for scale in EI_SCALES:
        ss = three_support_shaft(loads=[RadialLoad(LOAD_P[0], P, theta_deg=LOAD_P[2], label="P")],
                                 material_id=stiff_material(scale), label=f"v005_ei_x{scale:g}")
        _, res, prob = solve_hertz(ss, EB, n_elements, label=f"EI x {scale:g}")
        F = prob["force"](res.x)[np.argsort(prob["positions"])]
        errs[scale] = float(np.max(np.abs(F - harris))) / P
        judged = scale == SCALE_JUDGED
        rows.append(_rec("H1" if judged else "H0", f"bearing forces at EI x {scale:g} vs Harris 9.48", "N", F, harris,
                         P, TOL_RIGID if judged else None, source="Harris2007_vol2 Eq. 9.48, PDF p. 286",
                         ei_scale=scale, newton_converged=res.converged, newton_reason=res.reason, newton_iter=res.n_iter,
                         residual_norm=res.residual_norm, newton_residual_abs=NEWTON["residual_abs"]))
        hist += [{**h, "ei_scale": scale} for h in res.records()]
        sweep += [{"ei_scale": scale, "y": F[i] / P, "series": f"F_{i + 1}", "source": "FEM"} for i in range(3)]
        sweep += [{"ei_scale": scale, "y": harris[i] / P, "series": f"F_{i + 1}", "source": "Harris 9.48"}
                  for i in range(3)]
    ratio = errs[1e5] * 1e5 / (errs[1e3] * 1e3)
    rows.append({"case": "v005", "id": "H2", "quantity": "err x scale at 1e5 over that at 1e3 (O(1/scale))",
                 "unit": "-", "value": ratio, "judge": 1.0, "abs_diff": ratio - 1.0, "ref_scale": 1.0,
                 "rel_err": ratio - 1.0, "tol": H2_BAND[1] - 1.0,
                 "verdict": "PASS" if H2_BAND[0] <= ratio <= H2_BAND[1] else "MISS",
                 "source": "first-order perturbation in the shaft compliance", "tag": ""})
    for r in rows:                                  # set after a prototype: RECORD now, judged from the next run
        if r["id"] in CRITERIA_NEXT_RUN:
            r["verdict_next_run"], r["verdict"] = r["verdict"], "RECORD"
    return rows, hist, sweep


def run(results=RESULTS) -> dict:
    """Every part, the CSV tables and the figures; returns the records and the figure paths."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    first = first_check()
    lit = literature_series()
    checks, conv, coarse = mesh_series()
    k_checks, k_curve = stiffness_sweep()
    hz, flex = hertz_flexible()
    rig, hist, rig_sweep = rigid_limit()
    flex_hist = [{**h, "ei_scale": 1.0} for h in flex["newton"].records()]
    write_records_csv(results / "v005_first_check.csv", first)
    write_records_csv(results / "v005_literature.csv", lit)
    write_records_csv(results / "v005_check.csv", checks + k_checks)
    write_records_csv(results / "v005_richardson.csv", conv)
    write_records_csv(results / "v005_stiffness_sweep.csv", k_curve)
    write_records_csv(results / "v005_hertz.csv", hz + rig)
    write_records_csv(results / "v005_newton_history.csv", flex_hist + hist)

    figures = []
    system = three_support_shaft()
    for plane in ("XY", "XZ"):
        recs = []
        for name, r in coarse.items():
            recs += along_shaft(system, r, MODELS[name], springs=SPRINGS, settlements_mm=SETTLEMENTS_MM, plane=plane)
        figures.append(plots.deflection_along_shaft(
            recs, results, "v005", supports_mm=SUPPORTS_MM, loads_mm=(LOAD_P[0] if plane == "XY" else LOAD_Q[0],),
            variant=f"{plane}_n{N_ELEMENTS[0]}",
            caption=f"{plane}: spring {K_MID_N_PER_MM:g} N/mm at 110, k_theta 2e8 N.mm/rad at 260, settlements "
                    f"{SETTLEMENTS_MM[plane]}; {CAPTION}"))
    err = {}
    for c in checks:
        if c["quantity"] == "v" and c["id"] in ("C1", "C4", "T1"):
            key = (f"{c['theory']} vs {c['reference']}", c["n_elements"])
            err[key] = max(err.get(key, 0.0), abs(c["rel_err"]))
    figures.append(plots.error_vs_elements([{"series": s, "n_elements": n, "rel_err": e} for (s, n), e in err.items()],
                                           results, "v005", tol=TOL_NODAL,
                                           order_guides={"timoshenko/cowper/single_point vs r01": 2.0},
                                           caption=f"largest nodal v error over both planes; {CAPTION}"))
    figures.append(plots.ratio_sweep(k_curve, results, "v005", x="k_ratio", x_label=r"$k / k_\mathrm{mid}$ (-)",
                                     y_label=r"$R_i / P$ (-)", variant="stiffness",
                                     caption=f"P only, ends rigid, middle spring k, k_mid {K_MID_N_PER_MM:g} N/mm; "
                                             f"FEM n 27, EB; residual: FEM vs r01 (C3)"))
    recs = along_shaft(flex["system"], flex["result"], EB, springs=flex["secant"], series="EB, Hertz bearings")
    figures.append(plots.deflection_along_shaft(
        recs, results, "v005", supports_mm=SUPPORTS_MM, loads_mm=(LOAD_P[0],), variant="hertz",
        caption=f"Hertz bearings K {K_HERTZ:g} N/mm^1.5, n 1.5: FEM (Newton) vs r01 at the secant springs; "
                f"F = {', '.join(f'{f:.1f}' for f in flex['forces_N'])} N; {R003}"))
    figures.append(plots.ratio_sweep(rig_sweep, results, "v005", x="ei_scale", x_label=r"EI scale $s$ (-)",
                                     y_label=r"$F_i / P$ (-)", variant="rigid_limit",
                                     caption=f"three Hertz bearings, K {K_HERTZ:g} N/mm^1.5; lines: Harris Eq. 9.48 "
                                             f"(rigid shaft)"))
    # the H1 tolerance is one point (s = 1e5): drawn as its own marker, with the 1/s guide through it (H2)
    series = "EI x s vs Harris 9.48"
    err_recs = [{"ei_scale": r["ei_scale"], "quantity": "bearing forces", "series": series, "rel_err": r["rel_err"]}
                for r in rig if "ei_scale" in r]
    err_recs += [{"ei_scale": s, "quantity": "1/s through H1 tolerance", "series": series,
                  "rel_err": TOL_RIGID * SCALE_JUDGED / s} for s in EI_SCALES if s >= 1e2]   # s < 1e2: not asymptotic
    err_recs.append({"ei_scale": SCALE_JUDGED, "quantity": "H1 tolerance (next run)", "series": series,
                     "rel_err": TOL_RIGID})                                                 # drawn on top of the guide
    figures.append(relative_errors(err_recs, results, "v005", x="ei_scale", x_label=r"EI scale $s$ (-)",
                                   variant="rigid_limit",
                                   caption=f"max |F - F_Harris| / P; H1 tolerance {TOL_RIGID:g} at s = {SCALE_JUDGED:g} "
                                           f"and H2 (O(1/s)): RECORD in this run, fixed for the next"))
    figures.append(newton_history(flex_hist + hist, results, "v005", tol=NEWTON["residual_abs"],
                                  caption="numerical_solvers.newton, analytic Jacobian, line search; residual scaled by P"))
    every = first + lit + checks + k_checks + hz + rig
    return {"first": first, "literature": lit, "checks": checks, "k_checks": k_checks, "hertz": hz, "rigid": rig,
            "richardson": conv, "history": flex_hist + hist, "figures": figures,
            "verdicts": {v: sum(r.get("verdict") == v for r in every) for v in ("PASS", "MISS", "RECORD")}}


if __name__ == "__main__":
    out = run()
    print("v005:", out["verdicts"])
    for r in out["first"] + out["literature"] + out["hertz"] + out["rigid"]:
        print(f"{r['id']:>8} {r['verdict']:>6} {r['rel_err']:.3e} {r['quantity']}")
    worst = {}
    for r in out["checks"] + out["k_checks"]:
        if r["verdict"] in ("PASS", "MISS"):
            worst[r["id"]] = max(worst.get(r["id"], 0.0), abs(r["rel_err"]))
    print("worst judged:", {k: f"{v:.2e}" for k, v in worst.items()})
