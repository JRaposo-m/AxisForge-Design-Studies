"""Run of v002: the stepped gear shafts through the AxisForge FEM, against r01, the gear identities and Harris.

``python -m studies.v000_axisforge.v002_shaft_stepped_gear.study`` writes into ``results/`` (git-ignored):
``v002_check.csv`` (FEM vs r01 records of every solve), ``v002_identities.csv`` (mesh loads vs the gear
inputs and the pitch-point statics), ``v002_richardson.csv`` (mesh series of the gear-point deflection),
``v002_couple.csv`` (reaction shift of the couple vs helix angle and hand), ``v002_literature.csv``
(Harris2007_vol2 Eqs 9.24-9.27) and the figures of ``studies/outputs``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import build_shaft_result
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (RigidSupportFEMSolver,
                                                                                               RigidSupportSolution)

from numerical_solvers import RichardsonOptions, richardson
from studies.outputs.l000_axisforge.plots import shaft as plots
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.v000_axisforge.v001_shaft_uniform.check import node_value, richardson_verdict

from . import literature
from .check import J16, TOL_NODAL, along_shaft, check, gear_identities, mesh_judge, physical_couples, reference, rigid_body
from .construction import (ALPHA_N_DEG, BETA_DEG, FACE_WIDTH_MM, GEAR_POSITION_MM, HAND, LENGTH_MM, MODULE_MM, N_ELEMENTS, PHI_DEG,
                           ROTATION, STEPS_MM, SUPPORTS_MM, THEORIES, TORQUE_NM, Z, gear_system, uniform_nodes)

RESULTS = Path(__file__).resolve().parent / "results"
# base settlements of the two pins (A at 10 mm, B at 190 mm) along +v [mm]: of the order of the elastic
# deflection (about 0.03 mm here), different in each plane so that a swapped plane would show
SETTLEMENTS_MM = {"XY": (0.02, -0.03), "XZ": (-0.01, 0.04)}
SETTLEMENT_SWEEP_MM = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)   # delta_B; delta_A = -delta_B / 2, both planes
BETA_SWEEP_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
PHI_SWEEP_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 270.0)
# formal order of the nodal error, the max_p of Richardson (R3-17): Hermite cubic O(h^4) (nodally exact,
# flat); 2-node linear Timoshenko with one-point shear integration O(h^2) (v001 kga_reduced_integration)
ORDER = {"euler_bernoulli": 4.0, "timoshenko/cowper/single_point": 2.0, "timoshenko/hutchinson/single_point": 2.0}
MODELS = {k: THEORIES[k] for k in ORDER}
SPAN_MM = SUPPORTS_MM[1] - SUPPORTS_MM[0]
FLAT_RTOL = 1e-8                # [-] Richardson "flat": level differences <= 1e-8 |f| (numerical_solvers default), C5
CAPTION = (f"stepped shaft L 200 mm, d 20/28/25/28/20 mm, steps at 20/85/115/180 mm, pins at 10/190 mm; "
           f"pair mn {MODULE_MM:g} mm, z {Z[0]}/{Z[1]}, T 100 N.m at the pinion; couple F_a r_w: {J16}")


def solve(system, settings, n_elements: int | None = None, *, settlements_mm: dict | None = None,
          kGA_override: float | None = None):
    """``RigidSupportFEMSolver`` on a uniform mesh, then the prescribed pin settlements imposed on its own K.

    AxisForge's rigid supports are v = 0 (``boundary_dofs``) and have no settlement input
    (``# OPEN R5-31``). With settlements, the solve of AxisForge is redone on its own global
    stiffness K, load vector f_ext and DOF partition, the constrained transverse DOF of each pin
    prescribed: ``d_c = delta``, ``K_ff d_f = f_f - K_fc d_c``, reactions ``K d - f_ext`` (elimination
    with prescribed displacements, the method of the solver itself: Onate2013_vol2 1.3.4, Eq. (1.22)).
    The settled solution then goes through AxisForge's ``build_shaft_result`` (moments, shear forces,
    reactions per bearing), so every output is AxisForge's post-processing.

    Parameters
    ----------
    system : axisforge ShaftSystem
        The loaded shaft (two pins).
    settings : axisforge BeamModelSettings
        Beam model.
    n_elements : int, optional
        Uniform divisions of the whole shaft (``uniform_nodes``); None: the mandatory nodes only.
    settlements_mm : dict, optional
        ``{"XY": (delta_A, delta_B), "XZ": (...)}`` [mm] along +Y / +Z, for the bearings in
        position order; None: rigid pins (AxisForge's result returned as it is).
    kGA_override : float, optional
        Passed to the solver [N].

    Returns
    -------
    ShaftResults
    """
    extra = uniform_nodes(n_elements, system.shaft.total_length) if n_elements else None
    r = RigidSupportFEMSolver(settings, kGA_override=kGA_override).solve(system, extra_mandatory=extra)
    if not settlements_mm:
        return r
    K, free, fixed = r.K, list(r.free_dofs), list(r.constrained_dofs)
    x_nodes = np.asarray(r.x_nodes, dtype=float)
    d, f_tot, f_reac = {}, {}, {}
    for plane in ("xz", "xy"):
        f_ext = np.asarray(getattr(r, f"f_{plane}_ext"), dtype=float)
        dp = np.zeros_like(f_ext)
        # (delta_A, delta_B) belong to the pins BY POSITION, as in v001 check (physics review)
        pins = sorted(system.bearings, key=lambda b: b.position)
        for bearing, delta in zip(pins, settlements_mm.get(plane.upper(), (0.0, 0.0))):
            node = int(np.argmin(np.abs(x_nodes - bearing.position)))
            if abs(x_nodes[node] - bearing.position) > 1e-9 * max(1.0, bearing.position):
                raise ValueError(f"pin at {bearing.position} mm is not a mesh node")
            dp[3 * node + 1] = delta                    # DOF order [u, v, theta] per node: v of the pin node
        dp[free] = np.linalg.solve(K[np.ix_(free, free)], f_ext[free] - K[np.ix_(free, fixed)] @ dp[fixed])
        d[plane], f_tot[plane] = dp, K @ dp
        f_reac[plane] = f_tot[plane] - f_ext
    sol = RigidSupportSolution(x_nodes=list(r.x_nodes), elements=r.elements, K=K, free_dofs=free, constrained_dofs=fixed,
                               d_total_xz=d["xz"], d_total_xy=d["xy"], f_xz_ext=r.f_xz_ext, f_xy_ext=r.f_xy_ext,
                               f_xz_total=f_tot["xz"], f_xy_total=f_tot["xy"], f_xz_reaction=f_reac["xz"],
                               f_xy_reaction=f_reac["xy"], _kGA_override=kGA_override)
    return build_shaft_result(sol, system)


def _tagged(rows: list[dict], **extra) -> list[dict]:
    """The records with the inputs of their solve added (the swept parameters of the case table)."""
    return [{**r, **extra} for r in rows]


def mesh_series(resolved) -> tuple[list[dict], list[dict], dict]:
    """Every beam model on every mesh, both shafts, base settlements: check records, Richardson records, coarse solves.

    Richardson (``numerical_solvers``, kind "count", ``flat_atol=0``: the deflection is ~1e-2 mm,
    R3-10) on v_xy of shaft1 at the gear node; ``max_p`` = the formal order (R3-17); every level.
    """
    checks, conv, coarse = [], [], {}
    shaft1 = resolved.shafts[0]
    for ss in resolved.shafts:
        for name, settings in MODELS.items():
            def evaluate(n, settings=settings, name=name, ss=ss):
                r = solve(ss, settings, int(n), settlements_mm=SETTLEMENTS_MM)
                checks.extend(_tagged(check(ss, r, settings, settlements_mm=SETTLEMENTS_MM), sweep="mesh",
                                      beta_deg=BETA_DEG, hand=HAND, phi_deg=PHI_DEG, distribute=False))
                if int(n) == N_ELEMENTS[0]:
                    coarse[(ss.name, name)] = r
                return {"v_gear_xy_mm": node_value(r, GEAR_POSITION_MM, "v_xy")}
            if ss is not shaft1:
                for n in N_ELEMENTS:
                    evaluate(n)
                continue
            res = richardson(evaluate, N_ELEMENTS,
                             options=RichardsonOptions(kind="count", flat_atol=0.0, flat_rtol=FLAT_RTOL,
                                                       max_p=ORDER[name], stop_when_converged=False),
                             name="v002_mesh", label=name)
            sol = reference(ss, settings)["XY"]
            v_rb, _ = rigid_body(GEAR_POSITION_MM, *SUPPORTS_MM, *SETTLEMENTS_MM["XY"])
            exact = float(sol.v(GEAR_POSITION_MM)) + float(v_rb)
            for rec in res.records():
                conv.append({**rec, "theory": name, "r01_v_gear_xy_mm": exact, "tag": J16})
            checks.append(richardson_verdict(name, res.final["v_gear_xy_mm"], exact, case="v002", plane="XY",
                                             quantity="v_gear extrapolated", n_elements=N_ELEMENTS[-1],
                                             h_mm=LENGTH_MM / N_ELEMENTS[-1], single_point=name != "euler_bernoulli",
                                             flat_rtol=FLAT_RTOL, tag=J16, sweep="mesh"))
    return checks, conv, coarse


def gear_sweeps() -> tuple[list[dict], list[dict], list[dict]]:
    """Helix angle x hand (phi = 0, couple in XY) and line-of-centres angle (beta 20 deg): checks, identities, couple shift.

    Euler-Bernoulli and one-point Timoshenko (Cowper) on 40 elements, rigid pins, both shafts. The
    couple record per shaft: ``dR_A`` = FEM reaction at pin A in XY minus r01's without the couple
    [N], against the statics of the couple alone, ``C_XY / span`` (a couple C on a span s moves
    +C/s to pin A and -C/s to pin B), with ``C_XY = physical_couples`` of the judge built from the
    inputs (``mesh_judge``: F_x and rho with their signs), not of AxisForge's ``MeshForces``.
    """
    checks, ids, couple = [], [], []
    runs = [(b, h, 0.0) for b in BETA_SWEEP_DEG for h in ((1, -1) if b else (1,))]
    runs += [(BETA_DEG, HAND, p) for p in PHI_SWEEP_DEG if p != 0.0]
    for beta, hand, phi in runs:
        sweep = "phi" if phi != 0.0 else "beta"
        resolved = gear_system(beta_deg=beta, hand=hand, phi_deg=phi)
        judge = _judge(beta, hand)
        ids += _tagged(gear_identities(resolved, judge, theta_c_deg=phi), sweep=sweep, beta_deg=beta, hand=hand,
                       phi_deg=phi)
        for side, ss in enumerate(resolved.shafts, start=1):
            solved = {}
            for name in ("euler_bernoulli", "timoshenko/cowper/single_point"):
                solved[name] = solve(ss, MODELS[name], N_ELEMENTS[0])
                checks += _tagged(check(ss, solved[name], MODELS[name]), sweep=sweep, beta_deg=beta, hand=hand,
                                  phi_deg=phi, distribute=False)
            if sweep != "beta":
                continue
            r = solved["euler_bernoulli"]
            rho = judge["rho_mm"][side - 1]
            F_x = (judge["on_1"] if side == 1 else judge["on_2"])[2]
            C_xy = physical_couples(F_x, rho, phi)[0]                # r x F of the axial force, plane XY
            no_couple = reference(ss, MODELS["euler_bernoulli"], couples=False)["XY"]
            i_A = [b.position for b in ss.bearings].index(SUPPORTS_MM[0])
            dR = float(r.R_xy[i_A]) - no_couple.R_A
            norm = judge["F_tw_N"] * abs(rho) / SPAN_MM              # F_tw r_w / span [N]
            couple.append({"case": "v002", "shaft": ss.name, "beta_deg": beta, "hand": hand, "phi_deg": phi,
                           "F_tw_N": judge["F_tw_N"], "F_a_N": judge["F_a_N"], "r_w_mm": abs(rho), "C_xy_Nmm": C_xy,
                           "R_A_xy_fem_N": float(r.R_xy[i_A]), "R_A_xy_r01_no_couple_N": no_couple.R_A,
                           "dR_A_fem_N": dR, "dR_A_statics_N": C_xy / SPAN_MM,
                           "y_fem": dR / norm, "y_statics": C_xy / SPAN_MM / norm,
                           "rel_err": (dR - C_xy / SPAN_MM) / (judge["F_a_N"] * abs(rho) / SPAN_MM) if beta else math.nan,
                           "tag": J16})
    return checks, ids, couple


def _judge(beta: float, hand: int) -> dict:
    """``mesh_judge`` of the pair of construction (pinion = gear 1 drives, sense ROTATION, torque TORQUE_NM)."""
    return mesh_judge(torque_Nm=TORQUE_NM, beta_deg=beta, mn_mm=MODULE_MM, z1=Z[0], z2=Z[1], alpha_n_deg=ALPHA_N_DEG,
                      b_mm=FACE_WIDTH_MM, hand1=hand if beta else 0, rot1=ROTATION, gear1_drives=True)


def settlement_sweep() -> list[dict]:
    """Settlement delta_B = 0 ... 0.5 mm (delta_A = -delta_B/2), both planes, base gear: EB and Timoshenko, 40 elements."""
    resolved = gear_system()
    out = []
    for dB in SETTLEMENT_SWEEP_MM:
        settle = {p: (-0.5 * dB, dB) for p in ("XY", "XZ")}
        for ss in resolved.shafts:
            for name in ("euler_bernoulli", "timoshenko/cowper/single_point"):
                settings = MODELS[name]
                out += _tagged(check(ss, solve(ss, settings, N_ELEMENTS[0], settlements_mm=settle), settings,
                                     settlements_mm=settle),
                               sweep="settlement", beta_deg=BETA_DEG, hand=HAND, phi_deg=PHI_DEG, distribute=False)
    return out


def distributed_variant() -> list[dict]:
    """The base gear with F_c, F_t uniform over the face width (``distribute_loads=True``), base settlements, 40 and 80 elements."""
    resolved = gear_system(distribute=True)
    out = []
    for ss in resolved.shafts:
        for name, settings in MODELS.items():
            for n in N_ELEMENTS[:2]:
                r = solve(ss, settings, n, settlements_mm=SETTLEMENTS_MM)
                out += _tagged(check(ss, r, settings, settlements_mm=SETTLEMENTS_MM), sweep="distributed",
                               beta_deg=BETA_DEG, hand=HAND, phi_deg=PHI_DEG, distribute=True)
    return out


def literature_series() -> list[dict]:
    """Harris2007_vol2 Eqs 9.24-9.27 (PDF p. 281) on the uniform shaft of the literature, Euler-Bernoulli, 40 elements."""
    (case,) = literature.cases()
    system = case.build()
    settings = MODELS["euler_bernoulli"]
    r = solve(system, settings, literature.N_ELEMENTS, settlements_mm={"XY": literature.SETTLEMENTS_MM})
    fem = {"theta_1": node_value(r, 0.0, "theta_xy"), "theta_2": node_value(r, literature.SPAN_MM, "theta_xy"),
           "R_A": float(r.R_xy[0]), "R_B": float(r.R_xy[1])}
    rows = literature.records(fem, system)
    rows += _tagged(check(system, r, settings, settlements_mm={"XY": literature.SETTLEMENTS_MM}, label="harris"),
                    sweep="literature")
    return rows


def run(results=RESULTS) -> dict:
    """Mesh series, sweeps and literature; write the CSV tables and the figures; return the records."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    base = gear_system()
    checks, conv, coarse = mesh_series(base)
    ids = _tagged(gear_identities(base, _judge(BETA_DEG, HAND), theta_c_deg=PHI_DEG), sweep="base", beta_deg=BETA_DEG,
                  hand=HAND, phi_deg=PHI_DEG)
    g_checks, g_ids, couple = gear_sweeps()
    s_checks = settlement_sweep()
    d_checks = distributed_variant()
    lit = literature_series()
    all_checks = checks + g_checks + s_checks + d_checks
    write_records_csv(results / "v002_check.csv", all_checks)
    write_records_csv(results / "v002_identities.csv", ids + g_ids)
    write_records_csv(results / "v002_richardson.csv", conv)
    write_records_csv(results / "v002_couple.csv", couple)
    write_records_csv(results / "v002_literature.csv", lit)

    n0 = N_ELEMENTS[0]
    figures = []
    shaft1 = base.shafts[0]
    fields = (along_shaft(shaft1, coarse[("shaft1", "euler_bernoulli")], MODELS["euler_bernoulli"],
                          settlements_mm=SETTLEMENTS_MM, series="Euler-Bernoulli")
              + along_shaft(shaft1, coarse[("shaft1", "timoshenko/cowper/single_point")], MODELS["timoshenko/cowper/single_point"],
                            settlements_mm=SETTLEMENTS_MM, series="Timoshenko (Cowper, one-point)"))
    for plane in ("XY", "XZ"):
        dA, dB = SETTLEMENTS_MM[plane]
        figures.append(plots.deflection_along_shaft(
            [r for r in fields if r["plane"] == plane], results, "v002", supports_mm=SUPPORTS_MM,
            loads_mm=(GEAR_POSITION_MM,), steps_mm=STEPS_MM, variant=f"{plane}_n{n0}",
            caption=f"shaft1 (pinion), plane {plane}, n = {n0}; beta {BETA_DEG:g} deg, hand {HAND:+d}, phi {PHI_DEG:g} "
                    f"deg; settlements A {dA:g} mm, B {dB:g} mm\n{CAPTION}"))
    # error of the deflection field, worst plane and shaft, per model and mesh
    label = {("C1", "euler_bernoulli"): "Euler-Bernoulli vs r01 (C1)",
             ("T1", "timoshenko/cowper/single_point"): "Timoshenko one-point vs r01 (T1)",
             ("C4", "timoshenko/cowper/single_point"): "Timoshenko one-point vs r01 at kGA_eff (C4)"}
    err = {}
    for r in checks:
        key = label.get((r.get("id"), r.get("theory")))
        if key and r["quantity"] == "v":
            err[(key, r["n_elements"])] = max(err.get((key, r["n_elements"]), 0.0), r["rel_err"])
    figures.append(plots.error_vs_elements([{"series": s, "n_elements": n, "rel_err": e} for (s, n), e in err.items()],
                                           results, "v002", tol=TOL_NODAL,
                                           order_guides={"Timoshenko one-point vs r01 (T1)": 2},
                                           caption=f"max nodal |v - v_r01| / max |v_r01|, both planes and shafts, "
                                                   f"base settlements\n{CAPTION}"))
    figures.append(relative_errors(_worst(s_checks, "delta_B_mm"), results, "v002",
                                   x="delta_B_mm", x_label=r"settlement $\delta_B = -2\,\delta_A$ (mm)",
                                   log_x=False, variant="settlement",
                                   caption=f"n = {n0}; worst of both planes and shafts; beta {BETA_DEG:g} deg, phi "
                                           f"{PHI_DEG:g} deg; delta = 0 is the rigid-pin limit; errors below 1e-10 % "
                                           f"drawn at 1e-10 %\n{CAPTION}"))
    figures.append(relative_errors(_worst([r for r in g_checks if r["sweep"] == "beta"], "beta_deg", by_hand=True),
                                   results, "v002", x="beta_deg", x_label=r"helix angle $\beta$ (deg)", log_x=False,
                                   variant="helix",
                                   caption=f"n = {n0}, rigid pins, phi 0 (couple in XY); beta 0 = spur, no couple; "
                                           f"errors below 1e-10 % drawn at 1e-10 %\n{CAPTION}"))
    curve = [{"beta_deg": float(b), "y": y, "series": f"{s}, hand {h:+d}", "source": "statics C/span"}
             for (b, h, s, y) in _couple_curve()]
    fem_pts = [{"beta_deg": c["beta_deg"], "y": c["y_fem"], "series": f"{c['shaft']}, hand {c['hand']:+d}",
                "source": "FEM", **({"rel_err": c["y_fem"] / c["y_statics"] - 1.0} if c["beta_deg"] else {})}
               for c in couple]
    # the spur point belongs to both hands
    fem_pts += [{**p, "series": p["series"].replace("+1", "-1")} for p in fem_pts if p["beta_deg"] == 0.0]
    figures.append(plots.ratio_sweep(fem_pts + curve, results, "v002", x="beta_deg",
                                     x_label=r"helix angle $\beta$ (deg)",
                                     y_label=r"$\Delta R_A\,s/(F_{tw} r_w)$ (-)", log_x=False, variant="couple_shift",
                                     caption=f"reaction at pin A, plane XY, FEM minus r01 without the couple, over "
                                             f"F_tw r_w / span (s = {SPAN_MM:g} mm); line: statics of the couple "
                                             f"F_a r_w = F_tw r_w tan(beta); phi 0, n = {n0}, Euler-Bernoulli\n"
                                             f"{J16}: couple as ExternalMoment consumed by the FEM"))
    return {"check": all_checks, "identities": ids + g_ids, "richardson": conv, "couple": couple,
            "literature": lit, "figures": figures}


def _worst(rows: list[dict], x: str, *, by_hand: bool = False) -> list[dict]:
    """Largest judged error per (series, quantity, x) over planes and shafts, for ``relative_errors``.

    ``series`` (one panel): the beam model (and the hand); ``quantity``: the criterion id and field
    (C1 v, C2 theta, C3 R, C4 v / theta); ``tol`` and ``verdict`` of the worst record.
    """
    worst = {}
    for r in rows:
        if r["verdict"] == "RECORD":
            continue
        model = "Euler-Bernoulli" if r["theory"] == "euler_bernoulli" else "Timoshenko one-point"
        key = (model + (f", hand {r['hand']:+d}" if by_hand else ""), f"{r['id']} {r['quantity']}", r[x])
        if key not in worst or r["rel_err"] > worst[key]["rel_err"]:
            worst[key] = {"series": key[0], "quantity": key[1], x: key[2], "rel_err": r["rel_err"], "tol": r["tol"],
                          "verdict": r["verdict"]}
    return list(worst.values())


def _couple_curve(n: int = 31):
    """Statics of the couple on a dense helix sweep from the inputs alone (``mesh_judge``, no AxisForge):
    ``(beta, hand, shaft, C_XY / span / (F_tw r_w / span))``."""
    for beta in np.linspace(0.0, BETA_SWEEP_DEG[-1], n):
        for hand in (1, -1):
            j = _judge(float(beta), hand)
            for side, name in ((1, "shaft1"), (2, "shaft2")):
                F_x, rho = (j["on_1"] if side == 1 else j["on_2"])[2], j["rho_mm"][side - 1]
                yield beta, hand, name, physical_couples(F_x, rho, 0.0)[0] / (j["F_tw_N"] * abs(rho))


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    judged = [r for r in out["check"] if r["verdict"] != "RECORD"]
    print(f"check records: {len(out['check'])}, judged {len(judged)}, MISS {sum(r['verdict'] == 'MISS' for r in judged)}")
    print(format_table((("shaft", "shaft", "s"), ("id", "id", "s"), ("quantity", "quantity", "s"),
                        ("rel_err", "rel err", ".2e"), ("verdict", "verdict", "s")),
                       [r for r in out["identities"] if r.get("sweep") == "base"]))
    print(format_table((("id", "id", "s"), ("quantity", "quantity", "s"), ("measured", "FEM", ".9g"),
                        ("reference", "Harris", ".9g"), ("rel_err", "rel err", ".2e"), ("verdict", "verdict", "s")),
                       [r for r in out["literature"] if r["id"].startswith("L")]))
