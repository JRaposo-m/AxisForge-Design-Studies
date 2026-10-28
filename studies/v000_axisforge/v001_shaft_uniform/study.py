"""Run of v001: the uniform shaft through the AxisForge FEM, four beam models, five meshes, against r01 and Onate.

``python -m studies.v000_axisforge.v001_shaft_uniform.study`` writes into ``results/`` (git-ignored):
``v001_check.csv`` (check records of every model and mesh), ``v001_richardson.csv`` (the mesh series
of the load-point deflection), ``v001_shear.csv`` (Timoshenko / Euler-Bernoulli vs L/d),
``v001_literature.csv`` (Onate Tables 2.1, 1.1, 2.3) and the figures of ``studies/outputs``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver

from numerical_solvers import RichardsonOptions, richardson
from studies.outputs.l000_axisforge.plots import shaft as plots

from . import literature
from .check import TOL_NODAL, along_shaft, check, literature_record, reference, section
from .construction import LENGTH_MM, N_ELEMENTS, SUPPORTS_MM, THEORIES, uniform_nodes, uniform_shaft

RESULTS = Path(__file__).resolve().parent / "results"
X_PROBE_MM = 75.0                                  # F1, plane XZ: the load-point deflection of the mesh series
# shear sweep, L = 200 mm (d = 100 ... 2 mm). Below span/d ~ 3 (L/d 2: span/d 1.5, load-to-support a/d 0.5) neither
# beam theory is a valid shaft model (Saint-Venant end zones, local contact): there the ratio compares two models only
L_OVER_D = (2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
N_FINE = N_ELEMENTS[-1]
# formal order of the nodal error, the max_p of Richardson (R3-17): Hermite cubic O(h^4) (its nodal
# values are exact, so the series is flat); 2-node linear Timoshenko O(h^2) (kga_reduced_integration)
ORDER = {"euler_bernoulli": 4.0, "timoshenko/cowper/single_point": 2.0, "timoshenko/hutchinson/single_point": 2.0,
         "timoshenko/cowper/exact": 2.0}
CAPTION = ("uniform shaft L 200 mm, d 20 mm, E 210 GPa, nu 0.3; pins at 25 / 175 mm; F1 500 N at 75 mm (-Z), "
           "F2 200 N at 200 mm (+Y); no convention tag (forces only)")


def solve(system, settings, n_elements: int, *, kGA_override: float | None = None):
    """``RigidSupportFEMSolver`` on a uniform mesh of ``n_elements`` over the whole shaft."""
    return RigidSupportFEMSolver(settings, kGA_override=kGA_override).solve(
        system, extra_mandatory=uniform_nodes(n_elements, system.shaft.total_length))


def _node(result, x_mm: float, key: str = "v_xz") -> float:
    """Nodal value of a ShaftResults field at the node at ``x_mm``."""
    i = int(np.argmin(np.abs(np.asarray(result.x_nodes) - x_mm)))
    return float(np.asarray(getattr(result, key))[i])


def mesh_series(system) -> tuple[list[dict], list[dict], dict]:
    """Every beam model on every mesh: check records, Richardson records, and the solved coarse meshes."""
    checks, conv, coarse = [], [], {}
    for name, settings in THEORIES.items():
        def evaluate(n, settings=settings, name=name):
            r = solve(system, settings, int(n))
            checks.extend(check(system, r, settings))
            if int(n) == N_ELEMENTS[0]:
                coarse[name] = r
            return {"v_probe_mm": _node(r, X_PROBE_MM)}
        # flat_atol = 0: the deflection is ~1e-2 mm, the AxisForge default 1e-9 absolute would call any
        # series flat (R3-10); stop_when_converged False: every level, for the figure
        res = richardson(evaluate, N_ELEMENTS, options=RichardsonOptions(kind="count", flat_atol=0.0, max_p=ORDER[name],
                                                                         stop_when_converged=False),
                         name="v001_mesh", label=name)
        EI, kGA = section(system, settings)
        exact = float(reference(system, EI, kGA)["XZ"].v(X_PROBE_MM))
        for rec in res.records():
            conv.append({**rec, "theory": name, "r01_v_probe_mm": exact})
        checks.append(_richardson_verdict(name, res.final["v_probe_mm"], exact))
    return checks, conv, coarse


def _richardson_verdict(name: str, fin, exact: float) -> dict:
    """C5: the Euler-Bernoulli series is flat (nodally exact); C6: the one-point Timoshenko series extrapolates
    to r01 within TOL_NODAL (its error is exactly c h^2, kga_reduced_integration); full integration RECORD."""
    err = fin.f_extrapolated / exact - 1.0
    if name == "euler_bernoulli":
        cid, tol, ok = "C5", 0.0, fin.status == "flat"
    elif THEORIES[name].integration_method == "single_point":
        cid, tol, ok = "C6", TOL_NODAL, fin.status == "monotonic" and abs(err) <= TOL_NODAL
    else:
        cid, tol, ok = "T2", float("nan"), None
    return {"case": "v001", "label": "richardson", "theory": name, "plane": "XZ", "n_elements": N_ELEMENTS[-1],
            "h_mm": LENGTH_MM / N_ELEMENTS[-1], "uniform_mesh": True, "id": cid, "quantity": "v_probe extrapolated",
            "unit": "mm", "max_abs_diff": abs(fin.f_extrapolated - exact), "ref_scale": abs(exact), "rel_err": err,
            "tol": tol, "verdict": "RECORD" if ok is None else ("PASS" if ok else "MISS"),
            "reference": f"r01, Richardson {fin.status}, p {fin.p:.6g}, GCI_fm {fin.gci_fine_medium:.3g}", "kGA_N": float("nan")}


def shear_sweep(l_over_d=L_OVER_D, *, fem: bool = True) -> list[dict]:
    """Timoshenko (Cowper, one-point) / Euler-Bernoulli deflection at F1 vs L/d: r01, and the FEM at n = 128."""
    eb, tim = THEORIES["euler_bernoulli"], THEORIES["timoshenko/cowper/single_point"]
    out = []
    for lod in l_over_d:
        system = uniform_shaft(diameter_mm=LENGTH_MM / lod)
        refs = [float(reference(system, *section(system, s))["XZ"].v(X_PROBE_MM)) for s in (eb, tim)]
        row = {"L_over_d": float(lod), "d_mm": LENGTH_MM / lod, "r01_v_eb_mm": refs[0], "r01_v_timoshenko_mm": refs[1],
               "ratio_r01": refs[1] / refs[0]}
        if fem:
            v_eb, v_t = (_node(solve(system, s, N_FINE), X_PROBE_MM) for s in (eb, tim))
            row.update({"n_elements": N_FINE, "v_eb_mm": v_eb, "v_timoshenko_mm": v_t, "ratio_fem": v_t / v_eb,
                        "rel_err_ratio": (v_t / v_eb) / row["ratio_r01"] - 1.0})
        out.append(row)
    return out


def literature_series() -> tuple[list[dict], list[dict]]:
    """Onate Tables 2.1 (L1), 1.1 (L2), 2.3 (L3 slender, L4 thick RECORD): records and the r_w figure records."""
    t21, t11, t23 = literature.cases()
    rows, rw_fig = [], []
    eb, ri = THEORIES["euler_bernoulli"], THEORIES["timoshenko/cowper/single_point"]

    # Table 2.1: r_w = w_tip / (P L^3 / (3 EI)) of the cantilever by symmetry, lambda -> inf
    system = t21.build()
    L, P = literature.HALF_LENGTH_MM, literature.TIP_FORCE_N
    EI, _ = section(system, eb)
    k_slender = literature.SLENDER_KGA_FACTOR * EI / L**2
    w_eb = -P * L**3 / (3.0 * EI)                       # along -Z: negative, as the FEM's v_xz
    for row in t21.published:
        n = int(row["n_elements_printed"])
        r_w = _node(solve(system, ri, 2 * n, kGA_override=k_slender), L) / w_eb
        rows.append(literature_record("L1", "r_w", "-", r_w, row["value"], row["resolution"], source=t21.source,
                                      table="Table 2.1", n_elements_printed=n, element="2-node Timoshenko one-point",
                                      closed_form=1.0 - 1.0 / (4.0 * n * n) + 3.0 * EI / (k_slender * L**2)))
        rw_fig += [{"n_elements": n, "y": r_w, "series": "r_w", "source": "FEM"},
                   {"n_elements": n, "y": row["value"], "series": "r_w", "source": "published"}]
    rw_fig += [{"n_elements": float(n), "y": 1.0 - 1.0 / (4.0 * n * n), "series": "r_w", "source": "1 - 1/(4 n^2)"}
               for n in np.geomspace(1.0, 16.0, 60)]

    # Table 1.1 / Fig. 1.17: AxisForge Euler-Bernoulli central deflection error must print 0.00 %
    for load in ("point", "udl"):
        system = t11.build(load)
        x_mid = 0.5 * literature.ONATE_LENGTH_MM
        exact = float(reference(system, *section(system, eb))["XZ"].v(x_mid))
        for n in sorted({int(r["n_elements_printed"]) for r in t11.published if r["table"] == "Table 1.1"}):
            err = 100.0 * (_node(solve(system, eb, 2 * n), x_mid) / exact - 1.0)
            printed = {r["element"]: r["value"] for r in t11.published
                       if r["table"] == "Table 1.1" and r["load"] == load and int(r["n_elements_printed"]) == n}
            # L2 is C1 on Onate's SI system (the 0.0 is ours: Onate prints only the CCB / CVB errors); it covers
            # the unit conversion and the consistent load vector of the uniform load
            rows.append(literature_record("L2", "central deflection error (C1 on Onate system)", "%", err, 0.0, 0.01,
                                          source=t11.source,
                                          table="Table 1.1", load=load, n_elements_printed=n,
                                          element="AxisForge Euler-Bernoulli", printed_CCB=printed["CCB"],
                                          printed_CVB=printed["CVB"], r01_w_m=exact / literature.M_TO_MM))

    # Table 2.3: slender = Euler-Bernoulli on 2 elements per half (judged), thick = Timoshenko one-point on
    # 32 per half with Onate's example nu, k_z (RECORD, OPEN R5-24)
    for row in t23.published:
        if row["table"] != "Table 2.3":
            continue
        hL = row["h_over_L"]
        system = t23.build(hL)
        x_mid = 0.5 * literature.ONATE_LENGTH_MM
        if row["element"] == "slender solution":
            w_m = _node(solve(system, eb, 4), x_mid) / literature.M_TO_MM
            rows.append(literature_record("L3", "w_c slender", "m", w_m, row["value"], row["resolution"],
                                          source=t23.source, table="Table 2.3", h_over_L=hL, n_elements_printed=2,
                                          element="AxisForge Euler-Bernoulli"))
        else:
            k = literature.kga_square(hL)
            w_m = _node(solve(system, ri, 64, kGA_override=k), x_mid) / literature.M_TO_MM
            r01_m = float(reference(system, section(system, eb)[0], k)["XZ"].v(x_mid)) / literature.M_TO_MM
            rows.append(literature_record("L4", "w_c thick", "m", w_m, row["value"], row["resolution"],
                                          source=t23.source, judged=False, table="Table 2.3", h_over_L=hL,
                                          n_elements_printed=32, element="AxisForge Timoshenko one-point",
                                          r01_w_m=r01_m, kGA_N=k, nu_assumed=literature.ONATE_NU))
    return rows, rw_fig


def run(results=RESULTS) -> dict:
    """Mesh series, shear sweep and literature; write the CSV tables and the figures; return the records."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    system = uniform_shaft()
    checks, conv, coarse = mesh_series(system)
    shear = shear_sweep()
    lit, rw_fig = literature_series()
    write_records_csv(results / "v001_check.csv", checks)
    write_records_csv(results / "v001_richardson.csv", conv)
    write_records_csv(results / "v001_shear.csv", shear)
    write_records_csv(results / "v001_literature.csv", lit)

    n0 = N_ELEMENTS[0]
    fields = (along_shaft(system, coarse["euler_bernoulli"], THEORIES["euler_bernoulli"], series="Euler-Bernoulli")
              + along_shaft(system, coarse["timoshenko/cowper/single_point"], THEORIES["timoshenko/cowper/single_point"],
                            series="Timoshenko (Cowper, one-point)"))
    figures = [plots.deflection_along_shaft([r for r in fields if r["plane"] == plane], results, "v001",
                                            supports_mm=SUPPORTS_MM, loads_mm=(load_x,), variant=f"{plane}_n{n0}",
                                            caption=f"plane {plane}, n = {n0}; Timoshenko: dashed r01 at kGA, dotted r01 at "
                                                    f"kGA_eff (C4); {CAPTION}")
               for plane, load_x in (("XZ", X_PROBE_MM), ("XY", LENGTH_MM))]
    # error of the deflection field, worst plane, per model and mesh
    label = {("C1", "euler_bernoulli"): "Euler-Bernoulli vs r01 (C1)",
             ("T1", "timoshenko/cowper/single_point"): "Timoshenko one-point vs r01 (T1)",
             ("C4", "timoshenko/cowper/single_point"): "Timoshenko one-point vs r01 at kGA_eff (C4)",
             ("T1", "timoshenko/cowper/exact"): "Timoshenko full integration vs r01 (T1)"}
    err = {}
    for r in checks:
        key = label.get((r["id"], r["theory"]))
        if key and r["quantity"] == "v":
            err[(key, r["n_elements"])] = max(err.get((key, r["n_elements"]), 0.0), r["rel_err"])
    figures.append(plots.error_vs_elements([{"series": s, "n_elements": n, "rel_err": e} for (s, n), e in err.items()],
                                           results, "v001", tol=TOL_NODAL,
                                           order_guides={"Timoshenko one-point vs r01 (T1)": 2},
                                           caption=f"max nodal |v - v_r01| / max |v_r01|, both planes; {CAPTION}"))
    shear_fig = ([{"L_over_d": r["L_over_d"], "y": r["ratio_fem"], "series": "v_T / v_EB", "source": "FEM",
                   "rel_err": r["rel_err_ratio"]} for r in shear]
                 + [{"L_over_d": r["L_over_d"], "y": r["ratio_r01"], "series": "v_T / v_EB", "source": "r01"}
                    for r in shear_sweep(np.geomspace(L_OVER_D[0], L_OVER_D[-1], 40), fem=False)])
    figures.append(plots.ratio_sweep(shear_fig, results, "v001", x="L_over_d", x_label=r"$L/d$ (-)",
                                     y_label=r"$v_\mathrm{T}/v_\mathrm{EB}$ at F1 (-)", limit=1.0, variant="shear",
                                     caption=f"Cowper kappa, n = {N_FINE}; shaft of the case, d = L/(L/d), L 200 mm; ratio at F1, plane XZ; "
                                             "model-to-model ratio, L/d 2-3 outside beam theory (span/d < 3)"))
    figures.append(plots.ratio_sweep(rw_fig, results, "v001", x="n_elements", x_label=r"elements $n$ (-)",
                                     y_label=r"$r_w = w/w^{EB}$ (-)", limit=1.0, variant="onate_table_2_1",
                                     caption="Onate (2013) Table 2.1 (PDF p. 84): cantilever, end load, "
                                             "one-point Timoshenko, lambda -> inf; built as a span 2L, 2P at mid-span"))
    return {"check": checks, "richardson": conv, "shear": shear, "literature": lit, "figures": figures}


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("theory", "theory", "s"), ("n_elements", "n", "d"), ("plane", "plane", "s"), ("id", "id", "s"),
                        ("quantity", "quantity", "s"), ("rel_err", "rel err", ".2e"), ("tol", "tol", ".0e"),
                        ("verdict", "verdict", "s")), out["check"]))
    print(format_table((("id", "id", "s"), ("table", "table", "s"), ("quantity", "quantity", "s"),
                        ("measured", "AxisForge", ".6g"), ("reference", "printed", ".6g"), ("tol_abs", "tol", ".1e"),
                        ("verdict", "verdict", "s")), out["literature"]))
