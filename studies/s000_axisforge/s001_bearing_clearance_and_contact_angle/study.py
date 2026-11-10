"""Run of s001: clearance (deep groove, roller) and contact angle (angular contact) against the load distribution.

``python -m studies.s000_axisforge.s001_bearing_clearance_and_contact_angle.study`` writes into ``results/``
(git-ignored): ``s001_cases.csv`` (one record per bearing and load), ``s001_elements.csv`` (one per rolling
element), ``s001_v003.csv`` (v003.check of every solved bearing), ``s001_criteria.csv`` and the figures of
``studies/outputs/l000_axisforge``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from axisforge.outputs._format import write_records_csv
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import ISO16281BallSolver

from studies.outputs.l000_axisforge.plots import bearing_loads as plots
from studies.outputs.l100_dry_contact.plots import contact as contact_plots
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check

from .construction import ALPHA_0_DEG, FA_OVER_FR, FR_N, S_MM, angular_contact, cylindrical_roller, deep_groove
from .model import load_node, records, solve_bearing

RESULTS = Path(__file__).resolve().parent / "results"
UM_PER_MM = 1e3                                            # clearance on the figure axes in um
UNVALIDATED = "past single-row limit, unvalidated (R5-42)"
TAG = "local azimuth phi_j (0 on the radial-load line); global placement convention-dependent (R-003)"


def sweep() -> list[tuple[dict, object]]:
    """The cases of proposals.md s001: (ident, bearing, Fa) per case, every one at Fr = FR_N, untilted."""
    out = []
    for s in S_MM:
        for r in FA_OVER_FR:
            out.append(({"kind": "deep_groove", "s_mm": s, "Fa_over_Fr": r}, deep_groove(s), r * FR_N))
        out.append(({"kind": "cylindrical_roller", "s_mm": s, "Fa_over_Fr": 0.0}, cylindrical_roller(s), 0.0))
    for a0 in ALPHA_0_DEG:
        for r in FA_OVER_FR:                       # Fa > 0: the contact direction of the bearing
            out.append(({"kind": "angular_contact", "alpha_0_deg": a0, "Fa_over_Fr": r}, angular_contact(a0), r * FR_N))
    return out


def _v003(bearing, analysis, ident) -> list[dict]:
    """v003.check of one solved bearing (its judge, r03), or one record saying why it could not run."""
    try:
        return [{**ident, **r} for r in v003_check(bearing, analysis.load_distribution, case="s001")]
    except Exception as exc:                                            # noqa: BLE001 -- recorded
        return [{**ident, "case": "s001", "id": "-", "verdict": "ERROR", "note": f"{type(exc).__name__}: {exc}"}]


def criteria(cases: list[dict], checks: list[dict]) -> list[dict]:
    """The criteria of README section 6 over the whole sweep, one record each."""
    out = []

    def add(cid, what, ok, detail):
        out.append({"id": cid, "criterion": what, "verdict": "PASS" if ok else "MISS", "detail": detail})

    # S1: the judge at the r03 limit: deep groove (and roller, RECORD in v003) at Fa = 0, every s
    s0 = [c for c in checks if c.get("kind") == "deep_groove" and c.get("Fa_over_Fr") == 0.0]
    judged = [c for c in s0 if c["verdict"] in ("PASS", "MISS")]
    add("S1", "v003.check PASS for the deep groove bearing at Fa = 0 (every s; s = 0 is the r03 limit)",
        bool(judged) and all(c["verdict"] == "PASS" for c in judged),
        f"{sum(c['verdict'] == 'PASS' for c in judged)}/{len(judged)} judged records PASS")
    # S2: the loaded zone shrinks and Q_max grows monotonically with s (radial load)
    for kind in ("deep_groove", "cylindrical_roller"):
        rows = sorted((c for c in cases if c["kind"] == kind and c["Fa_over_Fr"] == 0.0), key=lambda c: c["s_mm"])
        psi, Qm = np.array([c["psi_l_deg"] for c in rows]), np.array([c["Q_max_N"] for c in rows])
        add("S2", f"{kind}, Fa = 0: psi_l decreases and Q_max increases with s",
            bool(np.all(np.diff(psi) < 0) and np.all(np.diff(Qm) > 0)),
            f"psi_l {psi[0]:.2f} -> {psi[-1]:.2f} deg, Q_max {Qm[0]:.1f} -> {Qm[-1]:.1f} N")
    # S3: every row solved and balanced
    bad = [c for c in cases if not (c["ok"] and c["balanced"])]
    add("S3", "every case ok and balanced (residual <= 1e-6 max(Fr, Fa))", not bad,
        f"{len(cases) - len(bad)}/{len(cases)}; max residual {max(c['residual_N'] for c in cases):.2e} N")
    # S4: SB-45 guard on every ball case with an axial load
    guarded = [c for c in cases if c["alpha_guard"] is not None]
    add("S4", "alpha_j alpha_ref > 0 for every loaded ball when Fa > 0 (SB-45)", all(c["alpha_guard"] for c in guarded),
        f"{sum(bool(c['alpha_guard']) for c in guarded)}/{len(guarded)} cases")
    return out


def run(results=RESULTS) -> dict:
    """Solve the sweep, check through v003, write the tables and figures, return the records."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    cases, elements, checks = [], [], []
    for ident, bearing, Fa in sweep():
        analysis, post, error = solve_bearing(bearing, load_node(bearing, FR_N, Fa))
        case, rows = records(bearing, analysis, postprocessed=post, error=error, **ident)
        case["flag"] = ""
        if analysis is not None:
            found = _v003(bearing, analysis, ident)
            checks += found
            # an angular contact bearing past r03's single-row limit (Fr tan(alpha_0) / Fa > about 1.25) is an
            # axially FREE single bearing: its ring slides back until it acts as a deep groove bearing with the
            # built-in clearance 2A(1 - cos alpha_0); a mounted 7210 has a partner or a preload (not modelled)
            if any("single-row limit" in str(r.get("note", "")) for r in found):
                case["flag"] = UNVALIDATED
        # R-137: an angular contact bearing without axial load is solved and reported ok; flagged, not plotted
        if ident["kind"] == "angular_contact" and Fa == 0.0:
            case["flag"] = "R-137"
        cases.append(case)
        elements += rows
    crit = criteria(cases, checks)
    for name, recs in (("cases", cases), ("elements", elements), ("v003", checks), ("criteria", crit)):
        write_records_csv(results / f"s001_{name}.csv", recs)
    return {"cases": cases, "elements": elements, "v003": checks, "criteria": crit,
            "figures": figures(cases, elements, checks, results)}


def _series(c: dict) -> str:
    """Legend name of a case: the bearing and its swept load ratio."""
    name = {"deep_groove": "DGBB", "cylindrical_roller": "CRB", "angular_contact": "ACBB"}[c["kind"]]
    return f"{name}, Fa/Fr {c['Fa_over_Fr']:g}" if c["kind"] != "cylindrical_roller" else name


def _zone_line(bearing_case: dict, bearing) -> list[dict]:
    """Q(phi) of a ball on a dense azimuth at the solved displacements: AxisForge's own Eq. (12)
    (``ISO16281BallSolver.elements`` on a dense ``phi_j``), so the zone edge shows between the balls."""
    phi = np.radians(np.linspace(-180.0, 180.0, 721))
    dense = SimpleNamespace(A=bearing.A, alpha_0=bearing.alpha_0, phi_j=phi)
    *_, d32 = ISO16281BallSolver.elements(dense, bearing_case["delta_r_mm"], bearing_case["delta_a_mm"],
                                          np.zeros_like(phi))
    return [{"series": _series(bearing_case) + f", s {UM_PER_MM * bearing_case['s_mm']:g} um", "source": "Eq. (12)",
             "phi_deg": float(np.degrees(f)), "Q_N": float(bearing.cp * q)} for f, q in zip(phi, d32)]


def figures(cases, elements, checks, results) -> list:
    """Figures of README section 7 through studies/outputs/l000_axisforge/plots/bearing_loads (v003's models)."""
    caption = f"Fr {FR_N:g} N, psi 0; {TAG}"
    figs = []
    # element loads of the deep groove bearing at three clearances, radial load: balls on their Eq. (12) curve
    pick = [c for c in cases
            if c["kind"] == "deep_groove" and c["Fa_over_Fr"] == 0.0 and c["s_mm"] in (0.0, 0.02, 0.06)]
    recs = []
    for c in pick:
        name = _series(c) + f", s {UM_PER_MM * c['s_mm']:g} um"
        recs += [{"series": name, "source": "AxisForge", "phi_deg": e["phi_local_deg"], "Q_N": e["Q_N"]}
                 for e in elements if e["kind"] == "deep_groove" and e["Fa_over_Fr"] == 0.0 and e["s_mm"] == c["s_mm"]]
        recs += _zone_line(c, deep_groove(c["s_mm"]))
    figs.append(plots.element_loads(recs, results, "s001", variant="deep_groove_s", caption=caption))
    # angular contact, Fa = Fr, three nominal angles (Fa = 0 rows: R-137, not plotted)
    recs = []
    for c in (c for c in cases if c["kind"] == "angular_contact" and c["Fa_over_Fr"] == 1.0):
        a0 = c["alpha_0_deg"]
        name = f"ACBB alpha_0 {a0:g} deg, Fa/Fr 1"
        recs += [{"series": name, "source": "AxisForge", "phi_deg": e["phi_local_deg"], "Q_N": e["Q_N"]}
                 for e in elements if e["kind"] == "angular_contact" and e.get("alpha_0_deg") == a0
                 and e["Fa_over_Fr"] == 1.0]
        recs += [{**r, "series": name} for r in _zone_line(c, angular_contact(a0))]
    figs.append(plots.element_loads(recs, results, "s001", variant="angular_contact_alpha0", caption=caption))
    # Q_max, eps (and the r03 judge through v003) against s; eps is capped at 1 (all elements loaded)
    sweep_recs = []
    for c in (c for c in cases if c["kind"] != "angular_contact"):
        for q in ("Q_max_N", "epsilon", "delta_r_mm", "delta_a_mm"):
            sweep_recs.append({"s_um": UM_PER_MM * c["s_mm"], "quantity": q, "y": c[q], "series": _series(c),
                               "source": "AxisForge"})
    judge = [k for k in checks if k.get("kind") != "angular_contact" and k.get("Fa_over_Fr") == 0.0
             and k.get("id") in ("C3", "C5")]
    for k in judge:
        sweep_recs.append({"s_um": UM_PER_MM * k["s_mm"], "quantity": {"C3": "Q_max_N", "C5": "epsilon"}[k["id"]],
                           "y": k["reference"], "series": _series(k),
                           "source": "r03 via v003" + ("" if k["verdict"] != "RECORD" else " (RECORD)")})
    figs.append(plots.load_zone_sweep(sweep_recs, results, "s001", x="s_um", x_label=r"diametral clearance $s$ [um]",
                                      panels={"Q_max_N": r"$Q_\max$ [N]", "epsilon": r"$\varepsilon$ [-]"},
                                      limits={"epsilon": 0.5}, variant="zone_vs_s",
                                      caption=f"{caption}; eps 0.5: zero clearance, radial load; eps capped at 1 "
                                              "(all elements loaded); CRB r03 RECORD (ISO profile, R5-BE3)"))
    # the judged errors of the deep groove bearing against r03, with v003's tolerance
    errs = [{"s_um": UM_PER_MM * k["s_mm"], "quantity": k["quantity"], "series": "DGBB, Fa = 0 (v003 judged)",
             "rel_err": k["rel_err"], "tol": k["tol"], "verdict": k["verdict"]}
            for k in checks if k.get("kind") == "deep_groove" and k.get("Fa_over_Fr") == 0.0
            and k.get("id") in ("C2", "C3", "C4", "C5")]
    figs.append(contact_plots.relative_errors(errs, results, "s001", x="s_um", x_label=r"diametral clearance $s$ [um]",
                                              variant="v003", log_x=False))
    figs.append(plots.load_zone_sweep(sweep_recs, results, "s001", x="s_um", x_label=r"diametral clearance $s$ [um]",
                                      panels={"delta_r_mm": r"$\delta_r$ [mm]", "delta_a_mm": r"$\delta_a$ [mm]"},
                                      variant="displacements_vs_s",
                                      caption=f"{caption}; delta_a from the alpha_0 touching state (R5-BE4)"))
    ac = [{"alpha_0_deg": c["alpha_0_deg"], "quantity": q, "y": c[q], "series": _series(c),
           "source": UNVALIDATED if c["flag"] == UNVALIDATED else "AxisForge"}
          for c in cases if c["kind"] == "angular_contact" and c["flag"] != "R-137"
          for q in ("Q_max_N", "epsilon", "delta_r_mm", "delta_a_mm", "alpha_max_deg")]
    ac += [{"alpha_0_deg": a, "quantity": "alpha_max_deg", "y": a, "series": "nominal", "source": "alpha = alpha_0"}
           for a in (min(ALPHA_0_DEG), max(ALPHA_0_DEG))]
    figs.append(plots.load_zone_sweep(ac, results, "s001", x="alpha_0_deg", x_label=r"nominal angle $\alpha_0$ [deg]",
                                      panels={"Q_max_N": r"$Q_\max$ [N]", "epsilon": r"$\varepsilon$ [-]",
                                              "delta_r_mm": r"$\delta_r$ [mm]", "delta_a_mm": r"$\delta_a$ [mm]",
                                              "alpha_max_deg": r"$\alpha$ at $Q_\max$ [deg]"},
                                      open_sources=(UNVALIDATED,), variant="angular_contact",
                                      caption=f"{caption}; single bearing, axially free; Fa = 0 rows not drawn "
                                              "(R-137); open markers past the single-row limit, unvalidated (R5-42)"))
    return figs


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("verdict", "verdict", "s"),
                        ("detail", "detail", "s")), out["criteria"]))
