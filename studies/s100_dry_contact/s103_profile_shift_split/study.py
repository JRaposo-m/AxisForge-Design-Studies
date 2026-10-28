"""Run of s103: the profile-shift split swept at fixed x1 + x2 for C40 (spur) and H501 (helical), validated through v107.

``python -m studies.s100_dry_contact.s103_profile_shift_split.study`` writes into ``results/`` (git-ignored):
``s103_sweep.csv`` (one row per tips variant and split: path, eps_alpha, zeta_f1, zeta_f2, tip thickness, p0 max, the
checks as cut), ``s103_points.csv`` (one row per split and point A..E, B / D full load), ``s103_v107.csv`` (every
v107 ``check`` record at every split), ``s103_criteria.csv`` (S0-S2) and the figures of ``studies/outputs``.

Two tips variants (``# OPEN R5-138``, R-306): k = 0 (v107) and AxisForge's tip alteration k (``addendum_reduction``,
FEUP's printed tips). Splits per pair and variant: ``N_SWEEP`` values strictly inside the ideal range of
``model.bounds`` plus the marked ones: FEUP's split, AxisForge's equal-specific-sliding split
(``assemble_pair(x_split="equal_specific_sliding")``), x1 = x2 and the other published splits (``PUBLISHED_SPLITS``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv

from studies.outputs.l000_axisforge.plots.bearing_loads import load_zone_sweep
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.v100_dry_contact.v107_fzg_contact_path import check as v107_check
from studies.v100_dry_contact.v107_fzg_contact_path.check import TOL_CLOSED_FORM

from .construction import N_SWEEP, PAIRS, PUBLISHED_SPLITS, STAGE, WHEEL_TORQUE_NM, pair_data, split_pair, x_sum
from .model import bounds, split_point

RESULTS = Path(__file__).resolve().parent / "results"
TOL_HERTZ = 1e-9          # [-] S2: bridge Hertz vs r02 Hertz (inside v107), the same SlipPY hertz_full and inputs
TOL_EQUAL = 1e-9          # [-] S1: |zeta_A1 - zeta_E2| of r04 at AxisForge's split (brentq xtol 1e-12 on x1)
J16_NOTE = "J-16 does not enter (no shaft: the couple F_a r_w is not used)"
TIPS = {False: "k = 0", True: "k (R-306)"}
# r04 (inside v107 check) builds the tips from x with k = 0: with the tip alteration these G records compare two
# different tips, so they are not judged (a_w and T1C do not depend on the tips)
TIP_DEPENDENT = ("T1A", "T1B", "T1D", "T1E")
MARK_MERGE = 0.01         # [-] marks closer than this in x1 share one line (FEUP and x1 = x2 are 0.005 / 0.003 apart)


def _splits(name: str, b: dict) -> list[tuple[float | None, str]]:
    """(x1, tag) of the sweep: inner points of the ideal bounds, FEUP's, x1 = x2, published; (None, "equal sliding")."""
    inner = np.linspace(b["x1_lo"], b["x1_hi"], N_SWEEP + 2)[1:-1]
    return ([(float(x), "sweep") for x in inner] + [(PAIRS[name]["x"][0], "FEUP"), (x_sum(name) / 2.0, "x1 = x2"),
                                                     (None, "equal sliding")]
            + [(x, tag) for tag, x in PUBLISHED_SPLITS.get(name, {}).items()])


def sweep(name: str, *, addendum_reduction: bool = False) -> dict:
    """Every split of one pair and tips variant: model records, v107 records, criteria, marks, figure records."""
    T = WHEEL_TORQUE_NM[STAGE]
    tips = TIPS[addendum_reduction]
    b = bounds(name, addendum_reduction=addendum_reduction)
    spur = PAIRS[name]["beta_n_deg"] == 0.0
    rows, pts, v107, crit, marks = [], [], [], [], {}
    for x1, tag in _splits(name, b):
        pair = split_pair(name, x1, equal_sliding=x1 is None, addendum_reduction=addendum_reduction)
        summary, points = split_point(pair, T)
        summary.update(pair=name, tips=tips, tag=tag)
        rows.append(summary)
        pts += [{"pair": name, "tips": tips, "x1": pair.x1, "tag": tag, **p} for p in points]
        recs = [{**r, "x1": pair.x1, "tips": tips, "tag": tag}
                for r in v107_check(pair, T, data=pair_data(name, pair.x1), case="s103")]
        v107 += recs
        if tag != "sweep":
            marks[tag] = pair.x1
        judged = [r for r in recs if r["verdict"] != "RECORD"
                  and not (addendum_reduction and r["id"] == "G" and r["quantity"] in TIP_DEPENDENT)]
        for group in ("G", "P", "F"):
            g = [r for r in judged if r["id"] == group]
            crit.append({"pair": name, "tips": tips, "x1": pair.x1, "tag": tag, "made_valid": summary["made_valid"],
                         "id": f"S0/{group}", "quantity": f"v107 check {group} (worst of {len(g)})",
                         "rel_err": max(abs(r["err"]) for r in g), "tol": TOL_CLOSED_FORM,
                         "verdict": "PASS" if all(r["verdict"] == "PASS" for r in g) else "MISS"})
        ref = {(r["quantity"], r["point"]): r["reference"] for r in recs if r["id"] == "P"}
        if tag == "equal sliding":
            # S1: AxisForge's split gives equal specific sliding at the roots in r04's numbers (through v107)
            diff = ref[("zeta1", "A")] - ref[("zeta2", "E")]
            crit.append({"pair": name, "tips": tips, "x1": pair.x1, "tag": tag, "made_valid": summary["made_valid"],
                         "id": "S1", "quantity": "r04 zeta1 at A - zeta2 at E", "rel_err": diff, "tol": TOL_EQUAL,
                         "verdict": "PASS" if abs(diff) <= TOL_EQUAL else "MISS"})
        if spur:
            # S2 (consistency, not a physics judge: both sides call SlipPY hertz_full with the same inputs): the
            # model's p0 against v107's r02 Hertz at the same point and load (C, A with L0 share; B full)
            mine = {p["point"]: p["p0_MPa"] for p in points}
            for r in (r for r in recs if r["id"] == "H"):
                key = "B full" if r["point"] == "B" else r["point"]
                if key not in ("A", "C", "B full") or key not in mine:
                    continue            # v107 puts E at full load, the model at L0 (1/2): not the same input
                err = mine[key] / r["reference"] - 1.0
                crit.append({"pair": name, "tips": tips, "x1": pair.x1, "tag": tag,
                             "made_valid": summary["made_valid"], "id": f"S2/{key}",
                             "quantity": f"p0 at {key} vs r02 (v107 H)", "rel_err": err, "tol": TOL_HERTZ,
                             "verdict": "PASS" if abs(err) <= TOL_HERTZ else "MISS"})
    # limits as cut that fall inside the ideal sweep (undercut of each gear, the mate's tip on a fillet)
    for label, key in (("undercut pinion", "x1 = x_Emin1"), ("undercut wheel", "x2 = x_Emin2"),
                       ("fillet pinion", "d_Nf1 = d_Ff1"), ("fillet wheel", "d_Nf2 = d_Ff2")):
        for x in b["roots"][key]:
            if b["x1_lo"] < x < b["x1_hi"]:
                marks[label] = x
    return {"rows": rows, "points": pts, "v107": v107, "criteria": crit, "marks": marks, "bounds": b,
            "tips": tips, "fig": _figure_records(rows, pts, v107, spur, addendum_reduction)}


def _figure_records(rows, pts, v107, spur, addendum_reduction) -> list[dict]:
    """Records of ``load_zone_sweep``: AxisForge (filled), Hertz outside its validity (open), r04 / r02 lines."""
    out = []
    for r in rows:
        x = r["x1"]
        # one series name per path point in every panel (A..E), so the legend of the first panel serves all
        out += [{"x1": x, "quantity": "zeta", "series": "A", "source": "AxisForge", "y": r["zeta_f1"]},
                {"x1": x, "quantity": "zeta", "series": "E", "source": "AxisForge", "y": r["zeta_f2"]},
                {"x1": x, "quantity": "eps_alpha", "series": "eps_alpha", "source": "AxisForge", "y": r["eps_alpha"]},
                {"x1": x, "quantity": "s_an", "series": "pinion tip", "source": "AxisForge", "y": r["s_an1"]},
                {"x1": x, "quantity": "s_an", "series": "wheel tip", "source": "AxisForge", "y": r["s_an2"]}]
        out += [{"x1": x, "quantity": "T1y", "series": k, "source": "AxisForge", "y": r[f"T1{k}_mm"]} for k in "ABCDE"]
    # spur: B and D on their single-contact side (full load); helical: the L0 share at every point
    shown = ("A", "B full", "C", "D full", "E") if spur else ("A", "B", "C", "D", "E")
    for p in pts:
        if p["point"] in shown:
            out.append({"x1": p["x1"], "quantity": "p0", "series": p["point"][0],
                        "source": "AxisForge + bridge" if p["hertz_valid"] else "Hertz not valid", "y": p["p0_MPa"]})
    for r in v107:
        if r["tag"] != "sweep":
            continue
        if r["id"] == "P" and (r["quantity"], r["point"]) in (("zeta1", "A"), ("zeta2", "E")):
            out.append({"x1": r["x1"], "quantity": "zeta", "series": r["point"],
                        "source": "r04 (v107)", "y": r["reference"]})
        elif (r["id"] == "G" and r["quantity"].startswith("T1")
              and not (addendum_reduction and r["quantity"] in TIP_DEPENDENT)):
            out.append({"x1": r["x1"], "quantity": "T1y", "series": r["quantity"][2], "source": "r04 (v107)",
                        "y": r["reference"]})
        elif spur and r["id"] == "H" and r["point"] in ("A", "C", "B"):
            out.append({"x1": r["x1"], "quantity": "p0", "series": r["point"],
                        "source": "r02 (v107)", "y": r["reference"]})
    return sorted(out, key=lambda r: r["x1"])


def _merged(marks: dict) -> dict:
    """``label -> x1`` with labels closer than ``MARK_MERGE`` joined on one line (so they do not overprint)."""
    out = {}
    for label, x in sorted(marks.items(), key=lambda kv: kv[1]):
        last = next(reversed(out), None)
        if last is not None and x - out[last] < MARK_MERGE:
            out[f"{last} / {label}"] = out.pop(last)
        else:
            out[label] = x
    return out


def run(results=RESULTS) -> dict:
    """Both pairs and both tips variants, the CSV tables and the figures; returns the records and the figure paths."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    out = {(name, ar): sweep(name, addendum_reduction=ar) for name in PAIRS for ar in TIPS}
    rows = [r for o in out.values() for r in o["rows"]]
    crit = [r for o in out.values() for r in o["criteria"]]
    write_records_csv(results / "s103_sweep.csv", rows)
    write_records_csv(results / "s103_points.csv", [p for o in out.values() for p in o["points"]])
    write_records_csv(results / "s103_v107.csv", [r for o in out.values() for r in o["v107"]])
    write_records_csv(results / "s103_criteria.csv", crit)
    figures = []
    for (name, ar), o in out.items():
        spur = PAIRS[name]["beta_n_deg"] == 0.0
        b = o["bounds"]
        spans = {f"not as cut ({b['lo_cut_by']})": (b["x1_lo"], b["x1_lo_cut"]),
                 f"not as cut ({b['hi_cut_by']})": (b["x1_hi_cut"], b["x1_hi"])}
        figures.append(load_zone_sweep(
            o["fig"], results, "s103", x="x1", x_label=r"$x_1$ at fixed $x_1 + x_2$ (-)",
            panels={"T1y": r"$T_1Y$ (mm), Y = A..E", "zeta": r"$\zeta_{f1}$ at A, $\zeta_{f2}$ at E (-)",
                    "eps_alpha": r"$\varepsilon_\alpha$ (-)", "p0": r"$p_0$ (MPa)", "s_an": r"$s_{an} / m_n$ (-)"},
            limits={"eps_alpha": 1.0, "s_an": 0.0}, x_marks=_merged(o["marks"]),
            x_spans={k: v for k, v in spans.items() if v[1] > v[0]},
            filled_sources=("AxisForge + bridge",), open_sources=("Hertz not valid",),
            variant=name + ("_k" if ar else ""),
            caption=(f"{name}, tips {o['tips']}, x1 + x2 = {x_sum(name):.4f}, {STAGE} {WHEEL_TORQUE_NM[STAGE]:g} N.m on "
                     f"the wheel, pinion drives;\nideal x1 in ({b['x1_lo']:.3f}: {b['lo_by']}, {b['x1_hi']:.3f}: "
                     f"{b['hi_by']}); as cut (zero allowances, ISO 53 counterpart rack, R5-132) ({b['x1_lo_cut']:.3f}, "
                     f"{b['x1_hi_cut']:.3f});\nshaded: ideal flanks a rack cutter does not make (undercut side: "
                     f"active root not checked, trochoid not computed);\nopen p0: Hertz not valid (tip edge); p0 L0 "
                     + ("(B, D at full load)" if spur else "(A..E), mid-face slice, RECORD; " + J16_NOTE)
                     + ("" if not ar else "; r04 tips are k = 0: T1A, B, D, E not judged"))))
    # relative_errors ticks every swept x: every 4th split, x1 rounded to 2 digits (all of them in s103_criteria.csv)
    shown = {key: sorted({r["x1"] for r in o["criteria"] if r["tag"] == "sweep"})[::4] for key, o in out.items()}
    figures.append(relative_errors(
        [{"x1": round(r["x1"], 2), "quantity": r["id"], "series": f"{r['pair']}, {r['tips']}", "rel_err": r["rel_err"],
          "tol": r["tol"], "verdict": r["verdict"]}
         for r in crit if r["tag"] == "sweep" and r["x1"] in shown[(r["pair"], r["tips"] != TIPS[False])]],
        results, "s103", x="x1", x_label=r"$x_1$ (-)", log_x=False, variant="vs_v107",
        caption=("S0: v107 check (r04) worst per group (k (R-306): T1A, B, D, E not judged); S2: bridge Hertz vs r02 "
                 "through v107 (spur, consistency); every 4th split; exact zeros drawn at the floor")))
    verdicts = {v: sum(r["verdict"] == v for r in crit) for v in ("PASS", "MISS")}
    return {"pairs": out, "criteria": crit, "rows": rows, "figures": figures, "verdicts": verdicts}


if __name__ == "__main__":
    res = run()
    print("s103:", res["verdicts"])
    for (name, ar), o in res["pairs"].items():
        b = o["bounds"]
        print(f"{name} {o['tips']}: x_sum {x_sum(name):.4f}, ideal x1 in ({b['x1_lo']:.4f} {b['lo_by']}, "
              f"{b['x1_hi']:.4f} {b['hi_by']}); as cut ({b['x1_lo_cut']:.4f} {b['lo_cut_by']}, {b['x1_hi_cut']:.4f} "
              f"{b['hi_cut_by']}); x_Emin {b['x_Emin'][0]:.4f} / {b['x_Emin'][1]:.4f}; marks "
              + ", ".join(f"{k} {v:.4f}" for k, v in o["marks"].items()))
        for r in o["rows"]:
            if r["tag"] != "sweep":
                print(f"  {r['tag']:>15}: x1 {r['x1']:.4f} k {r['k']:.4f} d_a {r['d_a1_mm']:.3f}/{r['d_a2_mm']:.3f} "
                      f"eps {r['eps_alpha']:.4f} zeta_f1 {r['zeta_f1']:.3f} zeta_f2 {r['zeta_f2']:.3f} "
                      f"p0max {r['p0_max_MPa']:.1f} at {r['p0_max_at']} s_an {r['s_an1']:.3f}/{r['s_an2']:.3f} "
                      f"made_valid {r['made_valid']}")
    worst = {}
    for r in res["criteria"]:
        k = (r["pair"], r["tips"], r["id"].split("/")[0])
        worst[k] = max(worst.get(k, 0.0), abs(r["rel_err"]))
    print("worst:", {" ".join(k): f"{v:.2e}" for k, v in worst.items()})
    print("MISS:", [(r["pair"], r["tips"], r["id"], r["x1"], r["rel_err"]) for r in res["criteria"]
                    if r["verdict"] == "MISS"])
