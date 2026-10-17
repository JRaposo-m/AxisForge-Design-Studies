# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FZG C40, load stages K5, K7, K9: Hertz pressure along the path against Martins2015 Table 4
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today in the gear adapter (`contacts`, OPEN R-023 R-031).
#
# **Case:** `validation/gears/fzg_geometry_ph` (V-GEAR-04), criteria P0, P1, P2 of the README.
#
# The wheel torque of each stage becomes the torque on gear 1, `T1 = T2 z1/z2`, and the wheel speed the
# speed of gear 1, `n1 = n2 z2/z1` (R-124); AxisForge gives the normal force `F_bn` (`mesh_forces`) and the
# contact inputs at the named points (`contact_inputs`); the bridge turns them into line contacts and Hertz
# gives p_H. P0 checks the radial force against the table (the wheel -> pinion conversion, d_w and alpha_wt).
# P1, Jose's V-GEAR-04 criterion, is the load-ratio test p_H(K7)/p_H(K5), p_H(K9)/p_H(K5): it proves the stage
# torques and p ~ sqrt(w) only, and is blind to the geometry, the point, E and the driver. p_H at A, B+ and C
# is recorded only (P2): 498 MPa at A, half load, pinion driving is the R-117 hypothesis, not a criterion.
# Criteria P0-P2: spur C40 only (R-027); section 3 also draws H501 (figures, mid-face section). `--smoke`: point A
# only.

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
from axisforge.outputs._format import format_table, read_records_csv, write_records_csv
from axisforge.solvers.machine_elements.gears.contact import contact_inputs       # c328999 only (GL-01)

from slippy_bridge.contact import contacts, hertz
from slippy_bridge.contact.outputs import rel_err
from validation._support.figures.gear_plots import load_stages, path_of_contact
from validation._support.style import use_style

from construction import DRIVER, FAMILY, RESULTS, WHEEL_SPEED_RPM, WHEEL_TORQUE_NM, build_pair, path_times

use_style()
SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs every point
NAME = "02_ph"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # the printed rows, copied unchanged

# %% [markdown]
# ## 1. The printed rows

# %%
rows = read_records_csv(REFERENCE, text_fields=("case_id", "kind", "quantity", "symbol", "value", "unit",
                                                "condition", "source_key", "location", "read_from", "note"))


def at(line, quantity, condition):
    """The printed value at CSV ``line`` (float), checked to be ``quantity`` under ``condition``."""
    r = rows[line - 2]
    if quantity not in r["quantity"] or condition not in r["condition"]:
        raise ValueError(f"CSV L{line} is not {quantity} / {condition}: the copied file changed")
    return float(r["value"])


# stage -> CSV line of p_H (Spur C40) and of the radial force F_r (record)
P_H_LINES = {"K5": (174, 172), "K7": (184, 182), "K9": (194, 192)}
p_H_printed = {k: at(lp, "Hertzian contact pressure", f"stage={k};gear=Spur C40") for k, (lp, _) in P_H_LINES.items()}
F_r_printed = {k: at(lf, "radial force", f"stage={k};gear=Spur C40") for k, (_, lf) in P_H_LINES.items()}
n2_rpm = WHEEL_SPEED_RPM[0]                       # 200 rpm, CSV L198; p_H does not depend on the speed

# %% [markdown]
# ## 2. Mesh forces per stage; P0
#
# P0: F_r within 0.5 N of the printed integer (its rounding) **[proposed]**; it checks the wheel -> pinion
# conversion (R-124), the working pitch diameter and the working pressure angle (OPEN GL-41).

# %%
COLUMNS = (("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".5g"), ("expected", "expected", ".5g"),
           ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"), ("result", "result", "s"))
pair = build_pair("C40")
z1, z2 = pair.gear1.z, pair.gear2.z
n1_rpm = n2_rpm * z2 / z1                         # wheel speed -> pinion speed (R-124)
T1 = {stage: T2_Nm * z1 / z2 for stage, T2_Nm in WHEEL_TORQUE_NM.items()}   # torque on gear 1 [N m] (R-124)
# gear 1 drives: T1 rot1 > 0 (OPEN Q-09 R-124)
forces = {stage: FAMILY.mesh_forces(pair, T1[stage], rot1=+1) for stage in WHEEL_TORQUE_NM}
criteria = []
for stage, f in forces.items():
    diff = f.F_r_N - F_r_printed[stage]
    criteria.append({"id": "P0", "quantity": f"F_r {stage} [N] [proposed]", "value": f.F_r_N,
                     "expected": F_r_printed[stage], "err": rel_err(f.F_r_N, F_r_printed[stage]), "limit": "<= 0.5 N",
                     "result": "PASS" if abs(diff) <= 0.5 else "FAIL"})
print(format_table(COLUMNS, criteria))
# figure: C40 and H501; the printed helical column "Helical H501/951" is keyed "H501" to share a group with the
# computed H501 (same F_r for both: one column in Martins2015 Table 4)
H501 = build_pair("H501")
F_r_H501_printed = {k: at(lf, "radial force", f"stage={k};gear=Helical H501/951")
                    for k, lf in {"K5": 175, "K7": 185, "K9": 195}.items()}
forces_H501 = {stage: FAMILY.mesh_forces(H501, T2_Nm * H501.gear1.z / H501.gear2.z, rot1=+1)   # wheel -> pinion (R-124)
               for stage, T2_Nm in WHEEL_TORQUE_NM.items()}
fig = load_stages({**{(stage, "C40"): F_r_printed[stage] for stage in forces},
                   **{(stage, "H501"): v for stage, v in F_r_H501_printed.items()}},
                  {**{(stage, "C40"): f.F_r_N for stage, f in forces.items()},
                   **{(stage, "H501"): f.F_r_N for stage, f in forces_H501.items()}},
                  quantity=r"$F_\mathrm{r}$", unit="N")
fig.savefig(RESULTS / "02_load_stages_Fr.png")
plt.close(fig)

# %% [markdown]
# ## 3. Along the path, stage K9: radii, speeds, load (figures, AxisForge only)
#
# One `contact_inputs` record per pair, at a dense grid of times plus the one-sided times of the named points
# (`path_times`, GL-09, GL-24), so that the steps of the share and of w at B and D are drawn as steps. No
# bridge here: p_H along the path is not drawn (OPEN R-022, R-118). H501 is drawn too (figures only): its B
# and D bound no single-contact zone (GL-25), so only A, C, E are named, and its radii and speeds are those of
# the mid-face section (R-027). Same speed and wheel torque conversion as section 2 (R-124).

# %%
STAGE = "K9"
N_DENSE = 201                                     # steps from A to E: ~0.1 mm apart on a ~20 mm path
for name in ("C40", "H501"):
    p, F_bn_N = (pair, forces[STAGE].F_bn_N) if name == "C40" else (H501, forces_H501[STAGE].F_bn_N)
    n1 = n2_rpm * p.gear2.z / p.gear1.z           # wheel speed -> pinion speed (R-124)
    named = path_times(p, n1)
    t = np.union1d(np.linspace(0.0, named["E"], N_DENSE), list(named.values()))   # sorted, one-sided times kept
    points = FAMILY.path_points(p, DRIVER)
    # mathtext without spaces: the wrapped note line never breaks inside "F_bn = 6371 N"
    note = rf"{name}, {STAGE}, $n_1={n1:.0f}\,\mathrm{{rpm}}$, $F_{{bn}}={F_bn_N:.0f}\,\mathrm{{N}}$, {DRIVER} drives"
    helical = p.gear1.betab != 0.0
    if helical:                                   # B and D bound no single-contact zone (GL-25)
        points = {k: points[k] for k in "ACE"}
    kinematics, load = path_of_contact(
        contact_inputs(p, DRIVER, n1, F_bn_N, t_s=t), points, note=note,
        kinematics_note="helical: radii and speeds at mid face width (R-027)" if helical else None,
        load_note="helical: share and w over the whole contact line (Q36)" if helical else None)
    kinematics.savefig(RESULTS / f"02_path_kinematics_{name}.png")
    load.savefig(RESULTS / f"02_path_load_{name}.png")
    plt.close(kinematics)
    plt.close(load)

# %% [markdown]
# ## 4. AxisForge -> bridge -> Hertz, per stage and point

# %%
times = path_times(pair, n1_rpm)
labels = ["A"] if SMOKE else list(times)
records = []
for stage, T2_Nm in WHEEL_TORQUE_NM.items():
    T1_Nm = T1[stage]
    inputs = contact_inputs(pair, DRIVER, n1_rpm, forces[stage].F_bn_N, t_s=[times[k] for k in labels])
    for c in contacts(pair, inputs, points=labels):
        ref = hertz(c.case)                       # Hertz line contact, SI
        records.append({**c.record(), "stage": stage, "T2_Nm": T2_Nm, "T1_Nm": T1_Nm, "F_bn_N": forces[stage].F_bn_N,
                        "F_r_N": forces[stage].F_r_N, "F_r_printed_N": F_r_printed[stage],
                        "p_H_MPa": ref["p0_Pa"] / 1e6,   # Pa -> MPa, the unit of the printed table
                        "p_H_printed_MPa": p_H_printed[stage]})

# %% [markdown]
# ## 5. Criteria P1, P2
#
# P1: at every point, p_H(K7)/p_H(K5) within 0.2 % of 685/498 = 1.3755 and p_H(K9)/p_H(K5) within 0.2 % of
# 874/498 = 1.7550 **[proposed]**. Hertz: p_H grows as the square root of the load whatever E and the point,
# so P1 proves the stage torques and p ~ sqrt(w) only. The 0.2 % is set by the rounding of the printed
# integers (0.5/498 + 0.5/685 = 0.17 %).
# P2: p_H at A, B+ and C beside the printed value, recorded only (R-117, R-118, GL-30).

# %%
by = {(r["stage"], r["point"]): r for r in records}       # Contact.record() flattens the location
for point in labels:
    base = by[("K5", point)]["p_H_MPa"]
    for stage in ("K7", "K9"):
        ratio, ratio_ref = by[(stage, point)]["p_H_MPa"] / base, p_H_printed[stage] / p_H_printed["K5"]
        err = rel_err(ratio, ratio_ref)
        criteria.append({"id": "P1", "quantity": f"p_H {stage}/K5 at {point} [proposed]", "value": ratio,
                         "expected": ratio_ref, "err": err, "limit": "<= 0.002",
                         "result": "PASS" if abs(err) <= 0.002 else "FAIL"})
    if point in ("A", "B+", "C"):
        for stage in WHEEL_TORQUE_NM:
            r = by[(stage, point)]
            criteria.append({"id": "P2", "quantity": f"p_H {stage} at {point} [MPa] (record, R-117)",
                             "value": r["p_H_MPa"], "expected": r["p_H_printed_MPa"],
                             "err": rel_err(r["p_H_MPa"], r["p_H_printed_MPa"]), "limit": "record",
                             "result": "RECORD"})
print(format_table(COLUMNS, criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
