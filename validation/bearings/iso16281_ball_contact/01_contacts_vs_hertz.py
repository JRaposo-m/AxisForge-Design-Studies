# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Contacts of a solved ball bearing against an independent Hertz
#
# **Case:** `validation/bearings/iso16281_ball_contact`
# **Type:** implementation check, same theory (AxisForge `cp` is Hertz; the judge is Hertz)
#
# For each load case the bearing is solved with the ISO/TS 16281 ball solver, every loaded ball
# is handed to the bridge (`slippy_bridge.contact.contacts`: one contact per raceway, SI units,
# rolling radius at the operating angle $\alpha_j$) and the independent Hertz
# (`references.contact.hertz.hertz_point`) is evaluated on it.
#
# Nothing is asserted here: every pass/fail criterion lives in `tests/` (see the README). This
# script produces the table of real contacts ($Q_j$, $\alpha_j$, $p_0$, semi-axes, approach per
# raceway) and reports two findings:
#
# - **F0** AxisForge's total approach $\delta_j$ against the summed Hertz approach at $\alpha_0$:
#   expected ratio $(1.48096/1.48)^{2/3} = 1.000433$ (rounded constant in `cp`).
# - **F1** AxisForge evaluates the raceway curvatures at the free angle $\alpha_0$, the bridge at
#   $\alpha_j$: effect on $p_0$ and on the approach, per raceway.

# %%
import math
import pathlib
from collections import defaultdict

from axisforge.outputs._format import format_table, write_records_csv
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import element_records
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    ISO16281BallSolver,
)

from references.contact.hertz import hertz_point
from slippy_bridge.contact import contact_case, contacts
from slippy_bridge.contact.outputs import rel_err

import construction as cd

# the case folder: the script's own folder, or the working directory when run as a notebook
HERE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
RESULTS = HERE / "results"                                          # git-ignored (D-0020)
RESULTS.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Solve, extract the contacts, evaluate Hertz
#
# AxisForge has no centrifugal term, so the inner and the outer contact of a ball carry the same
# $Q_j$ and $\alpha_j$ and differ only in the raceway radii. The ellipse edge angle is the
# contact angle plus the angle the semi-axis across rolling subtends on the groove,
# $\alpha_j + s_y / |r_\mathrm{groove}|$: $s_y$ is taken as an arc; read as a tangent-plane
# half-width (asin) the angle is about 0.5 deg larger. Measured about the groove centre of
# curvature. Indicative only: at $s_y / r = 0.38$ Hertz is strained. AxisForge has no shoulder
# geometry, so the angle is compared with nothing.
#
# Each row starts from `Contact.record()`, which carries `label` and `case_hash`: the join key
# with the static-step records of the BEM study.

# %%
contact_records, approach_records = [], []
for name, load in cd.LOADS.items():
    bearing = cd.build_bearing(load["s"])
    result = ISO16281BallSolver().solve_contact(bearing, Fr_xz=load["Fr"], Fr_xy=0.0, Fa=load["Fa"],
                                                delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
    approach_alpha0_m = defaultdict(float)                # Hertz, summed over both raceways, per ball [m]
    for c in contacts(bearing, result):                   # loaded balls, inner and outer raceway
        race, alpha_j = c.location["race"], c.location["alpha_rad"]
        judge = hertz_point(c.case.r1, c.case.r2, c.case.E, c.case.nu, c.case.load)
        # the same contact with the raceway radii as AxisForge stores them, at the free angle alpha_0
        at_alpha0 = contact_case(bearing, race, c.case.load)
        judge_alpha0 = hertz_point(at_alpha0.r1, at_alpha0.r2, at_alpha0.E, at_alpha0.nu, at_alpha0.load)
        approach_alpha0_m[c.location["element_j"]] += judge_alpha0["delta_m"]
        contact_records.append({
            "case": name, "solver_ok": bool(result.ok), **c.record(), "Q_N": c.case.load, **judge,
            "p0_alpha0_Pa": judge_alpha0["p0_Pa"], "delta_alpha0_m": judge_alpha0["delta_m"],
            "p0_alpha0_rel_diff": rel_err(judge_alpha0["p0_Pa"], judge["p0_Pa"]),
            "delta_alpha0_rel_diff": rel_err(judge_alpha0["delta_m"], judge["delta_m"]),
            # r2[1] is the groove radius [m], concave (negative): its magnitude. s_y as an arc.
            "edge_angle_rad": alpha_j + judge["s_y_m"] / abs(c.case.r2[1]),
        })
    for e in element_records(result):                     # AxisForge: delta_mm = approach of the two raceways
        if e["j"] in approach_alpha0_m:                   # loaded balls only
            approach_records.append({
                "case": name, "solver_ok": bool(result.ok), "j": e["j"], "Q_N": e["Q_N"],
                "delta_j_mm": e["delta_mm"], "delta_hertz_alpha0_m": approach_alpha0_m[e["j"]],
                # AxisForge [mm] -> [m], the only unit conversion of this script
                "delta_ratio": e["delta_mm"] * 1e-3 / approach_alpha0_m[e["j"]]})

write_records_csv(RESULTS / "contacts.csv", contact_records)
write_records_csv(RESULTS / "approach.csv", approach_records)

# %% [markdown]
# ## 2. Contacts: load, angle, Hertz pressure, semi-axes and approach per raceway
#
# `s_x` is the semi-axis along rolling, `s_y` across it (along the groove). The columns
# "(a0)" repeat $p_0$ and the approach with the raceway radii at $\alpha_0$; "diff" is
# (value at $\alpha_0$ - value at $\alpha_j$) / value at $\alpha_j$ (finding F1).

# %%
DEG = math.degrees(1.0)
CONTACT_COLUMNS = [
    ("case", "case", ""), ("race", "race", ""), ("element_j", "j", "d"),
    ("phi_local_rad", "phi [deg]", ".1f", DEG), ("Q_N", "Q [N]", ".1f"),
    ("alpha_rad", "alpha [deg]", ".3f", DEG), ("p0_Pa", "p0 [MPa]", ".1f", 1e-6),
    ("s_x_m", "s_x [mm]", ".4f", 1e3), ("s_y_m", "s_y [mm]", ".4f", 1e3),
    ("delta_m", "delta [um]", ".3f", 1e6),
    ("p0_alpha0_Pa", "p0 (a0) [MPa]", ".1f", 1e-6), ("p0_alpha0_rel_diff", "diff [%]", "+.3f", 100.0),
    ("delta_alpha0_m", "delta (a0) [um]", ".3f", 1e6), ("delta_alpha0_rel_diff", "diff [%]", "+.3f", 100.0),
    ("edge_angle_rad", "edge [deg]", ".2f", DEG),
]
print(format_table(CONTACT_COLUMNS, contact_records))

# %% [markdown]
# ## 3. Total approach: AxisForge against the summed Hertz approach at $\alpha_0$ (finding F0)
#
# `Q_j = cp delta_j**1.5` with the constant 1.48 where Hertz gives $\pi\sqrt{2}/3 = 1.48096$, so
# the ratio is expected to be 1.000433 for every ball and every load.

# %%
APPROACH_COLUMNS = [
    ("case", "case", ""), ("solver_ok", "solved", ""), ("j", "j", "d"), ("Q_N", "Q [N]", ".1f"),
    ("delta_j_mm", "delta_j AxisForge [um]", ".4f", 1e3),
    ("delta_hertz_alpha0_m", "Hertz inner+outer (a0) [um]", ".4f", 1e6),
    ("delta_ratio", "ratio", ".6f"),
]
print(format_table(APPROACH_COLUMNS, approach_records))
print(f"\nwritten: {(RESULTS / 'contacts.csv').relative_to(HERE)}, {(RESULTS / 'approach.csv').relative_to(HERE)}")
