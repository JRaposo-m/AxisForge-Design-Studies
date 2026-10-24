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
from slippy_bridge.contact.measures import rel_err

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

# %% [markdown]
# ## 4. The bearing in 3D: ball loads, Hertz pressures, contact angle (PyVista)
#
# `viz.bearing.bearing_scene` reads the bearing and its solved result directly and builds a scene
# (`pyvista.MultiBlock`): rings revolved from their section, balls at `phi_j_global`, one enlarged
# contact ellipse per loaded ball and raceway (Hertz values through `slippy_bridge.contact`), load
# arrows. `viz.scene.save` writes `results/viz/<case>.vtm` (ParaView) and one PNG per field, with a
# fixed camera; the three load cases share the colour limits, so colours compare across figures.
# `viz.scene.explore(scene, vb.FIELDS, clim=clim)` opens the same scene in a window with a field
# selector.
#
# `Fr` and `Fa` are drawn as given to the solver, read as the load on the inner ring; the balls are
# drawn where `phi_j_global` puts them. Arrow and loaded zone are on the same side by construction
# of the ball solver; this figure does not test the shaft path of the open review item on `phi_Fr`.
# Between "moderate" and "heavy" both `Fr` and the diametral clearance `s` change.

# %%
import pyvista as pv
from IPython.display import Image, display

from viz import bearing as vb, scene as vs

pv.set_jupyter_backend("static")                          # committed outputs are PNG only (D-0022)
VIZ = RESULTS / "viz"                                     # git-ignored (D-0020)
CASE_TEXT = {"radial_moderate": "radial load, moderate", "radial_heavy": "radial load, heavy",
             "combined": "combined radial and axial load"}

solved = {}                                               # section 1 keeps records only: solved again (ms)
for name, load in cd.LOADS.items():
    bearing = cd.build_bearing(load["s"])
    solved[name] = bearing, ISO16281BallSolver().solve_contact(
        bearing, Fr_xz=load["Fr"], Fr_xy=0.0, Fa=load["Fa"], delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
scenes = {name: vb.bearing_scene(b, r, case=CASE_TEXT[name]) for name, (b, r) in solved.items()}
clim = vs.clim_of(list(scenes.values()), vb.FIELDS)       # common limits of the three load cases
for name, scene in scenes.items():
    vs.save(scene, vb.FIELDS, VIZ, name, clim=clim)
for name in scenes:                                       # ball load Q [N]
    display(Image(filename=str(VIZ / f"{name}_Q.png")))

# %% [markdown]
# Hertz pressure $p_0$ at the inner and the outer contact of every loaded ball (one scale for both
# raceways; ellipses enlarged, factor in the figure), and the contact angle of the combined case
# (0 deg for every loaded ball of the two radial cases).

# %%
for shown in ("radial_moderate_p0", "radial_heavy_p0", "combined_p0", "combined_alpha"):
    display(Image(filename=str(VIZ / f"{shown}.png")))

# %% [markdown]
# Axial section through the most loaded ball, contact ellipses at true size: under radial load both
# contacts sit on the groove bottom; the axial load moves them to opposite flanks, and the contact
# line through the ball centre shows $\alpha_j$. The ring shoulders are a drawing value (AxisForge
# has no shoulder geometry).

# %%
for name in ("radial_heavy", "combined"):
    side = vb.bearing_scene(*solved[name], case=CASE_TEXT[name], view="side", ellipse_scale=1.0)
    vs.save(side, {"p0": vb.FIELDS["p0"]}, VIZ, f"{name}_side", clim=clim)
    display(Image(filename=str(VIZ / f"{name}_side_p0.png")))

# %% [markdown]
# ## 5. One contact solved with SlipPY: pressure against the Hertz ellipse (PyVista)
#
# The most loaded ball of `radial_heavy` on the inner raceway, solved as a smooth dry static step
# through the bridge (`slippy_bridge.contact.static_step.run`, bridge default grid and domain, a
# few seconds). The pressure is kept in memory and in `results/viz/*.vtm`; SlipPY's own output files
# go to a temporary folder. Hill view: height = pressure. Top view: SlipPY minus Hertz on the same
# nodes. The difference is edge discretisation (one grid node at the contact edge) plus the effect
# of the finite periodic domain; both solutions share the half-space assumption, so the agreement
# checks the numerical method, not the Hertz model. No criterion is asserted here.

# %%
import tempfile

from slippy_bridge.contact import Grid, pair, static_step

from viz import contact as vc

contact = contacts(*solved["radial_heavy"], races=("inner",), elements="max")[0]
grid = Grid.around(contact.case)                          # bridge defaults
with tempfile.TemporaryDirectory() as tmp:
    record, p = static_step.run(contact, pair(contact.case, grid), grid, tmp)
CONTACT_TEXT = "Ball on the inner raceway, most loaded ball, radial load (heavy)"
hill = vc.contact_scene(contact, grid, p, case=CONTACT_TEXT)
top = vc.contact_scene(contact, grid, p, case=CONTACT_TEXT, view="top")
vs.save(hill, {"p": vc.FIELDS["p"]}, VIZ, "contact_inner_radial_heavy")
vs.save(top, vc.FIELDS, VIZ, "contact_inner_radial_heavy_top")
for shown in ("contact_inner_radial_heavy_p", "contact_inner_radial_heavy_top_dp"):
    display(Image(filename=str(VIZ / f"{shown}.png")))
