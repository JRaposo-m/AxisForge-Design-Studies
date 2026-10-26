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
# # Effect of Shear Deformation on a Stepped Shaft Under Distributed Loads: Euler–Bernoulli vs. Timoshenko
#
# **Case:** `stepped_shafts / distributed_load / case_distributed_gear_parabolic`
# **Study type:** model comparison (study, not validation)
# **Status:** provisional
#
# ## Question
#
# What is the contribution of shear deformation to the deflection of a
# stepped shaft loaded only by distributed loads — gear mesh forces spread
# uniformly over the face width and a parabolic user load — and how much do
# the Cowper and Hutchinson shear correction factors differ?
#
# ## Method in brief
#
# The same resolved system is solved with three beam models — Euler–Bernoulli
# (EB) and Timoshenko with the Cowper and Hutchinson factors $\kappa$ — on the
# **minimal mesh** (mandatory nodes only). Each result is compared with the
# exact solution of its own beam theory (`references/beams/stepped_beam.py`), so that
# discretisation error is not mistaken for a physical effect. Mesh
# convergence of the Timoshenko element is treated in notebook 02.

# %%
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from validation._support.figures import shaft_studies as fg
from references.beams import stepped_beam as an
import construction as cd
from axisforge.core.loads import LoadPlane

fg.use_style()
pd.set_option("display.float_format", lambda v: f"{v:.4g}")

# %% [markdown]
# ## 1. Case definition
#
# All input data are taken from `construction.py`; nothing is redeclared
# in this notebook.

# %%
segs = cd.segments()
system = cd.build_system()
SUPPORTS = (cd.BALL_POSITION_MM, cd.ROLLER_POSITION_MM)

pd.DataFrame(
    [(label, s.x_lo, s.x_hi, s.d, s.I, s.A) for (_, _, label), s in zip(cd.SECTIONS, segs)],
    columns=["Section", "x_lo / mm", "x_hi / mm", "d / mm", "I / mm⁴", "A / mm²"],
)

# %%
pd.DataFrame(
    [
        ("Material", "—", cd.MATERIAL_ID, "—"),
        ("Young's modulus", "E", cd.E_MPA, "MPa"),
        ("Poisson's ratio", "ν", cd.POISSON, "—"),
        ("Shoulder fillet radius", "r", cd.SHOULDER_FILLET_MM, "mm"),
        ("Locating bearing (6204)", "x_A", cd.BALL_POSITION_MM, "mm"),
        ("Non-locating bearing (NU204)", "x_B", cd.ROLLER_POSITION_MM, "mm"),
        ("Gear face width", "b", cd.GEAR_FACE_WIDTH_MM, "mm"),
        ("Transmitted power", "P", cd.POWER_W / 1e3, "kW"),
        ("Parabolic load, resultant", "F", cd.PARABOLIC_RESULTANT_N, "N"),
        ("Parabolic load, width", "w", cd.PARABOLIC_WIDTH_MM, "mm"),
        ("Parabolic load, peak intensity", "q₀", cd.PARABOLIC_PEAK_N_PER_MM, "N/mm"),
    ],
    columns=["Quantity", "Symbol", "Value", "Unit"],
)

# %% [markdown]
# Distributed loads acting on each shaft after the gear mesh has been
# resolved, with their resultants in the two bending planes:

# %%
rows = []
for ss in system.shafts:
    for ld in ss.distributed_radial_loads:
        rows.append((ss.name, ld.label, ld.source, ld.x_lo, ld.x_hi, float(ld.magnitude),
                     float(ld.resultant_component(LoadPlane.XY)),
                     float(ld.resultant_component(LoadPlane.XZ))))
pd.DataFrame(rows, columns=["Shaft", "Load", "Source", "x_lo / mm", "x_hi / mm",
                            "|F| / N", "F_xy / N", "F_xz / N"])

# %%
def layout_figure(shaft: str):
    ss = next(s for s in system.shafts if s.name == shaft)
    x = np.linspace(0.0, cd.SHAFT_LENGTH_MM, 4001)
    colours = {"parabolic_user_load": "#eb6834"}
    series = []
    for ld in ss.distributed_radial_loads:
        q = np.array([abs(ld.q(xi)) if ld.x_lo <= xi <= ld.x_hi else 0.0 for xi in x])
        series.append(fg.Series(label=f"{ld.label} ({abs(ld.magnitude):.0f} N)", x=x, y=q,
                                color=colours.get(ld.label, "#2a78d6" if ld.label.endswith("Ft") else "#1baf7a")))
    edges = [segs[0].x_lo] + [s.x_hi for s in segs]
    return fg.shaft_layout(edges, [s.d for s in segs], series,
                           title=f"{shaft}: geometry and distributed load intensities |q(x)|",
                           supports=SUPPORTS, support_names=("A", "B"))

fig_layout_1 = layout_figure("shaft1")

# %%
fig_layout_2 = layout_figure("shaft2")

# %% [markdown]
# ## 2. Theoretical background
#
# **Euler–Bernoulli.** Plane cross-sections remain perpendicular to the
# neutral axis; on a stepped shaft the bending stiffness is piecewise
# constant:
#
# $$ \frac{d^2 v}{dx^2} = \frac{M(x)}{EI(x)}. $$
#
# **Timoshenko.** The cross-section rotation $\theta$ is independent of the
# slope; the shear strain, uniform over the section after the correction
# factor $\kappa$, is
#
# $$ \frac{d\theta}{dx} = \frac{M}{EI(x)}, \qquad
#    \gamma = \frac{dv}{dx} - \theta = \frac{V}{\kappa G A(x)}, \qquad
#    V = -\frac{dM}{dx}. $$
#
# The shear contribution to the deflection is therefore
# $v_s(x) = \int V/(\kappa G A)\,dx + C_1 x + C_2$. On a uniform shaft this
# reduces to $-M/(\kappa GA)$; on a stepped shaft $\kappa G A$ changes at each
# shoulder and the integral must be carried out section by section. In both
# cases the shear contribution adds to the bending contribution.
#
# **Shear correction factors, solid circular section:**
#
# $$ \kappa_\text{Cowper} = \frac{6(1+\nu)}{7+6\nu} \quad\text{(Cowper, 1966)}, \qquad
#    \kappa_\text{Hutchinson} = \frac{6(1+\nu)^2}{7+12\nu+4\nu^2} \quad\text{(Hutchinson, 2001)}. $$
#
# **Static determinacy.** With two simple supports the reactions and $M(x)$
# follow from equilibrium alone and are **identical for all three models**;
# only the deflections and rotations change.
#
# **Distributed loads in the FEM.** Loads are converted into consistent
# nodal loads: forces and moments for the EB (Hermite) element, forces only
# for the Timoshenko element, whose rotation is interpolated independently
# of the deflection.

# %%
kappa = {name: f(cd.POISSON) for name, f in an.KAPPA.items()}

from axisforge.mesh.shaft.element_type.shear_factor import ShearFactor  # cross-check
sf = ShearFactor()
assert np.isclose(kappa["cowper"], sf.cowper_factor(cd.POISSON))
assert np.isclose(kappa["hutchinson"], sf.hutchinson_factor(cd.POISSON))

exact = {
    (ss.name, name): an.solve_shaft(ss, segs, None if st.shear_theory is None else kappa[st.shear_theory],
                                    cd.SHAFT_LENGTH_MM)
    for ss in system.shafts for name, st in cd.THEORIES.items()
}
xd = np.linspace(0.0, cd.SHAFT_LENGTH_MM, 4001)
pd.DataFrame(
    [(shaft, name, float(np.max(sol.v(xd))), float(xd[np.argmax(sol.v(xd))]))
     for (shaft, name), sol in exact.items()],
    columns=["Shaft", "Model", "v_max exact / mm", "x(v_max) / mm"],
)

# %% [markdown]
# ## 3. Mesh
#
# Only the mandatory nodes are used: shaft ends, section boundaries
# (shoulders), bearing centres and extents, gear centre and face-width
# limits, and the limits and centroids of the distributed loads.

# %%
results = {name: cd.solve(system, settings) for name, settings in cd.THEORIES.items()}
r0 = results["euler_bernoulli"]["shaft1"]
display(Markdown(f"Minimal mesh: **{len(r0.x_nodes)} nodes**, "
                 f"element length {np.min(np.diff(r0.x_nodes)):.1f}–{np.max(np.diff(r0.x_nodes)):.1f} mm."))

# %% [markdown]
# ## 4. Results
#
# ### 4.1 Static determinacy check
#
# The reactions must coincide across models to round-off precision. The
# bending moment is sampled differently by the two elements: the Hermite
# element gives a linear $M$ within each element and the linear Timoshenko
# element a constant one, so under distributed loads both deviate from the
# exact, curved $M(x)$ on a coarse mesh.

# %%
rows = []
for name, res in results.items():
    for shaft, r in res.items():
        R_A, R_B = (b.Fr for b in r.bearing_nodes)
        sol = exact[(shaft, name)]
        rows.append((shaft, name, R_A, R_B,
                     float(np.hypot(sol.xy.R_A, sol.xz.R_A)), float(np.hypot(sol.xy.R_B, sol.xz.R_B)),
                     r.v_max, r.M_max, float(np.max(sol.M(xd)))))
summary = pd.DataFrame(rows, columns=["Shaft", "Model", "R_A FEM / N", "R_B FEM / N",
                                      "R_A exact / N", "R_B exact / N",
                                      "v_max FEM / mm", "M_max FEM / N·mm", "M_max exact / N·mm"])
summary

# %% [markdown]
# ### 4.2 Deflection along the shaft
#
# Upper panel: resultant deflection $v = \sqrt{v_{xy}^2 + v_{xz}^2}$ of the
# three FEM models on the minimal mesh. Lower panel: difference relative to
# the EB result, normalised by the maximum EB deflection.

# %%
STYLE = {
    "euler_bernoulli": dict(label="Euler–Bernoulli", color="#2a78d6", linestyle="-"),
    "timoshenko/cowper": dict(label=r"Timoshenko, $\kappa$ Cowper", color="#eb6834", linestyle="--"),
    "timoshenko/hutchinson": dict(label=r"Timoshenko, $\kappa$ Hutchinson", color="#1baf7a", linestyle="-."),
}

def shear_figure(shaft: str):
    ref = results["euler_bernoulli"][shaft]
    scale = np.max(np.abs(ref.v))
    series, deltas = [], []
    for name, res in results.items():
        r = res[shaft]
        series.append(fg.Series(x=r.x, y=r.v, **STYLE[name]))
        if name != "euler_bernoulli":
            deltas.append(fg.Series(x=r.x, y=100 * (np.asarray(r.v) - np.asarray(ref.v)) / scale,
                                    **STYLE[name]))
    return fg.field_comparison(
        series, deltas,
        title=f"{shaft}: resultant deflection, minimal mesh ({len(ref.x_nodes)} nodes)",
        ylabel=r"$v$ / mm",
        delta_label=r"$(v - v_\mathrm{EB})\,/\,\max|v_\mathrm{EB}|$ / %",
        supports=SUPPORTS, support_names=("A", "B"),
    )

fig_shaft1 = shear_figure("shaft1")

# %%
fig_shaft2 = shear_figure("shaft2")

# %% [markdown]
# ### 4.3 Comparison with the analytical solution
#
# Each FEM result is compared with the exact solution of the same beam
# theory. The error is the maximum nodal deviation over both bending planes,
# normalised by the maximum exact deflection.

# %%
def nodal_error(r, sol, field):
    x = np.asarray(r.x)
    num = max(float(np.max(np.abs(np.asarray(getattr(r, f"{field}_{p}")) - getattr(getattr(sol, p), field)(x))))
              for p in ("xy", "xz"))
    den = max(float(np.max(np.abs(getattr(getattr(sol, p), field)(xd)))) for p in ("xy", "xz"))
    return num / den

rows = []
for (shaft, name), sol in exact.items():
    r = results[name][shaft]
    rows.append((shaft, name, float(np.max(sol.v(xd))), r.v_max,
                 nodal_error(r, sol, "v"), nodal_error(r, sol, "M")))
check = pd.DataFrame(rows, columns=["Shaft", "Model", "v_max exact / mm", "v_max FEM / mm",
                                    "nodal error v", "nodal error M"])
check

# %% [markdown]
# ## 5. Conclusions

# %%
ck = check.set_index(["Shaft", "Model"])
def inc(shaft, model):
    return 100 * (ck.loc[(shaft, model), "v_max exact / mm"]
                  / ck.loc[(shaft, "euler_bernoulli"), "v_max exact / mm"] - 1)
sm = summary.set_index(["Shaft", "Model"])
inc_fem = 100 * (sm.loc[("shaft1", "timoshenko/cowper"), "v_max FEM / mm"]
                 / sm.loc[("shaft1", "euler_bernoulli"), "v_max FEM / mm"] - 1)
display(Markdown(f"""
1. **Physical effect (exact solution):** shear deformation increases the maximum deflection
   by **{inc('shaft1', 'timoshenko/cowper'):.2f} %** / **{inc('shaft2', 'timoshenko/cowper'):.2f} %**
   (shaft1 / shaft2, Cowper) and **{inc('shaft1', 'timoshenko/hutchinson'):.2f} %** /
   **{inc('shaft2', 'timoshenko/hutchinson'):.2f} %** (Hutchinson). The choice between the two
   factors is immaterial for this shaft.
2. **Euler–Bernoulli on the minimal mesh** is exact at the nodes for the deflection
   (error {ck.loc[('shaft1', 'euler_bernoulli'), 'nodal error v']:.1e}), which verifies the
   consistent load vector of the distributed loads and the piecewise stiffness. Its bending
   moment is not exact under distributed loads (error
   {100 * ck.loc[('shaft1', 'euler_bernoulli'), 'nodal error M']:.1f} %), because the
   Hermite element represents $M$ as linear within each element.
3. **Timoshenko on the minimal mesh** deviates from its exact solution by
   **{100 * ck.loc[('shaft1', 'timoshenko/cowper'), 'nodal error v']:.1f} %** in deflection,
   of the same order as the shear effect itself: the FEM shows {inc_fem:+.1f} % relative to EB
   instead of {inc('shaft1', 'timoshenko/cowper'):+.2f} %. The minimal mesh is **not adequate
   for quantifying the shear effect** with the linear Timoshenko element; the converged mesh
   is determined in notebook 02.
4. The reactions coincide for all models and with the exact values, as required by static
   determinacy.
"""))

# %% [markdown]
# ## References
#
# - Cowper, G. R. (1966). The shear coefficient in Timoshenko's beam theory.
#   *J. Appl. Mech.* 33(2), 335–340.
# - Hutchinson, J. R. (2001). Shear coefficients for Timoshenko beam theory.
#   *J. Appl. Mech.* 68(1), 87–92.
# - Timoshenko, S. P. & Gere, J. M. *Mechanics of Materials* — deflection of
#   statically determinate beams by integration.
