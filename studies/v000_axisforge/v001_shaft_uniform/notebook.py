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
# # v001 shaft uniform: AxisForge shaft FEM against r01 and Onate
#
# **Question.** Does `RigidSupportFEMSolver` reproduce the closed-form beam (r01) of a uniform steel shaft
# (L 200 mm, d 20 mm) on two pins (25 / 175 mm) under F1 = 500 N at 75 mm (-Z) and F2 = 200 N at the
# overhang tip (+Y), for Euler-Bernoulli and Timoshenko elements?
#
# **Judges.** r01 (`solve_uniform`) of the same loads and section. **Literature.** Onate (2013) vol. 2:
# Table 2.1 (r_w of the one-point Timoshenko element, PDF p. 84), Table 1.1 with Fig. 1.17 (PDF p. 65-66),
# Table 2.3 with Fig. 2.17 (PDF p. 124-126).
#
# **Criteria (README section 6, fixed before the run).** Euler-Bernoulli nodal v, theta (C1, C2) and every
# model's reactions (C3) within 1e-9 of r01; the one-point Timoshenko element equal to r01 at
# `1/kGA_eff = 1/kGA - h^2/(12 EI)` within 1e-9 (C4); Richardson: Euler-Bernoulli flat (C5), Timoshenko
# extrapolated to r01 within 1e-9 (C6); Onate's printed values to half a unit of the last digit (L1-L3).

# %%
from IPython.display import Image, display
from axisforge.outputs._format import format_table

from studies.v000_axisforge.v001_shaft_uniform import study

out = study.run()

# %% [markdown]
# ## 1. Deflection and rotation along the shaft, 8 elements
# Filled markers: AxisForge nodes; lines: r01. Euler-Bernoulli markers sit on their line already at 8 elements
# (nodal exactness); the one-point Timoshenko markers sit slightly inside their dashed line (too stiff, O(h^2)).

# %%
for f in out["figures"][:2]:
    display(Image(filename=str(f)))

# %% [markdown]
# ## 2. Judged records (C1-C6) and the error against the number of elements

# %%
print(format_table((("theory", "theory", "s"), ("n_elements", "n", "d"), ("plane", "plane", "s"), ("id", "id", "s"),
                    ("quantity", "quantity", "s"), ("rel_err", "rel err", ".2e"), ("tol", "tol", ".0e"),
                    ("verdict", "verdict", "s")),
                   [r for r in out["check"] if r["verdict"] != "RECORD" and (r["n_elements"] in (8, 128))]))
display(Image(filename=str(out["figures"][2])))

# %% [markdown]
# ## 3. Mesh series of the load-point deflection (Richardson, `numerical_solvers`)
# Euler-Bernoulli is flat (nothing to extrapolate); one-point Timoshenko has p = 2 and extrapolates to r01;
# full integration converges slowly from far (shear locking).

# %%
print(format_table((("theory", "theory", "s"), ("level", "n", ".0f"), ("value", "v [mm]", ".12f"),
                    ("status", "status", "s"), ("p", "p", ".6f"), ("f_extrapolated", "extrapolated [mm]", ".12f"),
                    ("gci_fine_medium", "GCI fm", ".2e"), ("r01_v_probe_mm", "r01 [mm]", ".12f")), out["richardson"]))

# %% [markdown]
# ## 4. Shear effect: Timoshenko / Euler-Bernoulli at F1 against L/d (slender limit 1)

# %%
print(format_table((("L_over_d", "L/d", "g"), ("ratio_r01", "r01", ".6f"), ("ratio_fem", "FEM n=128", ".6f"),
                    ("rel_err_ratio", "rel err", ".2e")), out["shear"]))
display(Image(filename=str(out["figures"][3])))

# %% [markdown]
# ## 5. Onate (2013): Table 2.1 (L1), Table 1.1 (L2), Table 2.3 (L3; L4 thick RECORD, nu and k_z not printed)

# %%
print(format_table((("id", "id", "s"), ("table", "table", "s"), ("quantity", "quantity", "s"),
                    ("measured", "AxisForge", ".6g"), ("reference", "printed", ".6g"), ("tol_abs", "tol", ".1e"),
                    ("verdict", "verdict", "s")), out["literature"]))
display(Image(filename=str(out["figures"][4])))

# %% [markdown]
# ## 6. Verdict

# %%
rows = out["check"] + out["literature"]
judged = [r for r in rows if r["verdict"] != "RECORD"]
print(f"judged {len(judged)}, PASS {sum(r['verdict'] == 'PASS' for r in judged)}, "
      f"MISS {sum(r['verdict'] == 'MISS' for r in judged)}")
