# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # v102 ellipses: elliptical BEM contacts against Hertz, alpha 1-100
#
# **Question.** Does the SlipPY BEM, through `slippy_bridge`, reproduce the Hertz ellipse (approach, p0,
# semi-axes, ellipticity, orientation) of two bodies given with their own radii and the angle between them,
# over `alpha = R_y/R_x` 1-100, and the worked contacts of Hamrock and Brewe (1983) Table 5 (PDF p. 6)?
#
# **Systems.** Two steel cylinders R 10 mm crossed at `theta = arccos((alpha - 1)/(alpha + 1))` (Table 1
# alphas, PDF p. 4), 50 N; Table 5: wheel on rail, ball on plane, ball on the outer ring (concave groove).
# **Reference.** r02 `hertz` (SlipPY `hertz_full`) at the BEM load. Spatial kernel, interference control,
# principal grid, `b/h` 16 and 32 asked (v104's rule), half-length 2.
#
# **Criteria (README section 6, fixed before the run).** v101's C1 (approach 0.5 %), C3 (p0 `M^-3/2`),
# C4 (semi-axes `1/M`); E1 `k = a/b` within `2/M`; E2 orientation of the BEM major axis within `h/a` rad;
# T5: Table 5 values within the v101 tolerance plus half a unit of the printed digit. Judged on the spatial
# kernel at `b/h >= 16` with the Hertz reference in range (a/R <= 0.1 per body); RECORD otherwise.

# %%
from IPython.display import Image, display
from axisforge.outputs._format import format_table

from studies.v100_dry_contact.v102_hertz_ellipses import study

out = study.run()
cols = (("literature_case", "case", "s"), ("b_over_h", "b/h", ".1f"), ("id", "id", "s"), ("quantity", "quantity", "s"),
        ("measured", "BEM", ".6g"), ("reference", "reference", ".6g"), ("rel_err", "err [%]", "+.4f", 100.0),
        ("tol", "tol [%]", ".3f", 100.0), ("verdict", "verdict", "s"))

# %% [markdown]
# ## 1. Alpha sweep (crossed cylinders, Table 1 alphas)

# %%
print(format_table(cols, [r for r in out["check"] if "_t1_" in r["literature_case"] and r["id"] not in ("L1", "L2")]))
display(Image(filename=str(out["figures"][0])))

# %% [markdown]
# ## 2. Hamrock and Brewe Table 5

# %%
print(format_table(cols, [r for r in out["check"] if "_t5_" in r["literature_case"] and r["id"] not in ("L1", "L2")]))
display(Image(filename=str(out["figures"][1])))

# %% [markdown]
# ## 3. Verdict

# %%
judged = [r for r in out["check"] if r["verdict"] != "RECORD"]
misses = [r for r in judged if r["verdict"] == "MISS"]
print(f"judged {len(judged)}, PASS {len(judged) - len(misses)}, MISS {len(misses)}")
print(format_table(cols, misses))
