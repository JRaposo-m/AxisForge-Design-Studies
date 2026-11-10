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
# # v104 BEM discretisation: grid spacing, domain and kernel on smooth Hertz contacts
#
# **Question.** Which grid spacing `h`, domain half-length and SlipPY kernel give a stated error in the
# approach and the peak pressure of a smooth Hertz point contact, solved through `slippy_bridge`? The answer
# is the grid rule this repository uses (v102, v103, Part C of R2, s101).
#
# **Systems.** v101's ball (12.7 mm) on a flat and an ellipsoid (R_x 10 mm, R_y 50 mm) on a flat, steel,
# 500 N, load control. **References.** r02 Hertz (through v101 `reference`); the grid studies of Polonsky
# and Keer (1999, PDF p. 9-10) and Muser (2014, PDF p. 7).
#
# **Criteria (README section 6, fixed before the run).** D1: spatial kernel, approach and p0 change less
# than 0.05 % against the largest domain. L1: Polonsky-Keer `eps_r < 1/M` where `M >= 64` (their range).
# L3: Muser's rule (`h = a/32`, cell `8a`): contact area from a fit of the pressure profile within 0.1 %.
# Recorded: every error of the sweeps, the order of `eps_p`, the frequency kernel's domain change.

# %%
from IPython.display import Image, display
from axisforge.outputs._format import format_table

from studies.v100_dry_contact.v104_bem_discretisation import study

out = study.run()

# %% [markdown]
# ## 1. Grid-spacing sweep (spatial kernel at half-length 2, frequency kernel at 4)

# %%
cols = (("label", "contact", "s"), ("kernel", "kernel", "s"), ("b_over_h", "b/h", ".1f"), ("span_x", "span", ".2f"),
        ("n_x", "n_x", "d"), ("n_y", "n_y", "d"), ("err_delta", "delta [%]", "+.3f", 100.0),
        ("err_p0", "p0 [%]", "+.3f", 100.0), ("err_b", "b [%]", "+.3f", 100.0), ("eps_r", "eps_r [%]", ".3f", 100.0),
        ("eps_p", "eps_p [%]", ".3f", 100.0), ("err_area_fit", "area fit [%]", "+.3f", 100.0),
        ("solve_time_s", "t [s]", ".1f"))
print(format_table(cols, [r for r in out["check"] if r["sweep"] == "h"]))
display(Image(filename=str(out["figures"][0])))

# %% [markdown]
# ## 2. Domain sweep (`b/h` 16 asked) and D1

# %%
print(format_table(cols, [r for r in out["check"] if r["sweep"] == "span"]))
print()
print(format_table((("id", "id", "s"), ("label", "contact", "s"), ("kernel", "kernel", "s"), ("span_x", "span", ".2f"),
                    ("value", "change [%]", ".4f", 100.0), ("limit", "tol [%]", ".3f", 100.0),
                    ("verdict", "verdict", "s")), [r for r in out["judged"] if r["id"] == "D1"]))
display(Image(filename=str(out["figures"][1])))

# %% [markdown]
# ## 3. The published grid studies (circle)

# %%
print(format_table((("id", "id", "s"), ("kernel", "kernel", "s"), ("M", "M", ".1f"), ("span_x", "span", ".2f"),
                    ("quantity", "quantity", "s"), ("value", "value", ".5f"), ("limit", "limit", ".5f"),
                    ("verdict", "verdict", "s")), [r for r in out["judged"] if r["id"] != "D1"]))

# %% [markdown]
# ## 4. The grid rule: coarsest `b/h` (asked and rounded down to a power of two in metres) from which the
# error stays within the stated value on every finer grid of the sweep

# %%
print(format_table((("label", "contact", "s"), ("kernel", "kernel", "s"), ("quantity", "quantity", "s"),
                    ("tol", "error [%]", ".2f", 100.0), ("span_x", "span", ".2f"), ("b_over_h_min", "b/h min", ".2f"),
                    ("reached", "reached", "s")), out["grid_rule"]))
judged = [r for r in out["judged"] if r["verdict"] != "RECORD"]
print(f"judged {len(judged)}, MISS {sum(r['verdict'] == 'MISS' for r in judged)}")
