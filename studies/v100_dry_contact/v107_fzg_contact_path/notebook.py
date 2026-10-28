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
# # v107 FZG contact along the path: AxisForge against r04 + r02 and Martins 2015
#
# **Question.** Does AxisForge (`pair_values_at`, `path_points`, `mesh_forces`) give the geometry, the flank
# radii and specific sliding at A..E, and the mesh forces of the FZG C40 (spur) and H501 (helical) pairs as the
# independent involute judge r04 does, and as Martins 2015 Table 4 (PDF p. 4) prints; and which reading of the
# printed Hertzian pressure p_H does FEUP use (Q-09)?
#
# **Criteria (README section 6, fixed before the run).** G, P, F against r04 within 1e-10 (P-12); Martins
# F_r, F_a within half a printed unit plus the torque's half unit; p_H readings (Hertz at C, full load; Hertz at
# A, its load share) RECORD. Flags: wheel torques (R-124), E 206 GPa and nu 0.3 assumed (R-118), pinion drives.

# %%
from axisforge.outputs._format import format_table

from studies.v100_dry_contact.v107_fzg_contact_path import study

out = study.run()
cols = (("pair", "pair", "s"), ("stage", "stage", "s"), ("id", "id", "s"), ("point", "pt", "s"),
        ("quantity", "quantity", "s"), ("axisforge", "AxisForge", ".7g"), ("reference", "reference", ".7g"),
        ("err", "err", "+.2e"), ("verdict", "verdict", "s"))
print(format_table(cols, [r for r in out["check"] if r["stage"] == "K9" and r["id"] in ("G", "P", "F")]))

# %% [markdown]
# ## Martins 2015 Table 4 and the p_H readings (all stages)

# %%
print(format_table(cols, [r for r in out["check"] if r["id"] in ("L", "H")]))

# %%
judged = [r for r in out["check"] if r["verdict"] != "RECORD"]
print(f"judged {len(judged)}, MISS {sum(r['verdict'] == 'MISS' for r in judged)}; largest r04 difference "
      f"{max(abs(r['err']) for r in judged if r['id'] in 'GPF'):.1e}")
