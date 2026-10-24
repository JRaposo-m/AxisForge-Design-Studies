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
# # FZG C40: dry Hertz contact along the path of contact
#
# Not validated.
#
# **Study:** `studies/gears/contact_along_path` (V-GEAR-05, demonstration), stage 1.
#
# At each named point A, B-, B+, C, D-, D+, E the tooth contact is a line contact of the two local equivalent
# cylinders, loaded by the share of the normal force that its tooth pair carries. This stage records the
# Hertz pressure and half-width, the radii, the speeds and the load share along the path: the inputs of the
# film stage. `--smoke`: point A only, at the K5 stage and 200 rpm of `validation/gears/fzg_geometry_ph`
# (smoke values, not the study's operating point, which is OPEN GL-39).

# %%
import sys

from axisforge.outputs._format import format_table, write_records_csv
from axisforge.solvers.machine_elements.gears.contact import contact_inputs

from slippy_bridge.contact import gear_contacts, hertz

from construction import (DRIVER, FAMILY, GEAR_E_MPA, GEAR_NU, RESULTS, WHEEL_SPEED_RPM, WHEEL_TORQUE_NM, build_pair,
                          path_times)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs every point
NAME = "01_hertz_path"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
# smoke only: FZG K5 on the wheel (fzg_geometry_ph CSV L168) at 200 rpm (L198); never a study result
T2_Nm, n2_rpm = (104.97, 200.0) if SMOKE else (WHEEL_TORQUE_NM, WHEEL_SPEED_RPM)
if T2_Nm is None or n2_rpm is None:
    raise NotImplementedError("operating point of the study: OPEN GL-39")

# %% [markdown]
# ## 1. The contacts along the path
#
# Wheel torque and speed to gear 1 (`T1 = T2 z1/z2`, `n1 = n2 z2/z1`, R-124); F_bn from `mesh_forces`;
# the contact inputs at the path times; the bridge turns each into a line `Contact` (SI).

# %%
pair = build_pair()
z1, z2 = pair.gear1.z, pair.gear2.z
n1_rpm = n2_rpm * z2 / z1
T1_Nm = T2_Nm * z1 / z2                           # gear 1 drives: T1 rot1 > 0 (OPEN Q-09)
F_bn_N = FAMILY.mesh_forces(pair, T1_Nm, rot1=+1).F_bn_N
times = path_times(pair, n1_rpm)
labels = ["A"] if SMOKE else list(times)
inputs = contact_inputs(pair, DRIVER, n1_rpm, F_bn_N, t_s=[times[k] for k in labels])
records = []
for c in gear_contacts(pair, inputs, driver=DRIVER, points=labels, E=GEAR_E_MPA, nu=GEAR_NU):
    ref = hertz(c.case)                           # Hertz line contact on the bridge axes, SI
    records.append({**c.record(), "T1_Nm": T1_Nm, "n1_rpm": n1_rpm, "F_bn_N": F_bn_N, "w_N_per_m": c.case.load,
                    "R_x_m": ref["R_x_m"], "p_H_Pa": ref["p0_Pa"], "b_m": ref["s_x_m"]})

# %% [markdown]
# ## 2. Table (no criterion: a demonstration)

# %%
print(format_table((("point", "point", "s"), ("g_y_mm", "g_y [mm]", "+.3f"), ("share", "share", ".3f"),
                    ("w_N_per_m", "w [N/mm]", ".1f", 1e-3), ("R_x_m", "R_x [mm]", ".3f", 1e3),
                    ("p_H_Pa", "p_H [MPa]", ".1f", 1e-6), ("b_m", "b [um]", ".1f", 1e6),
                    ("ue_m_per_s", "u_e [m/s]", ".3f"), ("vg_m_per_s", "v_g [m/s]", "+.3f")), records))
write_records_csv(RESULTS / f"{NAME}.csv", records)
