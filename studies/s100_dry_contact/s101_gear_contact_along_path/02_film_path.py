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
# # FZG C40: EHL film along the path of contact, quasi-steady
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` in `ehl.run` (the EHL solve is a scaffold).
#
# **Study:** `studies/gears/contact_along_path` (V-GEAR-05, demonstration), stage 2.
#
# Each point of the path is solved as a steady, isothermal, Newtonian line contact at its own mean
# entrainment speed and load per unit length (quasi-steady: the film is assumed to follow the point at once;
# sliding is ignored). No published film exists along a gear path (R-145). The check S1 is the Moes line
# formula at the same (M, L); the FEUP film formula (Fernandes2015a Eqs 3-4, sum speed) is shown only as the
# mean/sum bracket, a factor 2^0.67 = 1.59, never pass/fail (GL-14). `--smoke`: point A only, at the smoke
# operating point of `01_hertz_path`.

# %%
import math
import sys

from axisforge.outputs._format import format_table, write_records_csv
from axisforge.solvers.machine_elements.gears.contact import contact_inputs

from references.contact.ehl import e_prime, film_scalings, groups, moes
from slippy_bridge.contact import Grid, gear_contacts
from slippy_bridge.contact.measures import rel_err
from slippy_bridge.lubrication import ehl

from construction import (DRIVER, FAMILY, GEAR_E_MPA, GEAR_NU, OIL, OIL_FERNANDES2015A, OIL_MARTINS2015, RESULTS,
                          ROELANDS_P0_PA, SPAN_X, WHEEL_SPEED_RPM, WHEEL_TORQUE_NM, build_pair, path_times)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs every point
NAME = "02_film_path"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
T2_Nm, n2_rpm = (104.97, 200.0) if SMOKE else (WHEEL_TORQUE_NM, WHEEL_SPEED_RPM)   # smoke: as 01_hertz_path
if T2_Nm is None or n2_rpm is None:
    raise NotImplementedError("operating point of the study: OPEN GL-39")
FEUP_BRACKET = 2.0 ** 0.67                        # h ~ U^0.67: reading U with the sum instead of the mean speed

# %% [markdown]
# ## 1. The contacts along the path (as `01_hertz_path`)

# %%
pair = build_pair()
z1, z2 = pair.gear1.z, pair.gear2.z
n1_rpm = n2_rpm * z2 / z1
F_bn_N = FAMILY.mesh_forces(pair, T2_Nm * z1 / z2, rot1=+1).F_bn_N        # gear 1 drives (OPEN Q-09)
times = path_times(pair, n1_rpm)
labels = ["A"] if SMOKE else list(times)
path = gear_contacts(pair, contact_inputs(pair, DRIVER, n1_rpm, F_bn_N, t_s=[times[k] for k in labels]),
                     driver=DRIVER, points=labels, E=GEAR_E_MPA, nu=GEAR_NU)

# %% [markdown]
# ## 2. One steady EHL solve per point
#
# The oil (source, name, alpha) is OPEN R-295; eta0, z and rho at 80 C are not printed. Two finite convex radii
# (no flat body) have no EHL check yet (GL-35).

# %%
source, oil_name, alpha_key = OIL                 # OPEN R-295
oil = {"Martins2015": OIL_MARTINS2015, "Fernandes2015a": OIL_FERNANDES2015A}[source][oil_name]
records = []
for c in path:
    grid = Grid.around(c.case, span_x=SPAN_X)                                    # OPEN R-048: inlet/outlet domain
    rec, p, gap = ehl.run(c, grid, RESULTS / NAME, eta0_Pa_s=oil["eta0_Pa_s"], rho0_kg_per_m3=oil["rho80_kg_per_m3"],
                          viscosity=("roelands", oil["roelands_z"], ROELANDS_P0_PA),
                          density="dowson_higginson")                        # ue from c.kinematics
    # the run and the Moes formula must see the same pressure-viscosity slope (OPEN GL-40)
    if not math.isclose(rec["alpha_initial_per_Pa"], oil[alpha_key], rel_tol=1e-9):
        raise ValueError(f"SlipPY initial alpha {rec['alpha_initial_per_Pa']} is not the printed {oil[alpha_key]}")
    E = e_prime(c.case.E[0], c.case.nu[0], c.case.E[1], c.case.nu[1])
    g = groups(u_mean_m_per_s=rec["ue_m_per_s"], eta0_Pa_s=oil["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
               R_x_m=rec["R_x_m"], alpha_per_Pa=rec["alpha_initial_per_Pa"], w_N_per_m=c.case.load)
    H_run = film_scalings(h_m=rec["h_c_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"],
                          U_mean=g["U_mean"])["H_moes"]
    H_formula = moes(M=g["M_moes"], L=g["L_moes"], line=True, compressible=rec["density_model"] != "incompressible",
                     p_h_Pa=rec["p0_ref_Pa"])["H_c"]
    records.append({**rec, "oil": f"{source}:{oil_name}:{alpha_key}", "M_moes": g["M_moes"], "L_moes": g["L_moes"],
                    "H_c_run": H_run, "H_c_moes": H_formula, "err_H_c": rel_err(H_run, H_formula),
                    "feup_bracket_factor": FEUP_BRACKET})

# %% [markdown]
# ## 3. Check S1 (demonstration)
#
# S1: per point, h_c within +-10 % of the Moes line formula at the same M, L **[proposed]**. The FEUP bracket
# (x 1.59 between the mean and the sum reading) is recorded only (GL-14).

# %%
criteria = [{"id": "S1", "quantity": f"H_c at {r['point']} [proposed]", "value": r["H_c_run"],
             "expected": r["H_c_moes"], "err": r["err_H_c"], "limit": "<= 0.10",
             "result": "SMOKE" if SMOKE else ("PASS" if abs(r["err_H_c"]) <= 0.10 else "FAIL")} for r in records]
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.2f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
