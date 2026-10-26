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
# # Elliptical EHL contact (D = 0.2): SlipPY grid sweep against Canzi2010 Table 1
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today at `canzi_moes` (OPEN GL-13).
#
# **Case:** `validation/ehl/canzi2010` (V-EHL-05), criteria C1, C2 of the README.
#
# Canzi's point (alpha_bar 10, M2 2027, D 0.2, compressible) is turned into (M, L), then into an SI contact,
# and solved on a sequence of grids; the film is put in Moes' scaling. `--smoke`: the coarsest grid only.

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from references.contact.ehl import canzi_moes, e_prime, film_scalings, groups, moes_inverse
from slippy_bridge.contact import ContactCase, Grid, semi_axes
from slippy_bridge.contact.measures import rel_err
from slippy_bridge.lubrication import ehl

from construction import (DENSITY, E_PA, GRID_ALPHA_BAR, GRID_D, GRID_M2, GRID_N, GRID_SMOKE_N, NU, RESULTS,
                          RHO0_KG_PER_M3, R_X_M, ROELANDS_P0_PA, ROELANDS_Z, SPAN_X, SPAN_Y, U_MEAN_M_PER_S)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs the full sweep
NAME = "01_grid"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # Table 1 rows, copied unchanged

# %% [markdown]
# ## 1. The printed rows
#
# Rows are taken by their CSV line (header = line 1) and checked; the symbol column is never printed (GL-23).

# %%
rows = read_records_csv(REFERENCE, text_fields=("case_id", "kind", "quantity", "symbol", "value", "unit",
                                                "condition", "source_key", "location", "read_from", "note"))


def at(line, symbol, condition):
    """The CSV row at ``line``, checked to hold ``symbol`` under ``condition`` (a substring)."""
    r = rows[line - 2]
    if r["symbol"] != symbol or condition not in r["condition"]:
        raise ValueError(f"CSV L{line} is not {condition}: the copied file changed")
    return r


# Canzi's grid n -> CSV lines of (H_c, H_m); L20-21 (2048^2) is the converged reference (C1)
TABLE1 = {128: (12, 13), 256: (14, 15), 512: (16, 17), 1024: (18, 19), 2048: (20, 21)}
REF_HC = float(at(20, "H_c", "grid=2048x2048")["value"])
REF_HM = float(at(21, "H_m", "grid=2048x2048")["value"])
RUN_N = (GRID_SMOKE_N,) if SMOKE else GRID_N      # SlipPY grids, nodes per side [-]

# %% [markdown]
# ## 2. The SI point and the sweep
#
# (alpha_bar, M2, D) -> (M, L) by Eq. (19) with kappa and E_c (`canzi_moes(D=...)`, OPEN GL-13). The body
# has `R_y = R_x / D` (Canzi's D = R_x/R_y), so the contact ellipse is longer across rolling.

# %%
E = e_prime(E_PA, NU, E_PA, NU)
moes_point = canzi_moes(alpha_bar=GRID_ALPHA_BAR, M_canzi=GRID_M2, line=False, D=GRID_D)   # Eq. (19), OPEN GL-13
point = moes_inverse(M=moes_point["M_moes"], L=moes_point["L_moes"], line=False, E_prime_Pa=E["E_prime_Pa"],
                     R_x_m=R_X_M, u_mean_m_per_s=U_MEAN_M_PER_S, roelands_z=ROELANDS_Z, roelands_p0_Pa=ROELANDS_P0_PA)
case = ContactCase("canzi_T1", r1=(R_X_M, R_X_M / GRID_D), r2=math.inf, E=E_PA, nu=NU, load=point["load_N"])
s_x = semi_axes(case)[0]
U_mean = groups(u_mean_m_per_s=U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                R_x_m=R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], load_N=point["load_N"])["U_mean"]
records = []
for n in RUN_N:
    # OPEN GL-37: dyadic h in Grid.around; per-N grid study needs exact h (hand-built Grid) or records n_x, n_y
    grid = Grid.around(case, h=(SPAN_X[0] + SPAN_X[1]) * s_x / (n - 1), span_x=SPAN_X, span_y=SPAN_Y)
    rec, p, gap = ehl.run(case, grid, RESULTS / NAME, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=RHO0_KG_PER_M3,
                          viscosity=("roelands", ROELANDS_Z, ROELANDS_P0_PA), density=DENSITY,
                          ue_m_per_s=U_MEAN_M_PER_S)
    H = {k: film_scalings(h_m=rec[f"{k}_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_moes"]
         for k in ("h_c", "h_min")}
    line_hc, line_hm = TABLE1.get(n, (None, None))
    # OPEN GL-37: Canzi's value at the same n only if the run really has n nodes per side (n_x, n_y in rec)
    records.append({**rec, "H_c": H["h_c"], "H_m": H["h_min"],
                    "H_c_canzi_same_n": float(at(line_hc, "H_c", f"grid={n}x{n}")["value"]) if line_hc else math.nan,
                    "H_m_canzi_same_n": float(at(line_hm, "H_m", f"grid={n}x{n}")["value"]) if line_hm else math.nan,
                    "err_H_c": rel_err(H["h_c"], REF_HC), "err_H_m": rel_err(H["h_min"], REF_HM)})

# %% [markdown]
# ## 3. Criteria
#
# C1: H_c and H_m on SlipPY's finest grid within 3 % of Canzi at 2048^2 **[proposed]**. C2: the sweep is
# recorded beside Canzi's own grids. The smoke grid is not graded.

# %%
finest = records[-1]
criteria = [{"id": "C1", "quantity": f"{key} finest grid [proposed]", "value": finest[key], "expected": ref,
             "err": finest[f"err_{key}"], "limit": "<= 0.03",
             "result": "SMOKE" if SMOKE else ("PASS" if abs(finest[f"err_{key}"]) <= 0.03 else "FAIL")}
            for key, ref in (("H_c", REF_HC), ("H_m", REF_HM))]
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
