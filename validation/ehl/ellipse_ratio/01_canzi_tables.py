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
# # Elliptical EHL contacts: SlipPY film versus D against Canzi2010 Tables 4-6
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today at `canzi_moes` (OPEN GL-13).
#
# **Case:** `validation/ehl/ellipse_ratio` (V-EHL-08), criteria R1-R3 of the README.
#
# At fixed alpha_bar and L, M2 is recomputed per D from M2c (Canzi Eq. 23). Each (alpha_bar, M2, D) is turned
# into an SI ellipsoid on a flat, `R_y = R_x / D`, solved, and its film put in Moes' scaling. The CELINE
# prediction (Eq. 24) is recorded beside it. `--smoke`: Table 5 at D = 1 only. The Hamrock-Dowson rows
# (V-EHL-10) have no script: mean or sum speed is not settled (R-092).

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from references.contact.ehl import canzi_moes, celine, e_prime, film_scalings, groups, moes_inverse
from slippy_bridge.contact import ContactCase, Grid
from slippy_bridge.contact.measures import rel_err
from slippy_bridge.lubrication import ehl

from construction import (D_VALUES, DENSITY, E_PA, NU, RESULTS, RHO0_KG_PER_M3, R_X_M, ROELANDS_P0_PA, ROELANDS_Z,
                          SPAN_X, SPAN_Y, TABLE4, TABLE5, TABLE45_LINES, TABLE6_CALC_LINES, TABLE6_RUN,
                          U_MEAN_M_PER_S)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs every point
NAME = "01_canzi_tables"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # the printed rows, copied unchanged

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


def printed(alpha_bar, M2c, D):
    """Canzi's values at (alpha_bar, M2c, D): Tables 4-5 where they exist, Table 6 calculated and CELINE."""
    k = D_VALUES.index(D)                         # rows of a block follow the D order of CSV L2-7
    tag = f"alpha_bar={alpha_bar:g};M2c={M2c:g};D={D:g}"
    out = {"H_Ccalc": float(at(TABLE6_CALC_LINES[(alpha_bar, M2c)] + k, "H_Ccalc", tag)["value"]),
           "H_c_celine": float(at(TABLE6_CALC_LINES[(alpha_bar, M2c)] + 6 + k, "H_c(CELINE)", tag)["value"])}
    for key, first in TABLE45_LINES.get((alpha_bar, M2c), {}).items():
        out[f"{key}_ref"] = float(at(first + k, key, tag)["value"])
    return out


RUN = [(TABLE5, 1.0)] if SMOKE else [(block, D) for block in TABLE6_RUN for D in D_VALUES]

# %% [markdown]
# ## 2. Solve each point
#
# (alpha_bar, M2c, D) -> (M2, L) by Eqs (19) and (23) with kappa and E_c, whose printed upper limit 2 pi is
# probably a misprint (`canzi_moes(D=...)`, OPEN GL-13). The body is the composite
# ellipsoid `(R_x, R_x / D)` on a flat, R_x along rolling (OPEN R-073: ellipse orientation).

# %%
E = e_prime(E_PA, NU, E_PA, NU)
records = []
for (alpha_bar, M2c), D in RUN:
    ref = printed(alpha_bar, M2c, D)
    moes_point = canzi_moes(alpha_bar=alpha_bar, M_canzi=M2c, line=False, D=D)   # Eqs (19), (23); OPEN GL-13
    point = moes_inverse(M=moes_point["M_moes"], L=moes_point["L_moes"], line=False, E_prime_Pa=E["E_prime_Pa"],
                         R_x_m=R_X_M, u_mean_m_per_s=U_MEAN_M_PER_S, roelands_z=ROELANDS_Z,
                         roelands_p0_Pa=ROELANDS_P0_PA)
    case = ContactCase(f"canzi_a{alpha_bar:g}_M{M2c:g}_D{D:g}", r1=(R_X_M, R_X_M / D), r2=math.inf, E=E_PA, nu=NU,
                       load=point["load_N"])
    grid = Grid.around(case, span_x=SPAN_X, span_y=SPAN_Y)                    # OPEN R-048 R-114: domain and h
    rec, p, gap = ehl.run(case, grid, RESULTS / NAME, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=RHO0_KG_PER_M3,
                          viscosity=("roelands", ROELANDS_Z, ROELANDS_P0_PA), density=DENSITY,
                          u_mean_m_per_s=U_MEAN_M_PER_S)
    U_mean = groups(u_mean_m_per_s=U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                    R_x_m=R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], load_N=point["load_N"])["U_mean"]
    H = {k: film_scalings(h_m=rec[f"{k}_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_moes"]
         for k in ("h_c", "h_min")}
    records.append({**rec, "alpha_bar": alpha_bar, "M2c": M2c, "D": D, "H_c": H["h_c"], "H_m": H["h_min"],
                    "H_c/H_m": H["h_c"] / H["h_min"], **ref, "err_H_c": rel_err(H["h_c"], ref["H_Ccalc"]),
                    "err_H_m": rel_err(H["h_min"], ref.get("H_m_ref", math.nan)),
                    "H_c_celine_ours": celine(M=moes_point["M_moes"], L=moes_point["L_moes"], D=D)["H_c"]})

# %% [markdown]
# ## 3. Criteria
#
# R1: H_c within 3 % of H_Ccalc (Table 6, which equals Tables 4-5); R2: H_m within 5 % of Tables 4-5 (H_m
# sits in the side lobes for D >= 0.2 and is grid-sensitive); R3: H_c/H_m and CELINE recorded only. All
# **[proposed]**. The smoke run is not graded.

# %%
criteria = []
for r in records:
    checks = [("R1", "H_c", 0.03)] + ([("R2", "H_m", 0.05)] if "H_m_ref" in r else [])
    for cid, key, lim in checks:
        criteria.append({"id": cid, "quantity": f"{key} {r['label']} [proposed]", "value": r[key],
                         "expected": r["H_Ccalc"] if key == "H_c" else r["H_m_ref"], "err": r[f"err_{key}"],
                         "limit": f"<= {lim}",
                         "result": "SMOKE" if SMOKE else ("PASS" if abs(r[f"err_{key}"]) <= lim else "FAIL")})
    if "H_c/H_m_ref" in r:
        criteria.append({"id": "R3", "quantity": f"H_c/H_m {r['label']} (record)", "value": r["H_c/H_m"],
                         "expected": r["H_c/H_m_ref"], "err": rel_err(r["H_c/H_m"], r["H_c/H_m_ref"]),
                         "limit": "record", "result": "RECORD"})
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
