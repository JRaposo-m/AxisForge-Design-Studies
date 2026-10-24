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
# # EHL line contact: SlipPY against Venner & Bos (1994) and Canzi2010 Table 2
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today at `moes_inverse` (OPEN R-091 R-114 GL-18).
#
# **Case:** `validation/ehl/line_contact` (V-EHL-03, V-EHL-04, V-EHL-06), criteria N1-N3 of the README.
#
# A line contact in SlipPY is a 2-D grid periodic along the roller axis (array axis 1, R-084): the load per
# unit length is applied over the periodic strip by `ehl.run`. The film is put in Venner's scaling
# `H = h R / b^2` (Venner) or Moes' (Canzi). `--smoke`: M = 20, incompressible only.

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from references.contact.ehl import canzi_moes, e_prime, film_scalings, groups, moes_inverse
from slippy_bridge.contact import ContactCase, Grid, semi_axes
from slippy_bridge.contact.measures import rel_err
from slippy_bridge.lubrication import ehl

from construction import (CANZI_ALPHA_BAR, CANZI_DENSITY, CANZI_M1, CANZI_SPAN_X, DENSITIES, E_PA, L_MOES, M_HIGH,
                          M_LOW, N_X_RUN, NU, RESULTS, RHO0_KG_PER_M3, R_X_M, ROELANDS_P0_PA, ROELANDS_Z, SPAN_X_HIGH,
                          SPAN_X_LOW, U_MEAN_M_PER_S)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs every point
NAME = "01_film"
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


# (M, density) -> (span_x, CSV line of H_m, CSV line of H_c) at Venner's finest level
POINTS = {(M_LOW, "incompressible"): (SPAN_X_LOW, 44, 45), (M_LOW, "dowson_higginson"): (SPAN_X_LOW, 46, 47),
          (M_HIGH, "incompressible"): (SPAN_X_HIGH, 127, 128), (M_HIGH, "dowson_higginson"): (SPAN_X_HIGH, 129, 130)}
TAG = {"incompressible": "density=incompressible", "dowson_higginson": "density=D-H"}
RUN = [(M_LOW, "incompressible")] if SMOKE else [(M, d) for M in (M_LOW, M_HIGH) for d in DENSITIES]
E = e_prime(E_PA, NU, E_PA, NU)


def solve(label, M, L, span_x, density):
    """The SI line contact of the Moes point (M, L), solved; returns (record, U_mean)."""
    point = moes_inverse(M=M, L=L, line=True, E_prime_Pa=E["E_prime_Pa"], R_x_m=R_X_M, u_mean_m_per_s=U_MEAN_M_PER_S,
                         roelands_z=ROELANDS_Z, roelands_p0_Pa=ROELANDS_P0_PA)
    # cylinder (radius R_x, straight along y) on a flat: load per unit length [N/m]
    case = ContactCase(label, r1=(R_X_M, math.inf), r2=math.inf, E=E_PA, nu=NU, load=point["w_N_per_m"], line=True)
    s_x = semi_axes(case)[0]
    # OPEN GL-37: dyadic h in Grid.around; per-N grid study needs exact h (hand-built Grid) or records n_x, n_y
    grid = Grid.around(case, h=(span_x[0] + span_x[1]) * s_x / (N_X_RUN - 1), span_x=span_x)
    rec, p, gap = ehl.run(case, grid, RESULTS / NAME, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=RHO0_KG_PER_M3,
                          viscosity=("roelands", ROELANDS_Z, ROELANDS_P0_PA), density=density,
                          u_mean_m_per_s=U_MEAN_M_PER_S)
    U_mean = groups(u_mean_m_per_s=U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                    R_x_m=R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], w_N_per_m=point["w_N_per_m"])["U_mean"]
    return rec, U_mean


# %% [markdown]
# ## 2. Venner & Bos, M = 20 and M = 200

# %%
records = []
for M, density in RUN:
    span_x, line_hm, line_hc = POINTS[(M, density)]
    rec, U_mean = solve(f"vb_line_M{M:g}_{density}", M, L_MOES, span_x, density)
    H = {k: film_scalings(h_m=rec[f"{k}_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_venner"]
         for k in ("h_c", "h_min")}
    ref_hm, ref_hc = (float(at(n, s, TAG[density])["value"]) for n, s in ((line_hm, "H_m"), (line_hc, "H_c")))
    records.append({**rec, "source": "VennerBos1994", "M": M, "density": density, "H_c": H["h_c"], "H_m": H["h_min"],
                    "H_c_ref": ref_hc, "H_m_ref": ref_hm, "err_H_c": rel_err(H["h_c"], ref_hc),
                    "err_H_m": rel_err(H["h_min"], ref_hm)})

# %% [markdown]
# ## 3. Canzi2010 Table 2, alpha_bar = 20, M1 = 17.97
#
# (alpha_bar, M1) -> (M, L) by Canzi Eq. (3) (`canzi_moes`, OPEN GL-13). Compressible (D-H). The formula
# value of L166 is the judge of `references.contact.ehl.moes` (tests), not of SlipPY.

# %%
if not SMOKE:
    ref_calc = float(at(165, "H_c", "alpha_bar=20;M1=17.97;col=calculated")["value"])
    moes_point = canzi_moes(alpha_bar=CANZI_ALPHA_BAR, M_canzi=CANZI_M1, line=True, D=None)
    rec, U_mean = solve("canzi_T2_a20_M17.97", moes_point["M_moes"], moes_point["L_moes"], CANZI_SPAN_X, CANZI_DENSITY)
    H_c = film_scalings(h_m=rec["h_c_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_moes"]
    records.append({**rec, "source": "Canzi2010", "H_c": H_c, "H_c_ref": ref_calc, "err_H_c": rel_err(H_c, ref_calc)})

# %% [markdown]
# ## 4. Criteria
#
# N1 (M = 20): H_c within 2 %, H_m within 3 %; N2 (M = 200): H_c within 2 %, H_m within 5 %; N3 (Canzi):
# H_c within 3 %; all **[proposed]**. The smoke run is not graded.

# %%
LIMITS = {M_LOW: ("N1", 0.02, 0.03), M_HIGH: ("N2", 0.02, 0.05)}
criteria = []
for r in records:
    checks = ([("N3", "H_c", 0.03)] if r["source"] == "Canzi2010"
              else [(LIMITS[r["M"]][0], "H_c", LIMITS[r["M"]][1]), (LIMITS[r["M"]][0], "H_m", LIMITS[r["M"]][2])])
    for cid, key, lim in checks:
        criteria.append({"id": cid, "quantity": f"{key} {r['label']} [proposed]", "value": r[key],
                         "expected": r[f"{key}_ref"], "err": r[f"err_{key}"], "limit": f"<= {lim}",
                         "result": "SMOKE" if SMOKE else ("PASS" if abs(r[f"err_{key}"]) <= lim else "FAIL")})
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
