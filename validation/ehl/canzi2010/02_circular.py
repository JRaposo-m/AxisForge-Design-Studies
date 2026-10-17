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
# # Circular EHL contacts: SlipPY against Canzi2010 Table 3
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today at `canzi_moes` (OPEN GL-13).
#
# **Case:** `validation/ehl/canzi2010` (V-EHL-07), criterion C3 of the README.
#
# Six points alpha_bar in {20, 10, 5} x M2 in {100, 500}, D = 1, compressible. The "Hc calculated" column is
# the reference; the "Moes formula" column is recorded beside `references.contact.ehl.moes`. `--smoke`: the
# first point only.

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from references.contact.ehl import canzi_moes, e_prime, film_scalings, groups, moes, moes_inverse
from slippy_bridge.contact import ContactCase, Grid, hertz
from slippy_bridge.contact.outputs import rel_err
from slippy_bridge.lubrication import ehl

from construction import (CIRC_ALPHA_BAR, CIRC_CALCULATED_LINES, CIRC_M2, DENSITY, E_PA, NU, RESULTS, RHO0_KG_PER_M3,
                          R_X_M, ROELANDS_P0_PA, ROELANDS_Z, SPAN_X, SPAN_Y, U_MEAN_M_PER_S)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs all six points
NAME = "02_circular"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values_circular.csv"    # Table 3 rows, copied unchanged (GL-22)

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


POINTS = [(a, m) for a in CIRC_ALPHA_BAR for m in CIRC_M2]
RUN = POINTS[:1] if SMOKE else POINTS

# %% [markdown]
# ## 2. Solve each point
#
# (alpha_bar, M2) -> (M, L) by Canzi Eq. (14) (`canzi_moes`, OPEN GL-13); then the SI point by `moes_inverse`.
# The formula column needs the dimensional Hertz pressure p_h (Eq. 11; OPEN GL-12 GL-13).

# %%
E = e_prime(E_PA, NU, E_PA, NU)
records = []
for alpha_bar, M2 in RUN:
    line = CIRC_CALCULATED_LINES[(alpha_bar, M2)]
    tag = f"alpha_bar={alpha_bar:g};M2={M2:g}"
    ref_calc = float(at(line, "H_c", f"{tag};col=calculated")["value"])
    ref_formula = float(at(line + 1, "H_c", f"{tag};col=Moes formula")["value"])
    moes_point = canzi_moes(alpha_bar=alpha_bar, M_canzi=M2, line=False, D=None)      # circular, Eq. (14)
    point = moes_inverse(M=moes_point["M_moes"], L=moes_point["L_moes"], line=False, E_prime_Pa=E["E_prime_Pa"],
                         R_x_m=R_X_M, u_mean_m_per_s=U_MEAN_M_PER_S, roelands_z=ROELANDS_Z,
                         roelands_p0_Pa=ROELANDS_P0_PA)
    case = ContactCase(f"canzi_T3_a{alpha_bar:g}_M{M2:g}", r1=R_X_M, r2=math.inf, E=E_PA, nu=NU,
                       load=point["load_N"])
    grid = Grid.around(case, span_x=SPAN_X, span_y=SPAN_Y)                    # OPEN R-048 R-114: domain and h
    rec, p, gap = ehl.run(case, grid, RESULTS / NAME, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=RHO0_KG_PER_M3,
                          viscosity=("roelands", ROELANDS_Z, ROELANDS_P0_PA), density=DENSITY,
                          u_mean_m_per_s=U_MEAN_M_PER_S)
    U_mean = groups(u_mean_m_per_s=U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                    R_x_m=R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], load_N=point["load_N"])["U_mean"]
    H_c = film_scalings(h_m=rec["h_c_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_moes"]
    H_formula = moes(M=moes_point["M_moes"], L=moes_point["L_moes"], line=False, compressible=True,
                     p_h_Pa=hertz(case)["p0_Pa"])["H_c"]
    records.append({**rec, "alpha_bar": alpha_bar, "M2": M2, "H_c": H_c, "H_c_ref": ref_calc,
                    "err_H_c": rel_err(H_c, ref_calc), "H_c_formula": H_formula, "H_c_formula_printed": ref_formula})

# %% [markdown]
# ## 3. Criteria
#
# C3: H_c within 3 % of "Hc calculated" **[proposed]**, per point. The formula column is recorded only.

# %%
criteria = [{"id": "C3", "quantity": f"H_c alpha_bar={r['alpha_bar']:g} M2={r['M2']:g} [proposed]",
             "value": r["H_c"], "expected": r["H_c_ref"], "err": r["err_H_c"], "limit": "<= 0.03",
             "result": "SMOKE" if SMOKE else ("PASS" if abs(r["err_H_c"]) <= 0.03 else "FAIL")} for r in records]
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
