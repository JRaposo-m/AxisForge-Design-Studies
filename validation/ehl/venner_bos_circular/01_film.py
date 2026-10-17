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
# # Circular EHL contact: SlipPY against Venner & Bos (1994), M = 50 and M = 1000
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated. Raises `NotImplementedError` today at `moes_inverse` (OPEN R-091 R-114 GL-18).
#
# **Case:** `validation/ehl/venner_bos_circular` (V-EHL-01, V-EHL-02, V-EHL-13), criteria E1-E3 of the README.
#
# Each (M, L) point of the tables is turned into an SI contact (`moes_inverse`), solved by `ehl.run`, and
# its film put back in Venner's scaling `H = h R_x / a^2` (`film_scalings`). `--smoke`: M = 50,
# incompressible, 65x65 only.

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from references.contact.ehl import e_prime, film_scalings, groups, moes_inverse
from slippy_bridge.contact import ContactCase, Grid, semi_axes
from slippy_bridge.contact.outputs import rel_err
from slippy_bridge.lubrication import ehl

from construction import (DENSITIES, E_PA, L_MOES, M_HIGH, M_LOW, N_FINEST, N_SMOKE, NU, RESULTS, RHO0_KG_PER_M3,
                          R_X_M, ROELANDS_P0_PA, ROELANDS_Z, SPAN_X_HIGH, SPAN_X_LOW, SPAN_Y_HIGH, SPAN_Y_LOW,
                          U_MEAN_M_PER_S)

SMOKE = "--smoke" in sys.argv                     # smoke run; a notebook runs the full grids
NAME = "01_film"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # the printed rows, copied unchanged

# %% [markdown]
# ## 1. The printed rows
#
# Rows are taken by their CSV line (header = line 1) and checked against their symbol and condition.
# The symbol column is never printed (non-ASCII, GL-23).

# %%
rows = read_records_csv(REFERENCE, text_fields=("case_id", "kind", "quantity", "symbol", "value", "unit",
                                                "condition", "source_key", "location", "read_from", "note"))


def at(line, symbol, condition):
    """The CSV row at ``line``, checked to hold ``symbol`` under ``condition`` (a substring)."""
    r = rows[line - 2]
    if r["symbol"] != symbol or condition not in r["condition"]:
        raise ValueError(f"CSV L{line} is not {condition}: the copied file changed")
    return r


# (M, density) -> (span_x, span_y, CSV line of H_m, CSV line of H_c); H_m precedes H_c in the tables
POINTS = {(M_LOW, "incompressible"): (SPAN_X_LOW, SPAN_Y_LOW, 26, 27),
          (M_LOW, "dowson_higginson"): (SPAN_X_LOW, SPAN_Y_LOW, 28, 29),
          (M_HIGH, "incompressible"): (SPAN_X_HIGH, SPAN_Y_HIGH, 69, 70),
          (M_HIGH, "dowson_higginson"): (SPAN_X_HIGH, SPAN_Y_HIGH, 71, 72)}
TAG = {"incompressible": "density=incompressible", "dowson_higginson": "density=D-H"}
RUN = [(M_LOW, "incompressible")] if SMOKE else [(M, d) for M in (M_LOW, M_HIGH) for d in DENSITIES]
N = N_SMOKE if SMOKE else N_FINEST                # nodes per side asked [-]; the record keeps the n_x, n_y obtained

# %% [markdown]
# ## 2. Solve each point
#
# The SI point: `moes_inverse` fixes eta0 and the load from (M, L) for given E', R_x, u_mean, z and p0 (OPEN GL-34).
# The domain is given in units of the Hertz radius `a` with its (inlet, outlet) span (OPEN R-048).

# %%
E = e_prime(E_PA, NU, E_PA, NU)
records = []
for M, density in RUN:
    span_x, span_y, line_hm, line_hc = POINTS[(M, density)]
    point = moes_inverse(M=M, L=L_MOES, line=False, E_prime_Pa=E["E_prime_Pa"], R_x_m=R_X_M,
                         u_mean_m_per_s=U_MEAN_M_PER_S, roelands_z=ROELANDS_Z, roelands_p0_Pa=ROELANDS_P0_PA)
    # ball (radius R_x) on a flat: the composite body has R_x = R_y = the ball radius
    case = ContactCase(f"vb_M{M:g}_{density}", r1=R_X_M, r2=math.inf, E=E_PA, nu=NU, load=point["load_N"])
    s_x = semi_axes(case)[0]
    # OPEN GL-37: dyadic h in Grid.around; per-N grid study needs exact h (hand-built Grid) or records n_x, n_y
    grid = Grid.around(case, h=(span_x[0] + span_x[1]) * s_x / (N - 1), span_x=span_x, span_y=span_y)
    rec, p, gap = ehl.run(case, grid, RESULTS / NAME, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=RHO0_KG_PER_M3,
                          viscosity=("roelands", ROELANDS_Z, ROELANDS_P0_PA), density=density,
                          u_mean_m_per_s=U_MEAN_M_PER_S)
    U_mean = groups(u_mean_m_per_s=U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                    R_x_m=R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], load_N=point["load_N"])["U_mean"]
    H = {k: film_scalings(h_m=rec[f"{k}_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_venner"]
         for k in ("h_c", "h_min")}
    ref_hm = float(at(line_hm, "H_m", TAG[density])["value"])
    ref_hc = float(at(line_hc, "H_c", TAG[density])["value"])
    records.append({**rec, "M": M, "density": density, "H_c": H["h_c"], "H_m": H["h_min"],
                    "H_c_ref": ref_hc, "H_m_ref": ref_hm, "err_H_c": rel_err(H["h_c"], ref_hc),
                    "err_H_m": rel_err(H["h_min"], ref_hm)})

# %% [markdown]
# ## 3. Criteria
#
# E1 (M = 50): H_c within 2 %, H_m within 3 %; E2 (M = 1000): H_c within 2 %, H_m within 5 %; all
# **[proposed]**. E3, the D-H / incompressible ratio, is recorded only. The smoke grid is not graded.

# %%
LIMITS = {M_LOW: ("E1", 0.02, 0.03), M_HIGH: ("E2", 0.02, 0.05)}
criteria = []
for r in records:
    cid, lim_c, lim_m = LIMITS[r["M"]]
    for key, lim in (("H_c", lim_c), ("H_m", lim_m)):
        ok = not SMOKE and abs(r[f"err_{key}"]) <= lim
        criteria.append({"id": cid, "quantity": f"{key} M={r['M']:g} {r['density']} [proposed]",
                         "value": r[key], "expected": r[f"{key}_ref"], "err": r[f"err_{key}"],
                         "limit": f"<= {lim}", "result": "SMOKE" if SMOKE else ("PASS" if ok else "FAIL")})
by = {(r["M"], r["density"]): r for r in records}
for M in (M_LOW, M_HIGH):
    if (M, "incompressible") in by and (M, "dowson_higginson") in by:
        for key in ("H_c", "H_m"):
            ratio = by[(M, "dowson_higginson")][key] / by[(M, "incompressible")][key]
            ratio_ref = by[(M, "dowson_higginson")][f"{key}_ref"] / by[(M, "incompressible")][f"{key}_ref"]
            criteria.append({"id": "E3", "quantity": f"{key} D-H/incompressible M={M:g} (record)", "value": ratio,
                             "expected": ratio_ref, "err": rel_err(ratio, ratio_ref), "limit": "record",
                             "result": "RECORD"})

print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".4g"),
                    ("expected", "expected", ".4g"), ("err", "err [%]", "+.3f", 100.0), ("limit", "limit", "s"),
                    ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", records)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
