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
# # Elliptical contact over the radius ratio: Hamrock & Brewe (1983) Tables 1-2, judge and BEM
#
# **Case:** `validation/contact/hertz_ellipse` (V-DRY-02), criteria E1-E5 of the README.
#
# Hamrock & Brewe print the exact Hertz solution of one contact, $R_x = 1$ cm, $F = 4.448$ N,
# $E' = 2.1972\times10^7$ N/cm$^2$, for radius ratios $\alpha = R_y/R_x$ (Table 1: $k$, $\mathcal{F}$,
# $\mathcal{E}$; Table 2: $D_y$, $D_x$, $\sigma_{max}$, $\delta$). The rows $\alpha = 5 \dots 10$ are used.
# The case is converted to SI in `construction.py` (`RATIO_CASES`): body 1 carries $(R_x, \alpha R_x)$,
# body 2 is flat, both bodies $E = E'(1-\nu^2)$, $\nu = 0.3$, so $E' = 2E^*$.
#
# 1. **Judge self-check** (E1, E2): `references.contact.hertz.hertz_point` against the printed exact
#    values. $k = D_y/D_x = s_y/s_x$ (diameters, $x$ = rolling); $\mathcal{F}$, $\mathcal{E}$ are the
#    complete elliptic integrals at the judge's eccentricity, $m = e^2 = 1 - 1/k^2$.
# 2. **BEM** (E3-E5): SlipPY static BEM through `slippy_bridge.contact` (composite build, default
#    grid) against the judge.
#
# `--smoke`: the BEM on the coarsest grid ($2h_0$).

# %%
import math
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv
from scipy.special import ellipe, ellipk

from references.contact.hertz import hertz_point
from slippy_bridge.contact import Grid, pair, static_step

from construction import ALPHAS, CM, N_PER_CM2, RATIO_CASES, RESULTS

SMOKE = "--smoke" in sys.argv                     # step-7 smoke run; a notebook runs the default grid
NAME = "02_ratio"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # the printed rows, copied unchanged

# %% [markdown]
# ## 1. The printed rows
#
# `value` is read as text (some rows hold text such as "1/32" in other cases) and converted for
# the rows used. Units of the table: cm, N/cm$^2$; converted with `construction.CM`,
# `construction.N_PER_CM2`. The half unit of the last printed digit is kept: Tables 2-3 print
# three significant digits or fewer (`D_x` at $\alpha = 10$ is `0.0087` cm, $\pm 0.57$ %).

# %%
rows = read_records_csv(REFERENCE, text_fields=("case_id", "kind", "quantity", "symbol", "value", "unit",
                                                "condition", "source_key", "location", "read_from", "note"))
TO_SI = {"-": 1.0, "cm": CM, "N/cm^2": N_PER_CM2}


def printed(alpha, location, symbol):
    """The exact value of ``symbol`` at ``alpha`` in Table ``location``, in SI, and its print resolution.

    Returns (value, half unit of the last printed digit relative to the value).
    """
    tag = f"alpha={alpha:.4f};"
    (r,) = [r for r in rows if r["kind"] == "reference" and r["location"] == location
            and r["symbol"] == symbol and r["condition"].startswith(tag)]
    mantissa, _, exponent = r["value"].lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    half_unit = 0.5 * 10.0 ** (int(exponent or 0) - decimals)
    return float(r["value"]) * TO_SI[r["unit"]], half_unit / abs(float(r["value"]))


# %% [markdown]
# ## 2. Judge self-check (E1, E2) and BEM (E3-E5), per $\alpha$

# %%
def row(cid, alpha, quantity, value, expected, limit_rel, resolution=math.nan, ok=None):
    """One criteria record: relative error against ``expected``, PASS within ``limit_rel``."""
    err = value / expected - 1.0
    ok = abs(err) <= limit_rel if ok is None else ok
    return {"id": cid, "alpha": alpha, "quantity": quantity, "value": value, "expected": expected, "err": err,
            "limit": limit_rel, "print_half_unit": resolution, "result": "PASS" if ok else "FAIL"}


criteria, bem = [], []
for alpha in ALPHAS:
    case = RATIO_CASES[alpha]
    judge = hertz_point(case.r1, case.r2, case.E, case.nu, case.load)
    k = judge["s_y_m"] / judge["s_x_m"]                             # D_y/D_x, major axis across rolling
    m = 1.0 - 1.0 / k**2                                            # e^2 of the ellipse, b/a = 1/k
    for symbol, value in (("k", k), ("F_int", float(ellipk(m))), ("E", float(ellipe(m)))):
        expected, res = printed(alpha, "Table 1", symbol)
        criteria.append(row("E1", alpha, symbol, value, expected, 1e-4, res))
    for symbol, value in (("D_y", 2.0 * judge["s_y_m"]), ("D_x", 2.0 * judge["s_x_m"]),
                          ("sigma_max", judge["p0_Pa"]), ("delta", judge["delta_m"])):
        expected, res = printed(alpha, "Table 2", symbol)
        # "~0.5 % (three printed digits)" made explicit: 0.5 %, or the half unit of the last
        # printed digit where that is larger (D_x = 0.0087 cm at alpha = 10: 0.57 %)
        criteria.append(row("E2", alpha, symbol, value, expected, max(0.005, res), res))

    h0 = Grid.around(case).h
    grid = Grid.around(case, h=2.0 * h0 if SMOKE else None)         # default grid (smoke: 2 h0)
    record, _ = static_step.run(case, pair(case, grid), grid, RESULTS / "slippy_02", name=case.label)
    bem.append({"alpha": alpha, **record})
    for q, key in (("s_x", "s_x_m"), ("s_y", "s_y_m"), ("p0", "p0_Pa")):
        criteria.append(row("E3", alpha, q, record[key], judge[key], 0.01))
    criteria.append(row("E4", alpha, "delta [proposed]", record["delta_m"], judge["delta_m"], 0.02))
    criteria.append(row("E5", alpha, "s_y / s_x (major axis on array axis 1)", record["s_y_m"] / record["s_x_m"],
                        k, math.nan, ok=record["s_y_m"] > record["s_x_m"]))

print(format_table((("id", "ID", "s"), ("alpha", "alpha", "g"), ("quantity", "quantity", "s"),
                    ("value", "value [SI]", ".6g"), ("expected", "expected [SI]", ".6g"),
                    ("err", "err [%]", "+.4f", 100.0), ("limit", "limit [%]", "g", 100.0),
                    ("print_half_unit", "print +- [%]", ".3f", 100.0), ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", bem)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
