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
# # FZG gears C40 and H501: AxisForge geometry against the FEUP tables
#
# **PROVISIONAL (scaffold 2026-10-17)**: structure agreed in debate_gears-lubrication-scaffold_2026-10-17;
# not validated.
#
# **Case:** `validation/gears/fzg_geometry_ph` (V-GEAR-01), criteria G1-G3 of the README.
#
# The pairs are built from the printed z, m, alpha, beta, b and x (`construction.build_pair`); AxisForge
# derives the working centre distance, the tip diameters and the contact ratios, which are compared with
# the printed ones. Both printed wheel tip diameters of C40 are reported (L12 115.54, L110 118.54; R-121).
# The contact ratios are recorded, not graded: AxisForge gives 1.4624 against the printed 1.44 (GL-10).
# Only AxisForge runs here (no SlipPY); `--smoke` changes nothing.

# %%
import sys

from axisforge.outputs._format import format_table, read_records_csv, write_records_csv

from construction import RESULTS, build_pair

SMOKE = "--smoke" in sys.argv                     # the geometry is cheap: the smoke run is the full run
NAME = "01_geometry"
RESULTS.mkdir(exist_ok=True)                      # git-ignored (D-0020)
REFERENCE = RESULTS.parent / "reference_values.csv"             # the printed rows, copied unchanged

# %% [markdown]
# ## 1. The printed rows
#
# Rows are taken by their CSV line (header = line 1) and checked by quantity and condition; the symbol
# column is never printed (non-ASCII, GL-23).

# %%
rows = read_records_csv(REFERENCE, text_fields=("case_id", "kind", "quantity", "symbol", "value", "unit",
                                                "condition", "source_key", "location", "read_from", "note"))


def at(line, quantity, condition):
    """The printed value at CSV ``line`` (float), checked to be ``quantity`` under ``condition``."""
    r = rows[line - 2]
    if quantity not in r["quantity"] or condition not in r["condition"]:
        raise ValueError(f"CSV L{line} is not {quantity} / {condition}: the copied file changed")
    return float(r["value"])


# %% [markdown]
# ## 2. AxisForge against the tables
#
# G1: tip diameters within 0.01 mm (two printed decimals) **[proposed]**; G2: working centre distance within
# 0.01 mm of 91.5 **[proposed]**; G3: contact ratios recorded (known gap GL-10; the tests hold it as a
# known-gap xfail).

# %%
C40, H501 = build_pair("C40"), build_pair("H501")
TA, TD = "tip (addendum) diameter", "contact ratio"


def row(cid, quantity, value, line, kind, condition, limit):
    """One comparison record: AxisForge ``value`` against CSV ``line``; ``limit`` None means record only."""
    expected = at(line, kind, condition)
    diff = value - expected
    result = "RECORD" if limit is None else ("PASS" if abs(diff) <= limit else "FAIL")
    return {"id": cid, "quantity": quantity, "csv_line": line, "value": value, "expected": expected, "diff": diff,
            "limit": "record" if limit is None else f"<= {limit}", "result": result}


criteria = [
    row("G1", "C40 d_a pinion [mm] (F2015b)", C40.gear1.da, 11, TA, "gear=C40;member=pinion", 0.01),
    row("G1", "C40 d_a pinion [mm] (M2015)", C40.gear1.da, 109, TA, "gear=C40;member=pinion", 0.01),
    row("G1", "C40 d_a wheel [mm] (M2015 118.54)", C40.gear2.da, 110, TA, "gear=C40;member=wheel", 0.01),
    row("G1", "C40 d_a wheel [mm] (F2015b 115.54, R-121)", C40.gear2.da, 12, TA, "gear=C40;member=gear", None),
    row("G2", "C40 a_w [mm]", C40.al, 5, "centre distance", "gear=C40", 0.01),
    row("G3", "C40 eps_alpha (GL-10)", C40.epslon_alpha, 13, "transverse " + TD, "gear=C40", None),
    row("G3", "C40 eps_alpha (M2015, GL-10)", C40.epslon_alpha, 111, "transverse " + TD, "gear=C40", None),
    row("G3", "C40 eps_beta", C40.epslon_beta, 14, "overlap " + TD, "gear=C40", None),
    row("G3", "C40 eps_gamma", C40.epslon_gamma, 113, "total " + TD, "gear=C40", None),
    # H501: recorded only (GL-10: d_a 80.436/116.627 vs 80.37/116.57; eps_alpha 1.474 vs 1.45)
    row("G3", "H501 d_a pinion [mm] (GL-10)", H501.gear1.da, 25, TA, "gear=H501;member=pinion", None),
    row("G3", "H501 d_a wheel [mm] (GL-10)", H501.gear2.da, 26, TA, "gear=H501;member=gear", None),
    row("G3", "H501 a_w [mm]", H501.al, 19, "centre distance", "gear=H501", None),
    row("G3", "H501 eps_alpha (GL-10)", H501.epslon_alpha, 27, "transverse " + TD, "gear=H501", None),
    row("G3", "H501 eps_beta", H501.epslon_beta, 28, "overlap " + TD, "gear=H501", None),
    row("G3", "H501 eps_gamma", H501.epslon_gamma, 129, "total " + TD, "gear=H501", None),
]
print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("csv_line", "CSV L", ".0f"),
                    ("value", "AxisForge", ".5g"), ("expected", "printed", ".5g"), ("diff", "diff", "+.4f"),
                    ("limit", "limit", "s"), ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)
