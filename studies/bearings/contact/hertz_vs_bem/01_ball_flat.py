"""01: steel ball on a steel flat, SlipPY static BEM against Hertz (circular point contact).

Body 2 is flat, so the composite and the two_profiles builds differ only in the ball profile
(paraboloid-like equivalent body of the same radius: the same sphere). Both are run as a check
that ``pair`` is consistent. Output: ``results/ball_flat/`` next to this script (git-ignored).

Run: ``python 01_ball_flat.py [--build composite two_profiles]``.
"""
import argparse

from axisforge.outputs._format import format_table, write_records_csv

from slippy_bridge.contact import Grid, pair, static_step

from _common import plots
from _common.case import CASES, H_LEVELS, RESULTS, SPAN_X_LEVELS
from _common.sweep import CRITERIA_COLUMNS, RECORD_COLUMNS, criteria, sweep

CASE = CASES["ball_flat"]
OUT = RESULTS / CASE.label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", nargs="+", default=["composite", "two_profiles"],
                        choices=["composite", "two_profiles"])
    builds = parser.parse_args().build

    records = sweep(CASE, builds, H_LEVELS, SPAN_X_LEVELS, OUT / "slippy")
    write_records_csv(OUT / "results.csv", records)
    plots.convergence(records, OUT / "convergence.png")

    grid = Grid.around(CASE)                                 # default grid, for the pressure cuts
    _, p = static_step.run(CASE, pair(CASE, grid), grid, OUT / "slippy", name=f"{CASE.label}_cuts")
    plots.pressure_cuts(p, grid, CASE, OUT / "pressure_cuts.png")

    print(f"{CASE.label}   case_hash {CASE.case_hash()}")
    print(format_table(RECORD_COLUMNS, records))
    print()
    print(format_table(CRITERIA_COLUMNS, criteria(records)))


if __name__ == "__main__":
    main()
