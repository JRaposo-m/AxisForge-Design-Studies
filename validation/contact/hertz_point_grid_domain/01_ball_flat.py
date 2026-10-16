"""01: steel ball on a steel flat, SlipPY static BEM against Hertz (circular point contact).

Body 2 is flat, so the composite and the two_profiles builds differ only in the ball profile
(paraboloid-like equivalent body of the same radius: the same sphere). Both are run as a check
that ``pair`` is consistent. Output: ``results/`` of this case (git-ignored).

Run: ``python 01_ball_flat.py [--build composite two_profiles]``.
"""
import argparse

from axisforge.outputs._format import format_table, write_records_csv

from slippy_bridge.contact import Grid, pair, static_step

from validation._support.convergence import RECORD_COLUMNS, sweep
from validation._support.figures import contact_plots

from construction import CASE, H_LEVELS, RESULTS, SPAN_X_0, SPAN_X_LEVELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", nargs="+", default=["composite", "two_profiles"],
                        choices=["composite", "two_profiles"])
    builds = parser.parse_args().build

    records = sweep(CASE, builds, H_LEVELS, SPAN_X_LEVELS, RESULTS / "slippy", span_x_0=SPAN_X_0)
    write_records_csv(RESULTS / "results.csv", records)
    contact_plots.convergence(records, RESULTS / "convergence.png")

    grid = Grid.around(CASE, span_x=SPAN_X_0)                # default grid, for the pressure cuts
    _, p = static_step.run(CASE, pair(CASE, grid), grid, RESULTS / "slippy", name=f"{CASE.label}_cuts")
    contact_plots.pressure_cuts(p, grid, CASE, RESULTS / "pressure_cuts.png")

    print(f"{CASE.label}   case_hash {CASE.case_hash()}")
    print(format_table(RECORD_COLUMNS, records))


if __name__ == "__main__":
    main()
