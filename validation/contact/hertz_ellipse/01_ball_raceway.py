"""01: ball on the inner raceway of a deep groove ball bearing, SlipPY static BEM against Hertz.

Elliptical point contact, long across rolling (the groove conforms to the ball). The sweep runs on
the hand-written case (``construction.CASE``, SI); the AxisForge bearing of ``construction.py`` is
then passed through the bridge, checked to be the same contact (by value: its ``case_hash`` may
differ in the last bit after the mm -> m conversion) and solved once on the default grid.
Output: ``results/`` of this case (git-ignored).

Run: ``python 01_ball_raceway.py [--build composite two_profiles]``.
"""
import argparse

import numpy as np
from axisforge.outputs._format import format_table, write_records_csv

from slippy_bridge.contact import Grid, contacts, pair, static_step

from validation._support.convergence import RECORD_COLUMNS, sweep
from validation._support.figures import contact_plots

from construction import CASE, H_LEVELS, RESULTS, SPAN_X_0, SPAN_X_LEVELS, ball_bearing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", nargs="+", default=["composite", "two_profiles"],
                        choices=["composite", "two_profiles"])
    builds = parser.parse_args().build

    records = sweep(CASE, builds, H_LEVELS, SPAN_X_LEVELS, RESULTS / "slippy", span_x_0=SPAN_X_0)

    # AxisForge link: the bearing (mm, N, MPa) through the bridge must give the case contact (SI).
    # The mm -> m conversion may change the last bit, so values are compared, not the case hash.
    link = contacts(ball_bearing(), load=CASE.load, races=("inner",))[0]
    c = link.case
    np.testing.assert_allclose(c.r1 + c.r2 + c.E + c.nu + (c.load,),
                               CASE.r1 + CASE.r2 + CASE.E + CASE.nu + (CASE.load,), rtol=1e-12, atol=0.0)
    grid = Grid.around(c, span_x=SPAN_X_0)                   # default grid
    record, p = static_step.run(link, pair(c, grid), grid, RESULTS / "slippy", name=f"{CASE.label}_axisforge")
    # written beside the sweep, but kept out of the convergence figure
    table = records + [{"build": "composite", "sweep": "axisforge", "span_x": SPAN_X_0, **record}]

    write_records_csv(RESULTS / "results.csv", table)
    contact_plots.convergence(records, RESULTS / "convergence.png")
    contact_plots.pressure_cuts(p, grid, c, RESULTS / "pressure_cuts.png")

    print(f"{CASE.label}   case_hash {CASE.case_hash()}")
    print(format_table(RECORD_COLUMNS, table))
    print()
    print(f"AxisForge link: PASS  {link.subtype} {link.location['race']} race gives the case contact "
          f"within 1e-12 (case_hash {c.case_hash()})")


if __name__ == "__main__":
    main()
