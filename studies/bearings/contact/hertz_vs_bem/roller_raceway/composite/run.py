"""roller_raceway / composite: Hertz equivalent cylinder (elastic) against a flat, vs analytic line Hertz.

Line contact: the cylinder axis is the second array axis and the domain is periodic along it, as in
the SlipPy quasi-static notebook (``RoundSurface((R, inf, R))``, ``periodic_geometry=True``,
``periodic_axes=(False, True)``). The solution does not vary along the axis, so that axis has only
``NY`` nodes. The load of the case is per unit length [N/m]; the step carries ``load * length``.
Periodic geometry needs both surfaces discrete. What varies is the grid along x: ``N = 2**n``.
Run from anywhere: ``python run.py``. Results go to ``data/``.
"""
import math
import numpy as np
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))            # hertz_vs_bem/, to import _common

import slippy.contact as c
import slippy.surface as s

from _common import outputs, plots
from _common.case import CASES, EXTENT_IN_A, EXTENT_IN_A_BY_CASE, N_POWERS, N_POWERS_BY_CASE

BUILD = HERE.name                                   # "composite"
CASE = CASES[HERE.parent.name]                      # "roller_raceway"
NY = 16                                             # nodes along the cylinder axis (square cells)


def main() -> None:
    ref = c.hertz_full(**CASE.hertz_args())         # analytic reference (line contact), once per case

    # --- transcription Hertz -> SlipPy (composite: equivalent cylinder against a flat) ---------
    rx, ry = (float(r) for r in ref["relative_radii"])      # [m]; ry is infinite
    assert math.isinf(ry) and math.isfinite(rx), "line contact along y expected"
    semi_axes = (rx, math.inf, rx)                  # apex radii of curvature (rx, inf): cylinder along y
    assert CASE.moduli[0] == CASE.moduli[1] and CASE.v[0] == CASE.v[1], "composite needs one material"
    e_body, nu = CASE.moduli[0], CASE.v[0]          # both bodies are the same steel
    half_width = [r for r in ref["contact_radii"] if math.isfinite(r)][0]    # Hertz line half-width [m]
    extent_x = 2.0 * EXTENT_IN_A_BY_CASE.get(CASE.label, EXTENT_IN_A) * half_width   # [m]

    material = c.Elastic("equivalent", {"E": e_body, "v": nu})   # names must be unique per process
    rows = []
    for n_pow in N_POWERS_BY_CASE.get(CASE.label, N_POWERS):
        nx = 2**n_pow
        h = extent_x / nx
        length = NY * h                             # length of the periodic domain along the axis [m]
        x = (np.arange(nx) - nx / 2) * h                        # axis 0 = x (rolling direction)
        y = (np.arange(NY) - NY / 2) * h                        # axis 1 = cylinder axis
        X, Y = np.meshgrid(x, y, indexing="ij")

        analytic = s.RoundSurface(semi_axes)                    # analytic, no extent/shape: origin at the ball centre
        z = analytic.height(X, Y)                               # z = sqrt(rx^2 - x^2) - rx, cylinder curved in x
        round_s = s.Surface(profile=z, grid_spacing=h)          # discrete, as periodic_geometry requires
        flat_s = s.FlatSurface(shape=(nx, NY), grid_spacing=h, generate=True)
        round_s.material = material
        flat_s.material = material

        name = f"{CASE.label}_{BUILD}_N{nx}"
        slippy_dir = HERE / "data" / "slippy"
        slippy_dir.mkdir(parents=True, exist_ok=True)   # ContactModel uses os.mkdir (not makedirs)
        model = c.ContactModel(name, round_s, flat_s, output_dir=str(slippy_dir))
        load_total = CASE.load * length             # [N]
        model.add_step(c.StaticStep("load", normal_load=load_total, method="pk", tolerance=1e-9,
                                    periodic_geometry=True, periodic_axes=(False, True)))

        t0 = time.perf_counter()
        solution = model.solve()
        dt = time.perf_counter() - t0

        row = outputs.make_row(CASE, BUILD, nx, h, extent_x, ref, solution, dt, load_total=load_total)
        rows.append(row)
        pressure = outputs.pressure_from_loads(solution["loads_z"], h, float(solution["total_normal_load"]))
        plots.plot_run(HERE / "data" / "plots" / f"N{nx}.png", round_s, pressure, h, ref,
                       title=f"{CASE.label} / {BUILD} / N={nx}")
        print(f"N={nx:4d}  h={h:.3e}  nodes_in_2a={row['nodes_in_2b']:.1f}  "
              f"err a={row['err_a']:+.3%}  err p0={row['err_p0']:+.3%}  "
              f"err load={row['err_load']:+.2e}  t={dt:.2f} s")

    outputs.write_results(HERE / "data", CASE, BUILD, rows)


if __name__ == "__main__":
    main()
