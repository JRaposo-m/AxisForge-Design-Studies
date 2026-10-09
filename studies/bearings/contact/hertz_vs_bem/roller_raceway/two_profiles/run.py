"""roller_raceway / two_profiles: both real bodies (roller and raceway), each with its own geometry and material, vs analytic line Hertz.

Line contact: each body is a discrete array of the analytic ``RoundSurface`` height evaluated on the same
grid (periodic geometry needs two discrete surfaces). A cylinder with radius R along x is
``RoundSurface((R, inf, R))`` (semi-axes; apex radii a**2/c = R in x, b**2/c = inf in y). The cylinder axis is
the second array axis and the domain is periodic along it, as in the SlipPy quasi-static notebook
(``periodic_geometry=True``, ``periodic_axes=(False, True)``). The solution does not vary along the axis, so
that axis has only ``NY`` nodes. The load of the case is per unit length [N/m]; the step carries
``load * length``. What varies is the grid along x: ``N = 2**n``.
Run from anywhere: ``python run.py``. Results go to ``data/``.
"""
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))            # hertz_vs_bem/, to import _common

import numpy as np

import slippy.contact as c
import slippy.surface as s

from _common import outputs, plots
from _common.case import CASES, EXTENT_IN_A, EXTENT_IN_A_BY_CASE, N_POWERS, N_POWERS_BY_CASE

BUILD = HERE.name                                   # "two_profiles"
CASE = CASES[HERE.parent.name]                      # "roller_raceway"
NY = 16                                             # nodes along the cylinder axis (square cells)


def main() -> None:
    ref = c.hertz_full(**CASE.hertz_args())         # analytic reference (line contact), once per case

    assert CASE.angle == 0.0 and all(math.isinf(r[1]) for r in (CASE.r1, CASE.r2)), "line contact along y expected"
    half_width = [r for r in ref["contact_radii"] if math.isfinite(r)][0]    # Hertz line half-width [m]
    extent_x = 2.0 * EXTENT_IN_A_BY_CASE.get(CASE.label, EXTENT_IN_A) * half_width   # [m]

    # --- transcription Hertz -> SlipPy (each body on its own: cylinder of radius |r_x|) --------
    # RoundSurface height is convex (apex 0, <= 0 elsewhere); a concave body (r_x < 0) is its mirror image.
    r_1, r_2 = CASE.r1[0], CASE.r2[0]
    for r in (r_1, r_2):                            # SlipPy returns a flat (silently) outside |x| < |r|
        assert extent_x / 2.0 < abs(r), f"domain half-width {extent_x / 2.0:.4g} m reaches outside the cylinder |r|={abs(r):.4g} m"

    body_1 = c.Elastic("body_1", {"E": CASE.moduli[0], "v": CASE.v[0]})   # unique names per process
    body_2 = c.Elastic("body_2", {"E": CASE.moduli[1], "v": CASE.v[1]})
    rows = []
    for n_pow in N_POWERS_BY_CASE.get(CASE.label, N_POWERS):
        nx = 2**n_pow
        h = extent_x / nx
        length = NY * h                             # length of the periodic domain along the axis [m]
        x = (np.arange(nx) - nx / 2) * h            # nodes at i*h, apex at extent/2 -> node nx/2
        y = (np.arange(NY) - NY / 2) * h
        X, Y = np.meshgrid(x, y, indexing="ij")     # axis 0 = x (rolling), axis 1 = y (cylinder axis)
        roller_s  = s.Surface(profile=s.RoundSurface((r_1, math.inf, r_1)).height(X, Y), grid_spacing=h)
        raceway_s = s.Surface(profile=s.RoundSurface((r_2, math.inf, r_2)).height(X, Y), grid_spacing=h)
        roller_s.material = body_1
        raceway_s.material = body_2

        name = f"{CASE.label}_{BUILD}_N{nx}"
        slippy_dir = HERE / "data" / "slippy"
        slippy_dir.mkdir(parents=True, exist_ok=True)   # ContactModel uses os.mkdir (not makedirs)
        model = c.ContactModel(name, roller_s, raceway_s, output_dir=str(slippy_dir))
        load_total = CASE.load * length             # [N]
        model.add_step(c.StaticStep("load", normal_load=load_total, method="pk", tolerance=1e-9,
                                    periodic_geometry=True, periodic_axes=(False, True)))

        t0 = time.perf_counter()
        solution = model.solve()
        dt = time.perf_counter() - t0

        row = outputs.make_row(CASE, BUILD, nx, h, extent_x, ref, solution, dt, load_total=load_total)
        rows.append(row)
        pressure = outputs.pressure_from_loads(solution["loads_z"], h, float(solution["total_normal_load"]))
        plots.plot_run(HERE / "data" / "plots" / f"N{nx}.png", roller_s, pressure, h, ref,
                       title=f"{CASE.label} / {BUILD} / N={nx}")
        print(f"N={nx:4d}  h={h:.3e}  nodes_in_2a={row['nodes_in_2b']:.1f}  "
              f"err a={row['err_a']:+.3%}  err p0={row['err_p0']:+.3%}  "
              f"err load={row['err_load']:+.2e}  t={dt:.2f} s")

    outputs.write_results(HERE / "data", CASE, BUILD, rows)


if __name__ == "__main__":
    main()