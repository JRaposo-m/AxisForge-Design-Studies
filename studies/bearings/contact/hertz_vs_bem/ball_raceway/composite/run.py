"""composite: Hertz equivalent paraboloid (elastic) against a flat, vs analytic Hertz.

The type folder (``ball_flat``, ``ball_raceway``) selects the case from ``_common.case.CASES``;
this folder (``composite``) is the build. What varies is the grid: ``N = 2**n``.
Run from anywhere: ``python run.py``. Results go to ``data/``.
"""
import math
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
CASE = CASES[HERE.parent.name]                      # "ball_flat" or "ball_raceway"


def main() -> None:
    ref = c.hertz_full(**CASE.hertz_args())         # analytic reference, once per case

    # --- transcription Hertz -> SlipPy (composite: equivalent body against a flat) -------------
    rx, ry = (float(r) for r in ref["relative_radii"])      # [m]
    cc = rx if math.isfinite(rx) else ry            # SlipPy third semi-axis: first finite radius
    semi_axes = (math.sqrt(rx * cc), math.sqrt(ry * cc), cc)   # apex radii of curvature (rx, ry)
    assert CASE.moduli[0] == CASE.moduli[1] and CASE.v[0] == CASE.v[1], "composite needs one material"
    e_body, nu = CASE.moduli[0], CASE.v[0]          # both bodies are the same steel
    a_ref = max(ref["contact_radii"])
    extent = 2.0 * EXTENT_IN_A_BY_CASE.get(CASE.label, EXTENT_IN_A) * a_ref              # side of the square domain [m]

    hw = extent / 2.0                               # SlipPy returns a flat (silently) outside the ellipsoid
    if hw**2 * (1.0 / semi_axes[0] ** 2 + 1.0 / semi_axes[1] ** 2) > 1.0:
        raise ValueError(f"domain half-width {hw:.4g} m reaches outside the ellipsoid {semi_axes}: reduce the extent")
    material = c.Elastic("equivalent", {"E": e_body, "v": nu})   # names must be unique per process
    rows = []
    for n_pow in N_POWERS_BY_CASE.get(CASE.label, N_POWERS):
        n = 2**n_pow
        round_s = s.RoundSurface(semi_axes, extent=(extent, extent), shape=(n, n), generate=True)
        flat_s = s.FlatSurface(shift=(0, 0))
        round_s.material = material
        flat_s.material = material
        h = float(round_s.grid_spacing if not hasattr(round_s.grid_spacing, "__len__")
                  else round_s.grid_spacing[0])

        name = f"{CASE.label}_{BUILD}_N{n}"
        slippy_dir = HERE / "data" / "slippy"
        slippy_dir.mkdir(parents=True, exist_ok=True)   # ContactModel uses os.mkdir (not makedirs)
        model = c.ContactModel(name, round_s, flat_s, output_dir=str(slippy_dir))
        model.add_step(c.StaticStep("load", normal_load=CASE.load, method="pk", tolerance=1e-9))

        t0 = time.perf_counter()
        solution = model.solve()
        dt = time.perf_counter() - t0

        row = outputs.make_row(CASE, BUILD, n, h, extent, ref, solution, dt)
        rows.append(row)
        pressure = outputs.pressure_from_loads(solution["loads_z"], h, float(solution["total_normal_load"]))
        plots.plot_run(HERE / "data" / "plots" / f"N{n}.png", round_s, pressure, h, ref,
                       title=f"{CASE.label} / {BUILD} / N={n}")
        print(f"N={n:4d}  h={h:.3e}  nodes_in_2b={row['nodes_in_2b']:.1f}  "
              f"err p0={row['err_p0']:+.3%}  err delta={row['err_delta']:+.3%}  "
              f"apex offset={row['apex_node_offset']:.1f} nodes  t={dt:.2f} s")

    outputs.write_results(HERE / "data", CASE, BUILD, rows)


if __name__ == "__main__":
    main()
