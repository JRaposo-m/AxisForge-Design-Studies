"""two_profiles: both real bodies, each with its own geometry and material, vs analytic Hertz.

The type folder (``ball_flat``, ``ball_raceway``) selects the case from ``_common.case.CASES``;
this folder (``two_profiles``) is the build. Surface 1 (the body ``r1``, which must be curved) and
a curved surface 2 are discrete arrays of the analytic ``RoundSurface`` height on the same grid,
apexes on the same node; a flat surface 2 is analytic. A raceway groove (principal radii of opposite
sign) is a saddle that no ellipsoid represents, so its exact torus is written out in ``body_surface``.
What varies is the grid: ``N = 2**n``.
Run from anywhere: ``python run.py``. Results go to ``data/``.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))            # hertz_vs_bem/, to import _common

import slippy.contact as c
import slippy.surface as s

from _common import outputs, plots
from _common.case import CASES, EXTENT_IN_A, EXTENT_IN_A_BY_CASE, N_POWERS, N_POWERS_BY_CASE

BUILD = HERE.name                                   # "two_profiles"
CASE = CASES[HERE.parent.name]                      # "ball_flat" or "ball_raceway"


def body_surface(r, X, Y, h):
    """SlipPy surface of one body from its signed Hertz radii ``r = (rx, ry)`` [m].

    Convex > 0, concave < 0, ``math.inf`` = flat in that direction. SlipPy convention:
    ``profile = -sag`` (apex 0, falling away from the apex for a convex body).

    * both infinite: analytic flat;
    * same sign (sphere, cylinder, ellipsoid): analytic ``RoundSurface`` height with semi-axes
      ``(sqrt(|rx|*c), sqrt(|ry|*c), c)``, i.e. apex radii ``|rx|``, ``|ry|``, ``c`` = first finite
      radius; mirrored if the body is concave;
    * opposite signs (raceway groove: rolling radius ``rx`` > 0, groove radius ``ry`` < 0): a saddle,
      which no ellipsoid represents. Exact torus: the groove circle of radius ``|ry|`` is swept
      around an axis at distance ``rx`` from the apex. Near the apex ``sag ~ x^2/(2 rx) + y^2/(2 ry)``,
      the paraboloid assumed by Hertz.
    """
    if all(math.isinf(v) for v in r):
        return s.FlatSurface(shift=(0, 0))
    rx, ry = r
    finite = [v for v in r if math.isfinite(v)]
    if len(finite) == 2 and finite[0] * finite[1] < 0:
        s_y = ry - math.copysign(1.0, ry) * np.sqrt(ry**2 - Y**2)       # drop across the groove [m]
        rho = rx - s_y                                                  # local rolling radius [m]
        sag = rx - math.copysign(1.0, rx) * np.sqrt(rho**2 - X**2)      # drop along the rolling direction [m]
        return s.Surface(profile=-sag, grid_spacing=h)
    c_axis = abs(finite[0])
    semi_axes = (math.sqrt(abs(rx) * c_axis), math.sqrt(abs(ry) * c_axis), c_axis)
    sign = math.copysign(1.0, finite[0])
    return s.Surface(profile=sign * s.RoundSurface(semi_axes).height(X, Y), grid_spacing=h)


def main() -> None:
    assert CASE.angle == 0.0, "the builds assume aligned principal axes"
    assert not all(math.isinf(v) for v in CASE.r1), "surface 1 must be the curved body (SlipPy needs it discrete)"
    ref = c.hertz_full(**CASE.hertz_args())         # analytic reference, once per case

    a_ref = max(ref["contact_radii"])
    extent = 2.0 * EXTENT_IN_A_BY_CASE.get(CASE.label, EXTENT_IN_A) * a_ref              # side of the square domain [m]

    body_1 = c.Elastic("body_1", {"E": CASE.moduli[0], "v": CASE.v[0]})   # unique names per process
    body_2 = c.Elastic("body_2", {"E": CASE.moduli[1], "v": CASE.v[1]})
    rows = []
    for n_pow in N_POWERS_BY_CASE.get(CASE.label, N_POWERS):
        n = 2**n_pow
        h = extent / n
        x = (np.arange(n) - n / 2) * h              # nodes at i*h, apex at extent/2 -> node n/2
        X, Y = np.meshgrid(x, x)
        surf_1 = body_surface(CASE.r1, X, Y, h)
        surf_2 = body_surface(CASE.r2, X, Y, h)
        surf_1.material = body_1
        surf_2.material = body_2

        name = f"{CASE.label}_{BUILD}_N{n}"
        slippy_dir = HERE / "data" / "slippy"
        slippy_dir.mkdir(parents=True, exist_ok=True)   # ContactModel uses os.mkdir (not makedirs)
        model = c.ContactModel(name, surf_1, surf_2, output_dir=str(slippy_dir))
        model.add_step(c.StaticStep("load", normal_load=CASE.load, method="pk", tolerance=1e-9))

        t0 = time.perf_counter()
        solution = model.solve()
        dt = time.perf_counter() - t0

        row = outputs.make_row(CASE, BUILD, n, h, extent, ref, solution, dt)
        rows.append(row)
        pressure = outputs.pressure_from_loads(solution["loads_z"], h, float(solution["total_normal_load"]))
        plots.plot_run(HERE / "data" / "plots" / f"N{n}.png", surf_1, pressure, h, ref,
                       title=f"{CASE.label} / {BUILD} / N={n}")
        print(f"N={n:4d}  h={h:.3e}  nodes_in_2b={row['nodes_in_2b']:.1f}  "
              f"err p0={row['err_p0']:+.3%}  err delta={row['err_delta']:+.3%}  "
              f"apex offset={row['apex_node_offset']:.1f} nodes  t={dt:.2f} s")

    outputs.write_results(HERE / "data", CASE, BUILD, rows)


if __name__ == "__main__":
    main()