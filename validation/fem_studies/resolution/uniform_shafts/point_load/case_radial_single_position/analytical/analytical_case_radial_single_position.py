"""
validation/fem_studies/resolution/uniform_shafts/point_load/case_radial_single_position/analytical/analytical_case_radial_single_position.py

Closed-form (NON-FEM) reference solution for case_radial_single_position,
in both Euler-Bernoulli and Timoshenko beam theory. Lives in its own
analytical/ folder, a plain THIRD SIBLING of the FEM case's own
euler_bernoulli/ and timoshenko/ folders (same shape: its own
results/, its own comparison/ -- see compare_analytical_case_radial_single_position.py
in analytical/comparison/), with one difference: no abaqus_results/,
since there is nothing Abaqus-shaped to diff against here. Both
theories' closed-form output sit under analytical/results/<theory>/,
split internally rather than via separate top-level folders the way
euler_bernoulli/timoshenko are -- see common/paths.py's own docstring
for the full layout and results_dir()'s exact contract from this
script's own perspective.

WHY THIS IS VALID FOR THIS CASE (read before reusing this pattern
elsewhere): both bearings are modelled by the FEM solver as v=0 point
constraints ONLY -- u=0 is added at the locating bearing, but theta
(bending rotation) is NEVER constrained by any bearing, regardless of
arrangement (see
solvers/.../fem_solvers/constraints/boundary_conditions.py). That makes
every bearing in this suite a true pin support with zero moment
reaction, which is exactly what a classical "simply supported beam with
overhangs, point loads, closed-form solution" assumes. If a future case
ever introduces a genuinely moment-reacting boundary condition, this
Macaulay-bracket approach would need a different (still closed-form,
but not "simple support") formulation -- flag that explicitly rather
than reusing this file's assumptions unchecked.

METHOD -- Macaulay's method, extended for Timoshenko shear deformation.
Each bending plane (XY, XZ) is solved INDEPENDENTLY, exactly mirroring
solvers/README.md's own "Each bending plane is solved independently
against the same stiffness matrix and superposed afterwards" -- here
there is no stiffness matrix, but the same independence holds for the
same physical reason (the two planes are uncoupled for a circular
section with no cross terms).

For one plane, given point loads P_k at positions x_k (support
reactions R_A/R_B counted as point loads too, solved from statics) at a
uniform beam of bending stiffness EI and shear stiffness kappa*G*A:

    M(x)      = SUM_k F_k * <x - x_k>            (bending moment)
    V(x)      = -SUM_k F_k * <x - x_k>^0          (shear force, k only
                                                     where x_k < x -- sign
                                                     fixed to match Q=-dM/dx,
                                                     see solve_plane()'s V())
    v(x)      = (1/EI)      * SUM_k F_k * <x-x_k>^3 / 6
              + (1/(kappa*G*A)) * SUM_k F_k * <x-x_k>
              + C1*x + C2

<.> is the Macaulay bracket: <x-x_k>^n = (x-x_k)^n if x >= x_k, else 0.
The bending term's second derivative reproduces M(x)/EI exactly; the
shear term's first derivative reproduces V(x)/(kappa*G*A) exactly --
this is the standard shear-corrected Macaulay formulation (see e.g.
Timoshenko beam theory textbooks referenced in
claude/timoshenko_beam_references.md). C1, C2 are solved from the two
physical boundary conditions v(x_A) = 0, v(x_B) = 0 -- NOT applied
separately to the bending and shear terms, since only the TOTAL
deflection has to vanish at a support.

kappa*G*A -> infinity recovers pure Euler-Bernoulli (shear term
vanishes) -- rather than special-casing that branch, EULER_KGA below is
just a very large number, and the code path is identical for both
theories. Confirmed negligible effect at that magnitude by the
self-check block (shear term contribution to v(x_A)/v(x_B) is exactly
zero regardless, since the WHOLE point of solving C1/C2 against the
combined bending+shear expression is that BOTH terms are re-zeroed at
the supports together, every time -- using a literal infinite kGA
instead would just require a special no-shear code path for no benefit).

SELF-CHECKS (run every time, raise AssertionError loudly rather than
silently producing a wrong curve -- "flag, don't silently fix"):
  1. v(x_A) and v(x_B) both come back numerically zero (validates the
     C1/C2 linear solve).
  2. M(x) at the beam's own far end (x=SHAFT_LENGTH_MM) comes back
     numerically zero -- this is NOT a modelling assumption, it's an
     independent check of global moment equilibrium (nothing forces it
     to be zero unless R_A/R_B were computed correctly), using a point
     the reaction solve itself never referenced.

GEAR-MESH FORCES: NOT re-derived by hand here. This module imports
build_system() from the sibling case_radial_single_position.py (one
directory up) and reads the SAME already-resolved ShaftSystem it does
-- same RadialLoad list, both "user"- and "gear_mesh"-sourced, each
already carrying its own position and (Fy, Fz) plane decomposition via
Load.component(). Computing Ft/Fr from torque and gear geometry by hand
a second time here would risk a second, independently-wrong source of
truth for the exact numbers this script is supposed to be an
independent check against; using the same resolved loads instead means
this script is deliberately only independent from the FEM DISCRETIZATION
step (mesh, elements, stiffness matrix, DOFs), not from the geometry/
statics step (core/, gear meshing) that produces the load case in the
first place. Flagged here explicitly since it's a real, deliberate
scope boundary, not an oversight.
"""
from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing 'common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )
sys.path.insert(0, str(_root))
from common.paths import results_dir  # noqa: E402

# analytical/'s own sibling case_radial_single_position.py, one
# directory up -- imported by path (not a package), same "walk up to a
# known anchor" trick used everywhere else in this suite, anchored on
# THIS file's own parent's parent rather than on 'common'.
_CASE_DIR = _p.parent.parent
sys.path.insert(0, str(_CASE_DIR))
from case_radial_single_position import (  # noqa: E402
    build_system, SHAFT_LENGTH_MM, SHAFT_DIAMETER_MM,
)

from axisforge.core.loads import LoadPlane  # noqa: E402
from axisforge.core.materials import get_material  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Section / material -- must match what case_radial_single_position.py's
# own make_shaft_geometry() actually builds. SectionSpec there does NOT
# pass material_id, so it falls back to SectionSpec's own default
# ("AISI_1045" -- see fixtures/construction/shafts/shaft_fixture.py).
# Section is solid (no inner_diameter passed), diameter SHAFT_DIAMETER_MM.
# FLAG: this is currently the one piece of case_radial_single_position.py
# NOT read back programmatically (unlike geometry/loads/bearings, which
# all come from build_system() itself) -- if that script's material_id
# or SHAFT_DIAMETER_MM / hollow-ness ever changes, this constant needs
# updating by hand to match. Kept a plain constant rather than silently
# guessed, so a mismatch is a one-line diff to spot, not a hidden bug.
# ---------------------------------------------------------------------------
_MATERIAL_ID = "AISI_1045"

_material = get_material(_MATERIAL_ID)
E_MPA = _material.E                      # 207 000 MPa
NU = _material.poisson_ratio             # 0.3
G_MPA = E_MPA / (2.0 * (1.0 + NU))       # ~79 615.4 MPa

_D = SHAFT_DIAMETER_MM
I_MM4 = 3.141592653589793 / 64.0 * _D**4     # second moment of area, solid circular
A_MM2 = 3.141592653589793 / 4.0 * _D**2      # cross-sectional area, solid circular

EI = E_MPA * I_MM4     # N.mm^2
GA = G_MPA * A_MM2     # N

# Shear correction factors, solid circular section, ratio=0 --
# formulas confirmed against
# axisforge/mesh/shaft/element_type/timoshenko/shear_factor.py
# (ShearFactor.cowper_factor / .hutchinson_factor), not re-derived
# independently -- this IS the same formula the FEM element itself
# uses, only evaluated here in closed form instead of inside a
# stiffness matrix. Re-deriving a DIFFERENT shear-factor formula here
# would defeat the point of an independent check (it would just be
# testing two different theories against each other, not testing the
# FEM's discretization of ONE theory).
def _cowper_factor(v: float) -> float:
    return 6.0 * (1.0 + v) / (7.0 + 6.0 * v)


def _hutchinson_factor(v: float) -> float:
    return 6.0 * (1.0 + v) ** 2 / (7.0 + 12.0 * v + 4.0 * v ** 2)


KAPPA = {
    "cowper": _cowper_factor(NU),
    "hutchinson": _hutchinson_factor(NU),
}

N_STATIONS = 401  # 0.5 mm spacing over 200 mm -- fine enough that
                   # compare.py's np.interp of this onto the FEM's own
                   # (coarser) x_nodes never has to extrapolate, and
                   # the analytical curve itself looks smooth on a plot.


# ---------------------------------------------------------------------------
# Macaulay-bracket beam solver -- one plane at a time.
# ---------------------------------------------------------------------------

def _macaulay(x, x0, n):
    """<x - x0>^n, vectorised. n=0 is a Heaviside step at x0 (1 AT x0,
    since we use >=), not a Dirac impulse -- there is no point load
    magnitude embedded in n=0 itself, that comes from the caller's own
    F_k multiplier."""
    d = x - x0
    if n == 0:
        return (d >= 0.0).astype(float)
    out = (d ** n)
    out = out * (d >= 0.0)
    return out


@dataclass
class PlaneResult:
    x_A: float
    x_B: float
    R_A: float
    R_B: float
    loads: list  # [(x_k, F_k), ...] -- APPLIED loads only, not reactions
    v: "callable"   # v(x_array) -> deflection [mm]
    M: "callable"   # M(x_array) -> bending moment [N.mm]
    V: "callable"   # V(x_array) -> shear force [N]


def solve_plane(loads: list[tuple[float, float]], x_A: float, x_B: float,
                 kGA: float) -> PlaneResult:
    """
    loads : list of (position_mm, force_N) for this plane only --
        already includes BOTH user- and gear_mesh-sourced point loads,
        NOT the support reactions (those are solved here).
    x_A, x_B : support positions, x_A < x_B (true pins, no moment
        reaction -- see module docstring for why that holds here).
    kGA : kappa * G * A for this theory [N] -- pass a very large number
        (e.g. 1e30) for the Euler-Bernoulli branch, which makes the
        shear term's contribution to v(x) exactly zero at working
        precision without a separate code path.

    Returns a PlaneResult whose .v(x) and .M(x) accept a numpy array
    of x-positions [mm] and return deflection [mm] / moment [N.mm].
    """
    import numpy as np

    if x_A >= x_B:
        raise ValueError(f"solve_plane: expected x_A < x_B, got {x_A}, {x_B}")

    sum_F = sum(F for _, F in loads)
    sum_M_about_A = sum(F * (x - x_A) for x, F in loads)  # loads only

    R_B = -sum_M_about_A / (x_B - x_A)
    R_A = -sum_F - R_B

    # Every force in the problem, reactions included -- this IS the
    # complete list M(x)/V(x) are built from.
    all_forces = list(loads) + [(x_A, R_A), (x_B, R_B)]

    def M(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for xk, Fk in all_forces:
            out += Fk * _macaulay(x, xk, 1)
        return out

    def V(x):
        """Shear force -- Q = -dM/dx (RESOLVED, see reference textbook
        Eq 1.24: Q = -dM/dx = -EI*d3w/dx3, cross-checked and confirmed
        against AxisForge's own EulerBernoulliPostProcessing.shear_force()
        and TimoshenkoPostProcessing.shear_force() / TimoshenkoBeam.
        shear_strain_matrix(), both of which independently match the
        same book exactly). The un-negated Macaulay sum below is
        +dM/dx (a step function: jumps by Fk at each xk, INCLUDING the
        reactions), so the returned value is negated to match Q=-dM/dx
        -- this was the root cause of every V-column sign mismatch
        against AxisForge's own V_xy/V_xz found when comparing reports
        (both Timoshenko and Euler-Bernoulli), previously left as an
        open item in this docstring."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for xk, Fk in all_forces:
            out += Fk * _macaulay(x, xk, 0)
        return -out

    def v_unnormalized(x):
        """v(x) BEFORE the C1*x + C2 correction -- bending + shear
        terms only, both built from the same all_forces list."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for xk, Fk in all_forces:
            out += Fk * _macaulay(x, xk, 3) / 6.0 / EI
            out += Fk * _macaulay(x, xk, 1) / kGA
        return out

    # Solve C1, C2 from v(x_A) = 0, v(x_B) = 0 on the TOTAL
    # (unnormalized + C1*x + C2) expression:
    #   v_unnorm(x_A) + C1*x_A + C2 = 0
    #   v_unnorm(x_B) + C1*x_B + C2 = 0
    v0_A = float(v_unnormalized(np.array([x_A]))[0])
    v0_B = float(v_unnormalized(np.array([x_B]))[0])
    lhs = np.array([[x_A, 1.0], [x_B, 1.0]])
    rhs = np.array([-v0_A, -v0_B])
    C1, C2 = np.linalg.solve(lhs, rhs)

    def v(x):
        return v_unnormalized(x) + C1 * np.asarray(x, dtype=float) + C2

    return PlaneResult(x_A=x_A, x_B=x_B, R_A=R_A, R_B=R_B, loads=loads, v=v, M=M, V=V)


# ---------------------------------------------------------------------------
# Per-shaft driver
# ---------------------------------------------------------------------------

def _plane_loads(shaft_system, plane) -> list[tuple[float, float]]:
    """Every RadialLoad on this shaft (user + gear_mesh sourced),
    projected onto ONE plane via Load.component() -- zero-magnitude
    entries kept out on purpose (a load with no component in this
    plane contributes nothing and would just be numerical noise in the
    reaction sums)."""
    out = []
    for ld in shaft_system.radial_loads:
        F = ld.component(plane)
        if F != 0.0:
            out.append((ld.position, F))
    return out


def solve_shaft(shaft_system, kGA: float):
    """Returns (result_xy, result_xz) -- one PlaneResult per bending
    plane, using this shaft's OWN bearing positions (read from the
    resolved system, not hardcoded -- shaft1/shaft2 share the same
    bearing positions in this case group today via
    common/bearings.py, but this does not assume that)."""
    x_A, x_B = sorted(b.position for b in shaft_system.bearings)

    loads_xy = _plane_loads(shaft_system, LoadPlane.XY)
    loads_xz = _plane_loads(shaft_system, LoadPlane.XZ)

    result_xy = solve_plane(loads_xy, x_A, x_B, kGA)
    result_xz = solve_plane(loads_xz, x_A, x_B, kGA)
    return result_xy, result_xz


def _self_check(name: str, result_xy: "PlaneResult", result_xz: "PlaneResult") -> None:
    import numpy as np

    tol_v = 1e-9   # mm -- BC residual, should be at linear-solve precision
    tol_M = 1e-6   # N.mm -- far-end moment-equilibrium residual

    for label, r in (("xy", result_xy), ("xz", result_xz)):
        vA = float(r.v(np.array([r.x_A]))[0])
        vB = float(r.v(np.array([r.x_B]))[0])
        assert abs(vA) < tol_v, (
            f"{name}/{label}: v(x_A={r.x_A}) = {vA!r}, expected ~0 -- "
            f"C1/C2 boundary-condition solve did not converge, do not "
            f"trust this curve."
        )
        assert abs(vB) < tol_v, (
            f"{name}/{label}: v(x_B={r.x_B}) = {vB!r}, expected ~0 -- "
            f"C1/C2 boundary-condition solve did not converge, do not "
            f"trust this curve."
        )
        M_far_end = float(r.M(np.array([SHAFT_LENGTH_MM]))[0])
        assert abs(M_far_end) < tol_M, (
            f"{name}/{label}: M(x={SHAFT_LENGTH_MM}) = {M_far_end!r}, "
            f"expected ~0 -- global moment equilibrium check failed, "
            f"reactions R_A/R_B are wrong."
        )
    print(f"    [self-check OK] {name}: BCs + far-end moment equilibrium "
          f"all within tolerance")


# ---------------------------------------------------------------------------
# Output -- CSV (schema-compatible with common/compare.py's compare_shaft(),
# used with an IDENTITY column_map since these columns are already named
# the same as the AxisForge resolution_csv columns they'll be diffed
# against) + a short text report + overlay plots.
#
# Column set mirrors resolution_csv.py's own BENDING & SHEAR + DEFLECTION
# columns (M_xz/M_xy/M, V_xz/V_xy/V, v_xz/v_xy/v) -- NOT its full set:
# u_mm (axial), theta_xz/theta_xy (bending rotation) and T_Nm (torque)
# are OUT OF SCOPE for this closed-form solver, which only models
# bending + transverse shear (no axial-load or torsion formulation
# implemented here) -- left out rather than filled with a placeholder
# value, so a caller can tell "not modelled" from "modelled and zero".
# ---------------------------------------------------------------------------

_CSV_COLUMNS = ("x_mm", "M_xz_Nmm", "M_xy_Nmm", "M_Nmm",
                 "V_xz_N", "V_xy_N", "V_N",
                 "v_xz_mm", "v_xy_mm", "v_mm")


def _write_csv(x, result_xy, result_xz, out_path: Path, decimals: int = 10) -> None:
    import numpy as np

    v_xy = result_xy.v(x)
    v_xz = result_xz.v(x)
    v_res = np.sqrt(v_xy ** 2 + v_xz ** 2)

    # M_xy is the bending moment from the XY-plane solve, M_xz from the
    # XZ-plane solve -- same plane-to-component naming ShaftResults
    # itself uses (see results/README.md).
    M_xy = result_xy.M(x)
    M_xz = result_xz.M(x)
    M_res = np.sqrt(M_xy ** 2 + M_xz ** 2)

    V_xy = result_xy.V(x)
    V_xz = result_xz.V(x)
    V_res = np.sqrt(V_xy ** 2 + V_xz ** 2)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for i in range(len(x)):
        writer.writerow(
            f"{val:.{decimals}f}" for val in
            (x[i], M_xz[i], M_xy[i], M_res[i],
             V_xz[i], V_xy[i], V_res[i],
             v_xz[i], v_xy[i], v_res[i])
        )
    out_path.write_text(buf.getvalue(), encoding="utf-8")


def _write_report(shaft_name: str, theory_label: str,
                   result_xy: "PlaneResult", result_xz: "PlaneResult",
                   out_path: Path) -> None:
    import numpy as np
    x = np.linspace(0.0, SHAFT_LENGTH_MM, N_STATIONS)

    def _plane_block(plane_name: str, r: "PlaneResult") -> list[str]:
        v = r.v(x)
        M = r.M(x)
        V = r.V(x)
        i_v = int(np.argmax(np.abs(v)))
        i_M = int(np.argmax(np.abs(M)))
        i_V = int(np.argmax(np.abs(V)))
        return [
            f"{plane_name} plane -- x_A={r.x_A:.3f} mm, x_B={r.x_B:.3f} mm",
            f"  R_A = {r.R_A:10.3f} N   R_B = {r.R_B:10.3f} N",
            "  applied loads: " + (", ".join(
                f"{F:.2f} N @ x={x0:.2f} mm" for x0, F in r.loads
            ) or "(none)"),
            f"  v_max = {v[i_v]:.6f} mm @ x={x[i_v]:.2f} mm",
            f"  M_max = {M[i_M]:.2f} N.mm @ x={x[i_M]:.2f} mm",
            f"  V_max = {V[i_V]:.2f} N @ x={x[i_V]:.2f} mm",
            "",
        ]

    lines = [
        f"{shaft_name} -- analytical (closed-form Macaulay) -- {theory_label}",
        "=" * 78,
        "",
        "Method: Macaulay's method (shear-corrected for Timoshenko), true pin",
        "supports (v=0 only, no moment reaction -- confirmed against",
        "solvers/.../constraints/boundary_conditions.py). See this script's",
        "own module docstring for the full derivation and assumptions.",
        "",
        "M/V sign convention: M follows the standard SUM_k Fk*<x-xk>",
        "Macaulay convention; V = -dM/dx (Q=-dM/dx, matching the",
        "reference textbook's Eq 1.24), confirmed against AxisForge's",
        "own EulerBernoulliPostProcessing/TimoshenkoPostProcessing",
        "shear_force() -- see solve_plane()'s own V() docstring.",
        "",
    ]
    lines += _plane_block("XY", result_xy)
    lines += _plane_block("XZ", result_xz)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_pair(x, y_xy, y_xz, x_A, x_B, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y_xy, "-", label=f"{ylabel.split()[0]}_xy (analytical)", color="black")
    ax.plot(x, y_xz, "--", label=f"{ylabel.split()[0]}_xz (analytical)", color="black")
    ax.axvline(x_A, color="grey", linestyle=":", linewidth=1)
    ax.axvline(x_B, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_plots(shaft_name: str, theory_label: str,
                  result_xy: "PlaneResult", result_xz: "PlaneResult",
                  out_dir: Path) -> None:
    import numpy as np
    x = np.linspace(0.0, SHAFT_LENGTH_MM, N_STATIONS)

    _plot_pair(
        x, result_xy.v(x), result_xz.v(x), result_xy.x_A, result_xy.x_B,
        "v [mm]", f"{shaft_name} -- analytical deflection ({theory_label})",
        out_dir / f"{shaft_name}_analytical_deflection.png",
    )
    _plot_pair(
        x, result_xy.M(x), result_xz.M(x), result_xy.x_A, result_xy.x_B,
        "M [N.mm]", f"{shaft_name} -- analytical bending moment ({theory_label})",
        out_dir / f"{shaft_name}_analytical_moment.png",
    )
    _plot_pair(
        x, result_xy.V(x), result_xz.V(x), result_xy.x_A, result_xy.x_B,
        "V [N]", f"{shaft_name} -- analytical shear force ({theory_label})",
        out_dir / f"{shaft_name}_analytical_shear.png",
    )


def _run_one_theory(system, kGA: float, theory_label: str, out_dir: Path) -> None:
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    plots_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, SHAFT_LENGTH_MM, N_STATIONS)

    for ss in system.shafts:
        result_xy, result_xz = solve_shaft(ss, kGA)
        _self_check(f"{ss.name}/{theory_label}", result_xy, result_xz)
        _write_csv(x, result_xy, result_xz, csv_dir / f"{ss.name}_analytical.csv")
        _write_report(ss.name, theory_label, result_xy, result_xz,
                      out_dir / f"{ss.name}_analytical_report.txt")
        _write_plots(ss.name, theory_label, result_xy, result_xz, plots_dir)
        print(f"[OK] {ss.name} ({theory_label}) analytical csv/report/plots "
              f"written under {out_dir}")


def main() -> None:
    _construction, system = build_system()

    print("case_radial_single_position -- analytical (closed-form) solution")
    print(f"  E={E_MPA:.1f} MPa, nu={NU}, G={G_MPA:.2f} MPa, "
          f"I={I_MM4:.3f} mm^4, A={A_MM2:.3f} mm^2  "
          f"(material={_MATERIAL_ID}, d={_D:.1f} mm, solid)")

    # Euler-Bernoulli -- shear term forced to ~zero via a very large kGA.
    # results_dir()'s extra segments join AFTER "results" here (unlike
    # case_radial_single_position.py's own euler_bernoulli/timoshenko
    # split, where theory sits ABOVE results/ -- see paths.py's own
    # docstring for why this script uses the plain results_dir() form).
    _run_one_theory(system, kGA=1.0e30, theory_label="euler_bernoulli",
                     out_dir=results_dir(__file__, "euler_bernoulli"))

    # Timoshenko -- one subfolder per shear theory, mirroring the FEM
    # case script's own cowper/hutchinson sweep.
    for shear_theory, kappa in KAPPA.items():
        kGA = kappa * GA
        print(f"  timoshenko/{shear_theory}: kappa={kappa:.6f}, kGA={kGA:.2f} N")
        _run_one_theory(system, kGA=kGA, theory_label=f"timoshenko/{shear_theory}",
                         out_dir=results_dir(__file__, "timoshenko", shear_theory))


if __name__ == "__main__":
    main()
