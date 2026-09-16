"""
check_cantilever_lambda_sweep.py

Standalone verification rig -- NOT part of the AxisForge repo, NOT going
through ShaftSystem/Mesh1D/RigidSupportFEMSolver at all. Two UNIFORM
(single-diameter) cantilever "shafts", swept over a range of slenderness,
built directly from Elem + StiffnessMatrixBuilder, to check AxisForge's
own Timoshenko element against closed-form beam theory -- specifically
the Ferreira/Fantuzzi-style cantilever benchmark (Fig 2.6 in the
reference you posted): a 2-node Timoshenko element loaded as a
cantilever, r_w = w_tip(Timoshenko FEM) / w_tip(exact Euler-Bernoulli),
plotted against the beam slenderness ratio lambda.

WHY THIS BYPASSES ShaftSystem ENTIRELY
---------------------------------------
A true cantilever needs u=v=theta=0 at the fixed end. boundary_dofs()
in constraints/boundary_conditions.py can only ever set v=0 (always)
and u=0 (locating bearing only) at bearing nodes -- no bearing
arrangement can ever produce theta=0. So a rigid cantilever BC is
structurally inexpressible through the bearing-based path, regardless
of how many/how stiff the bearings are made. Rather than bend
RigidSupportFEMSolver into a second mode (which its own docstring
explicitly argues against -- "not one of several modes... a sibling
module, not a flag"), this script talks to Elem/StiffnessMatrixBuilder
directly:

  - Elem has a plain public constructor (length, E, I, A, v,
    idx_node_1, idx_node_2, x_a, x_b, settings) -- confirmed from
    elem.py, no Mesh1D/ShaftSystem required to build one.
  - StiffnessMatrixBuilder(mesh, elements, frame) only ever reads
    mesh.x_nodes / mesh.n_nodes from its `mesh` argument (confirmed
    from build_stiffness_matrix.py) -- so a tiny local duck-typed
    stand-in (_FlatMesh below) is enough, no real Mesh1D needed.
  - _explicit_boundary_dofs() below is a local mirror of the
    `boundary_dofs_explicit()` function proposed (not yet applied) as
    an addition to constraints/boundary_conditions.py in this same
    conversation -- same contract (raw DOF indices in, (free,
    constrained) out), kept local here so this script runs today
    without requiring that repo change first. If you do add it to the
    repo, swap the import in and delete the local copy.

frame=False is used throughout (pure bending/shear, no axial): the
cantilever test never applies or reads an axial load, and frame=False
simply leaves axial DOFs uncoupled at zero in K -- fine here because
free_dofs below excludes every axial DOF outright (never solved for,
never singular).

THEORY -- closed form, derived independently in this conversation via
sympy directly from AxisForge's own B_b/B_s matrices (two_noded.py),
confirmed to reduce EXACTLY to the formula you quoted earlier this
session (r_w=(3*lambda**2+3)/(4*lambda**2)) under the substitution
Phi = 3/lambda**2, and confirmed AGAIN against the two asymptotes
labelled on your reference figure (0.75 and 0.938 as lambda -> inf):

    Phi = 12*E*I/(kGA*L**2)              (AxisForge's own shear param)
    lambda = sqrt(3/Phi) = (L/2)*sqrt(kGA/EI)   (the reference's param)

    single_point (= the reference's "reduced integration, 1 point"),
    1 element:  r_w(Phi) = (Phi+3)/4              -> Phi->0: r_w->3/4
    2 elements: r_w(Phi) = Phi/4 + 15/16          -> Phi->0: r_w->15/16 = 0.9375

    exact (= the reference's "exact integration, 2 points") --
    the LOCKING case, not a good-behaviour case:
    1 element:  r_w(Phi) = Phi*(Phi+4)/(4*(Phi+1))  -> Phi->0: r_w->0
    2 elements: r_w(Phi) = Phi*(Phi+4)/(4*Phi+1)    -> Phi->0: r_w->0

0.75 and 0.9375 match the figure's "(r_w=0.75; lambda->inf)" and
"(r_w=0.938; lambda->inf)" labels to the precision the figure prints
them at -- this is why we're confident in the Phi<->lambda mapping
above, not just asserting it.

WHAT THIS SCRIPT ACTUALLY CHECKS
----------------------------------
For two independently-chosen uniform "shafts" (different diameter AND
material -- deliberately, so a match is not an accident of one
particular cross-section), sweep lambda across a wide range, and for
each point:
  1. Solve the 1-element and 2-element cantilever FEM directly
     (single_point AND exact integration_method), read v_tip.
  2. Compute numeric r_w = v_tip / (P*L**3/(3*EI)) (the closed-form
     exact Euler-Bernoulli deflection is the denominator, matching the
     reference's own definition -- not an EB FEM solve).
  3. Compare against the four closed forms above. Report max error --
     this should be at machine precision (~1e-12) if AxisForge's
     stiffness_element() truly implements the matrices it documents.
     A large error here would mean the code and its own documented
     formulas have diverged -- flag it, don't explain it away.

Also writes a CSV + PNG plot (r_w vs lambda, log-x) so you can eyeball
it directly against your reference figure.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.element_type.timoshenko.two_noded import TimoshenkoBeam
from axisforge.mesh.shaft.element_type.timoshenko.shear_factor import ShearFactor
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.build_stiffness_matrix import (
    StiffnessMatrixBuilder,
)

HERE = Path(__file__).resolve().parent

SHEAR_THEORY = "cowper"
P_TIP = 1000.0  # N -- arbitrary, r_w is load-independent (linear system)


# ---------------------------------------------------------------------------
# Minimal stand-ins so we never need ShaftSystem/Mesh1D/RigidSupportFEMSolver
# ---------------------------------------------------------------------------

class _FlatMesh:
    """Duck-types the two attributes StiffnessMatrixBuilder actually reads
    off its `mesh` argument (mesh.x_nodes, mesh.n_nodes) -- confirmed from
    build_stiffness_matrix.py. Nothing Mesh1D-specific is used."""

    def __init__(self, x_nodes: list[float]):
        self.x_nodes = x_nodes
        self.n_nodes = len(x_nodes)


def _explicit_boundary_dofs(n_nodes: int, constrained: list[int]) -> tuple[list[int], list[int]]:
    """
    Local mirror of the proposed `boundary_dofs_explicit()` addition to
    constraints/boundary_conditions.py (not yet applied to the repo --
    see this script's module docstring). Same contract: raw global DOF
    indices in (3*i, 3*i+1, 3*i+2 = u,v,theta for node i), (free,
    constrained) out. No bearing/ShaftSystem logic at all.
    """
    n_dofs = 3 * n_nodes
    constrained = sorted(set(constrained))
    invalid = [d for d in constrained if d < 0 or d >= n_dofs]
    if invalid:
        raise ValueError(
            f"_explicit_boundary_dofs: DOF index/indices {invalid} out of "
            f"range for {n_nodes} nodes (0..{n_dofs - 1})."
        )
    free = [d for d in range(n_dofs) if d not in constrained]
    return free, constrained


# ---------------------------------------------------------------------------
# Cantilever rig
# ---------------------------------------------------------------------------

def kGA_for(E: float, poisson: float, A: float, shear_theory: str) -> float:
    """Uses AxisForge's own shear_factor() so kGA here matches exactly
    what stiffness_element() itself will use -- no re-derivation of the
    Cowper factor by hand."""
    G = E / (2.0 * (1.0 + poisson))
    dummy_settings = BeamModelSettings(
        beam_theory="timoshenko", shear_theory=shear_theory, integration_method="single_point",
    )
    dummy_elem = Elem(length=1.0, E=E, I=1.0, A=A, v=poisson,
                       idx_node_1=0, idx_node_2=1, x_a=0.0, x_b=1.0, settings=dummy_settings)
    k = TimoshenkoBeam().shear_factor(dummy_elem, ShearFactor(), shear_theory)
    return k * G * A


def solve_cantilever_tip(L: float, E: float, I: float, A: float, poisson: float,
                          n_elem: int, integration_method: str, P: float = P_TIP) -> float:
    """Builds a uniform n_elem-element cantilever directly from Elem +
    StiffnessMatrixBuilder, fixes node 0 fully (u=v=theta=0), applies a
    transverse tip load, solves, returns v_tip."""
    settings = BeamModelSettings(
        beam_theory="timoshenko", shear_theory=SHEAR_THEORY, integration_method=integration_method,
    )
    x_nodes = list(np.linspace(0.0, L, n_elem + 1))
    elements = [
        Elem(length=x_nodes[i + 1] - x_nodes[i], E=E, I=I, A=A, v=poisson,
             idx_node_1=i, idx_node_2=i + 1, x_a=x_nodes[i], x_b=x_nodes[i + 1], settings=settings)
        for i in range(n_elem)
    ]

    mesh = _FlatMesh(x_nodes)
    builder = StiffnessMatrixBuilder(mesh, elements, frame=False)
    K = builder.build()

    n_nodes = len(x_nodes)
    free_dofs, _ = _explicit_boundary_dofs(n_nodes, [0, 1, 2])
    # frame=False leaves every axial DOF uncoupled/zero in K -- this rig
    # never applies or reads an axial load, so drop them from free_dofs
    # outright (never solved for, never a source of singularity).
    free_dofs = [d for d in free_dofs if d % 3 != 0]

    K_red = K[np.ix_(free_dofs, free_dofs)]

    n_dofs = 3 * n_nodes
    f = np.zeros(n_dofs)
    tip_node = n_nodes - 1
    f[3 * tip_node + 1] = P

    d_full = np.zeros(n_dofs)
    d_full[free_dofs] = np.linalg.solve(K_red, f[free_dofs])
    return d_full[3 * tip_node + 1]


# ---------------------------------------------------------------------------
# Closed-form theory (see module docstring for the derivation/validation)
# ---------------------------------------------------------------------------

def theory_rw(Phi: float, n_elem: int, integration_method: str) -> float:
    if integration_method == "single_point":
        if n_elem == 1:
            return (Phi + 3.0) / 4.0
        elif n_elem == 2:
            return Phi / 4.0 + 15.0 / 16.0
    elif integration_method == "exact":
        if n_elem == 1:
            return Phi * (Phi + 4.0) / (4.0 * (Phi + 1.0))
        elif n_elem == 2:
            return Phi * (Phi + 4.0) / (4.0 * Phi + 1.0)
    raise ValueError(f"theory_rw: no closed form for n_elem={n_elem}, integration_method={integration_method!r}")


def lambda_from_Phi(Phi: float) -> float:
    return math.sqrt(3.0 / Phi)


# ---------------------------------------------------------------------------
# Two independent uniform "shafts" -- different diameter AND material, so
# agreement isn't an accident of one specific cross-section. Both swept
# over the SAME lambda range: if r_w truly depends only on Phi (hence only
# on lambda), the two shafts' r_w(lambda) curves must be indistinguishable
# even though their (L, d, E) values never coincide.
# ---------------------------------------------------------------------------

SHAFTS = {
    "shaft_A_steel_d20": dict(d=20.0, E=210000.0, poisson=0.30),   # steel
    "shaft_B_alu_d55":   dict(d=55.0, E=71000.0,  poisson=0.33),   # aluminium, different d and E
}

# Sweep target: Phi from very slender (1e-4) to very stubby (50) -- lambda
# from ~5477 down to ~0.24 via lambda=sqrt(3/Phi).
PHI_TARGETS = np.geomspace(1e-4, 50.0, 24)


def run() -> None:
    rows = []
    for shaft_name, geo in SHAFTS.items():
        d = geo["d"]
        E = geo["E"]
        poisson = geo["poisson"]
        I = math.pi * d ** 4 / 64.0
        A = math.pi * d ** 2 / 4.0
        kGA = kGA_for(E, poisson, A, SHEAR_THEORY)

        for Phi_target in PHI_TARGETS:
            # Phi = 12EI/(kGA*L^2)  =>  L = sqrt(12EI/(kGA*Phi_target))
            L = math.sqrt(12.0 * E * I / (kGA * Phi_target))
            Phi_actual = 12.0 * E * I / (kGA * L ** 2)  # sanity round-trip
            lam = lambda_from_Phi(Phi_actual)
            v_euler = P_TIP * L ** 3 / (3.0 * E * I)

            for n_elem in (1, 2):
                for integ in ("single_point", "exact"):
                    v_tip = solve_cantilever_tip(L, E, I, A, poisson, n_elem, integ)
                    rw_numeric = v_tip / v_euler
                    rw_theory = theory_rw(Phi_actual, n_elem, integ)
                    err = abs(rw_numeric - rw_theory)
                    rel_err = err / abs(rw_theory) if rw_theory else err
                    rows.append(dict(
                        shaft=shaft_name, d=d, E=E, L=L, Phi=Phi_actual, lam=lam,
                        n_elem=n_elem, integration_method=integ,
                        rw_numeric=rw_numeric, rw_theory=rw_theory,
                        abs_err=err, rel_err=rel_err,
                    ))

    # -------------------------------------------------------------
    # Console summary: worst-case error per (n_elem, integration_method)
    # -- this is the actual pass/fail signal. Should be ~machine precision.
    # -------------------------------------------------------------
    print("=" * 92)
    print("Cantilever lambda-sweep -- AxisForge FEM vs closed-form theory")
    print("=" * 92)
    combos = sorted(set((r["n_elem"], r["integration_method"]) for r in rows))
    for n_elem, integ in combos:
        subset = [r for r in rows if r["n_elem"] == n_elem and r["integration_method"] == integ]
        worst = max(subset, key=lambda r: r["rel_err"])
        print(f"  n_elem={n_elem}  integration_method={integ:13s}  "
              f"max rel. error = {worst['rel_err']:.3e}  "
              f"(at shaft={worst['shaft']}, lambda={worst['lam']:.3g})")

    print()
    print("Asymptotic check (most slender point in the sweep, Phi -> 0):")
    for n_elem, integ in combos:
        subset = [r for r in rows if r["n_elem"] == n_elem and r["integration_method"] == integ]
        most_slender = min(subset, key=lambda r: r["Phi"])
        print(f"  n_elem={n_elem}  {integ:13s}  lambda={most_slender['lam']:.1f}  "
              f"r_w_numeric={most_slender['rw_numeric']:.6f}  "
              f"r_w_theory={most_slender['rw_theory']:.6f}")

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------
    csv_path = HERE / "cantilever_lambda_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {csv_path.name} written ({len(rows)} rows)")

    # -------------------------------------------------------------
    # Plot: r_w vs lambda (log-x), one panel per n_elem, both shafts
    # overlaid per integration_method to show shaft-independence.
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, n_elem in zip(axes, (1, 2)):
        for integ, style in (("single_point", "-"), ("exact", "--")):
            for shaft_name, marker in zip(SHAFTS, ("o", "x")):
                subset = sorted(
                    (r for r in rows if r["n_elem"] == n_elem and r["integration_method"] == integ
                     and r["shaft"] == shaft_name),
                    key=lambda r: r["lam"],
                )
                lams = [r["lam"] for r in subset]
                rws = [r["rw_numeric"] for r in subset]
                ax.plot(lams, rws, style, marker=marker, markersize=4,
                        label=f"{integ} ({shaft_name})", alpha=0.8)
        # theory overlay (shaft-independent, function of Phi/lambda only)
        lam_dense = np.geomspace(min(r["lam"] for r in rows), max(r["lam"] for r in rows), 200)
        for integ, style in (("single_point", "-"), ("exact", "--")):
            phi_dense = 3.0 / lam_dense ** 2
            rw_dense = [theory_rw(p, n_elem, integ) for p in phi_dense]
            ax.plot(lam_dense, rw_dense, style, color="black", linewidth=1,
                    label=f"{integ} theory")
        ax.set_xscale("log")
        ax.set_xlabel("lambda = sqrt(3/Phi)")
        ax.set_title(f"{n_elem} element(s)")
        ax.axhline(1.0, color="gray", linewidth=0.5)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("r_w = w_tip(Timoshenko) / w_tip(exact Euler-Bernoulli)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Cantilever benchmark -- AxisForge vs closed-form theory (two independent uniform shafts)")
    fig.tight_layout()
    png_path = HERE / "cantilever_lambda_sweep.png"
    fig.savefig(png_path, dpi=150)
    print(f"[OK] {png_path.name} written")


if __name__ == "__main__":
    run()