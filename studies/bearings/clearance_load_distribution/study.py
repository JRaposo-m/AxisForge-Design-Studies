# studies\bearings\clearance_load_distribution\study.py
"""
Effect of the radial internal clearance s on the internal load distribution of the
bearings (ISO/TS 16281) of the two-shaft spur gear system built in
construction.py.

Workflow
--------
1. build the system once (construction.build_system) and solve the shaft FEM once:
   with rigid supports the shaft is statically determinate, so the bearing reactions
   (Fr, Fa) do NOT depend on s. Only the bearing-internal distribution does.
2. for every s of the sweep, re-assemble the bearings (construction.build_bearings)
   and run the ISO/TS 16281 contact solver against the SAME FEM bearing nodes. Bearing 1
   is a deep groove ball bearing, bearing 2 a cylindrical roller bearing: each bearing is
   given to the solver the dispatcher resolves for its family (ball / roller).
   The inner-ring tilt psi is the shaft slope at the bearing, as given by the FEM
   (psi_input=False in the solver: no override). Its own effect is NOT studied here:
   see the misalignment study.
3. hand the rows to outputs.py, which owns everything that is written or drawn
   (console summary, checks, CSVs, figures). Change plots/columns there, not here.

Conventions
-----------
s            total radial internal clearance Gr [mm] (= diametral play, ISO 5753-1)
Q_j          rolling-element contact force [N] (ball, or sum of the roller laminae);
             phi_j = 0 is the element on the load line (phi_Fr)
q_k          roller lamina load [N] of the most loaded roller (ISO/TS 16281 sec. 5.2),
             lamina k at x_k along the roller length
delta_r      radial approach along phi_Fr [mm], measured from the concentric position:
             it includes the s/2 needed to close the clearance, so Fr / delta_r is a
             SECANT stiffness, not a tangent one
L10r         ISO/TS 16281 basic reference rating life [10^6 rev]; L10h assumes
             rpm = construction.SHAFT_RPM[shaft] (inner ring rotating)
"""
import math
from pathlib import Path

import numpy as np

import construction as cs
import outputs
# importing contact_solver registers the ball and roller solvers with the dispatcher
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (  # noqa: F401
    contact_solver,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import (
    resolve_solver_cls,
)

# =============================================================================
# Inputs
# =============================================================================

# --- sweep: total radial internal clearance Gr [mm], s >= 0 ---------------------
# 0 -> no clearance; ~0.005-0.020 is the usual CN range of a small deep groove ball
# bearing, C3/C4 above it (check the exact limits in ISO 5753-1).
S_VALUES_MM = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040]

THEORY_KEY = "timoshenko/cowper"            # key of construction.THEORY

# --- outputs (what is drawn/written is defined in outputs.py) -------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SHOW_PLOTS = False
# bearings shown in the figures: {kind: label}, one per bearing type (shaft_1 side)
REF_LABELS = {kind: cs.bearing_label(cs.SHAFT_NAMES[0], kind, n)
              for n, _, _, kind in cs.BEARING_SPECS}

ROW_FIELDS = [
    "s_mm", "shaft", "label", "kind", "position_mm", "arrangement", "Z", "alpha0_deg",
    "Fr_N", "Fa_N", "psi_mrad", "delta_r_mm", "delta_a_mm", "Kr_secant_N_per_mm",
    "n_loaded", "zone_half_angle_deg", "Q_max_N", "Q_max_over_Fr_per_Z", "Mz_Nmm",
    "q_lamina_max_N", "lamina_loaded_frac", "lamina_peak_over_mean",     # rollers only
    "equil_err_rel", "ok", "residual", "n_iter",
    "L10r_Mrev", "L10h_h", "postprocessed", "error",
]
Q_FIELDS = ["s_mm", "shaft", "label", "kind", "j", "phi_deg", "Q_N"]
# lamina loads q_k of the most loaded roller (line-contact bearings only)
LAMINA_FIELDS = ["s_mm", "shaft", "label", "kind", "roller_j", "k", "x_mm", "q_N"]

# =============================================================================
# Solve
# =============================================================================


def solve_fem(system):
    """{shaft_name: ShaftResults}, one global solve per shaft (reference clearance)."""
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
        RigidSupportFEMSolver,
    )
    settings = cs.THEORY[THEORY_KEY]
    return {ss.name: RigidSupportFEMSolver(settings).solve(ss) for ss in system.shafts}


def _solve_group(solver_cls, ss, bearings, shaft_results):
    """One solver class for the bearings it applies to (a solver refuses the others).
    postprocess can fail on its own (e.g. an unloaded row has no finite life), so a
    failure retries without it and the distribution is kept.
    -> (results {label: BearingAnalysisResult} or None, error text, postprocessed)"""
    try:
        return solver_cls(postprocess=True).solve(ss, bearings, shaft_results), "", True
    except Exception as exc_full:                                   # noqa: BLE001
        try:
            res = solver_cls(postprocess=False).solve(ss, bearings, shaft_results)
            return res, f"postprocess failed: {type(exc_full).__name__}: {exc_full}", False
        except Exception as exc:                                    # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}", False


def _solve_shaft(ss, bearings, shaft_results):
    """{label: (analysis or None, error, postprocessed)}. Bearings are grouped by the
    ISO/TS 16281 solver the dispatcher resolves for them (ball / roller)."""
    groups: dict[type, dict] = {}
    for label, b in bearings.items():
        groups.setdefault(resolve_solver_cls(b, label=label), {})[label] = b
    out = {}
    for solver_cls, brgs in groups.items():
        results, error, postprocessed = _solve_group(solver_cls, ss, brgs, shaft_results)
        for label in brgs:
            out[label] = (None if results is None else results.get(label), error, postprocessed)
    return out


def _row(s, ss, label, kind, bearing, analysis, error, postprocessed):
    row = {k: math.nan for k in ROW_FIELDS}
    row.update(s_mm=s, shaft=ss.name, label=label, kind=kind, position_mm=bearing.position,
               arrangement=bearing.arrangement, Z=bearing.Z,
               alpha0_deg=math.degrees(bearing.alpha_0),
               ok=False, postprocessed=postprocessed, error=error)
    if analysis is None:
        return row, [], []

    ld = analysis.load_distribution
    r = ld.row
    Q = np.asarray(r.Q_j, dtype=float)
    phi = (np.asarray(r.phi_j, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    loaded = Q > 0.0
    dFr, _ = ld.equilibrium_error
    Fr = ld.Fr
    rpm = cs.SHAFT_RPM[ss.name]
    life = analysis.basic_life

    row.update(
        Fr_N=Fr, Fa_N=ld.Fa, psi_mrad=1e3 * ld.psi,
        delta_r_mm=ld.delta_r, delta_a_mm=ld.delta_a,
        Kr_secant_N_per_mm=Fr / ld.delta_r if ld.delta_r > 0.0 else math.nan,
        n_loaded=r.n_loaded,
        zone_half_angle_deg=math.degrees(np.max(np.abs(phi[loaded]))) if loaded.any() else math.nan,
        Q_max_N=float(Q.max()),
        Q_max_over_Fr_per_Z=float(Q.max()) / (Fr / bearing.Z) if Fr > 0.0 else math.nan,
        Mz_Nmm=ld.Mz,
        equil_err_rel=abs(dFr) / Fr if Fr > 0.0 else math.nan,
        ok=bool(ld.ok), residual=ld.residual, n_iter=ld.n_iter,
    )
    if life is not None:
        row.update(L10r_Mrev=life.L10r, L10h_h=life.L10r * 1e6 / (60.0 * rpm))

    q_rows = [dict(s_mm=s, shaft=ss.name, label=label, j=j,
                   phi_deg=math.degrees(phi[j]), Q_N=float(Q[j])) for j in range(len(Q))]
    lam_rows = []
    if ld.is_line_contact:                        # lamina model: look inside the most loaded roller
        j = int(np.argmax(Q))
        q_k = np.asarray(r.q_jk[j], dtype=float)
        pos = q_k > 0.0
        row.update(q_lamina_max_N=float(q_k.max()),
                   lamina_loaded_frac=float(pos.mean()),
                   lamina_peak_over_mean=float(q_k.max() / q_k[pos].mean()) if pos.any() else math.nan)
        lam_rows = [dict(s_mm=s, shaft=ss.name, label=label, kind=kind, roller_j=j, k=k,
                         x_mm=float(r.x_k[k]), q_N=float(q_k[k])) for k in range(len(q_k))]
    return row, q_rows, lam_rows


KIND_OF = {cs.bearing_label(name, kind, n): kind
           for name in cs.SHAFT_NAMES for n, _, _, kind in cs.BEARING_SPECS}


def run_sweep(system, fem):
    rows, q_rows, lam_rows = [], [], []
    for s in S_VALUES_MM:
        bearings = cs.build_bearings(s)
        for ss in system.shafts:
            brgs = bearings[ss.name]
            solved = _solve_shaft(ss, brgs, fem[ss.name])
            for label, bearing in brgs.items():
                analysis, error, postprocessed = solved[label]
                row, q, lam = _row(s, ss, label, KIND_OF[label], bearing, analysis,
                                   error, postprocessed)
                rows.append(row)
                q_rows.extend(q)
                lam_rows.extend(lam)
    return rows, q_rows, lam_rows

# =============================================================================
# Run
# =============================================================================


def main():
    system = cs.build_system(cs.S_REFERENCE_MM)
    fem = solve_fem(system)
    rows, q_rows, lam_rows = run_sweep(system, fem)

    outputs.report(rows, q_rows, lam_rows, s_values=S_VALUES_MM, ref_labels=REF_LABELS,
                   out_dir=OUT_DIR, row_fields=ROW_FIELDS, q_fields=Q_FIELDS,
                   lamina_fields=LAMINA_FIELDS, show=SHOW_PLOTS)


if __name__ == "__main__":
    main()