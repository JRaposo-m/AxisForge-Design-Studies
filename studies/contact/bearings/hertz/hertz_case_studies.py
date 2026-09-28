"""
studies/contact/bearings/hertz/hertz_case_studies.py

Case studies: Hertz contact stress (slippy.contact.hertz_full) for the
individual rolling elements of the bearings already solved by
ISO/TS 16281 in system_construction.py (same folder -- imported here,
not rebuilt).

Case 1 -- point contact (ball bearings, B1_dgb / B3_ang):
    per loaded element (Q_j > 0), raceway geometry is RE-DERIVED at the
    WORKING contact angle alpha_j -- not the nominal alpha_0. alpha_j
    shifts element-to-element under combined radial+axial load (see
    ISO16281BallSolver.elements()); only r_bx_inner/r_bx_outer depend on
    the angle (r_ax/r_ay and r_by_inner/r_by_outer don't), so a fresh
    BallBearingGeometry(alpha_0=alpha_j) is built per element. Both
    inner and outer raceway contacts are solved.

Case 2 -- line contact (roller bearings, B2_roll / B4_roll), COARSE:
    per loaded roller, a single uniform line load = Q_j / Lwe. This
    intentionally ignores the lamina model's own non-uniform q_jk /
    edge effect (P_xk) -- ISO/TS 16281's Sec 5.2 exists precisely to
    capture that, so this is a simplification, not a replacement. A
    per-lamina Hertz study (finer, using q_jk / lamina width) is a
    separate, more detailed case for later -- not done here.

Units: E in MPa, lengths in mm, loads in N -- hertz_full returns
pressures in MPa (N/mm^2) under these consistent units.
"""

import numpy as np
import slippy.contact as sc

from axisforge.core.machine_elements.bearings.families.geometry import BallBearingGeometry

from system_construction import brg_dgb, brg_ang, brg_roll1, brg_roll2, iso_results


def ball_case_study(bearing, label):
    """Returns a list of per-(element, race) records, one dict each, so
    hertz_critical_plots.py (same folder) can reuse this instead of
    recomputing the same Hertz solves."""
    row = iso_results[label].load_distribution.row
    print(f"\n=== {label} (point contact, Z={bearing.Z}) ===")
    records = []
    for j, (Qj, alpha_j) in enumerate(zip(row.Q_j, row.alpha_j)):
        if Qj <= 0.0:
            continue
        geo = BallBearingGeometry(Dw=bearing.Dw, Dpw=bearing.Dpw, A=bearing.A,
                                   s=bearing.s, ri=bearing.ri, re=bearing.re,
                                   alpha_0=float(alpha_j))
        for race, r2 in (("inner", [geo.r_bx_inner, geo.r_by_inner]),
                         ("outer", [geo.r_bx_outer, geo.r_by_outer])):
            res = sc.hertz_full(
                r1=[geo.r_ax, geo.r_ay], r2=r2,
                moduli=[bearing.e1, bearing.e2], v=[bearing.nu1, bearing.nu2],
                load=float(Qj),
            )
            a, b = res["contact_radii"]
            print(f"  elem {j:>2} ({race:>5})  alpha_j={np.degrees(alpha_j):6.2f} deg  "
                  f"Q_j={Qj:8.2f} N  p0={res['max_pressure']:9.1f} MPa  "
                  f"a={a:.4f} mm  b={b:.4f} mm")
            records.append(dict(label=label, contact="point", elem=j, race=race,
                                 Qj=float(Qj), result=res))
    return records


def roller_case_study(bearing, label):
    """Same as ball_case_study, for line contact (COARSE Q_j/Lwe)."""
    row = iso_results[label].load_distribution.row
    print(f"\n=== {label} (line contact, Z={bearing.Z}, COARSE -- Q_j/Lwe) ===")
    r1 = [bearing.Dwe / 2.0, float("inf")]
    r2_inner = [(bearing.Dpw - bearing.Dwe) / 2.0, float("inf")]
    r2_outer = [(bearing.Dpw + bearing.Dwe) / 2.0, float("inf")]

    records = []
    for j, Qj in enumerate(row.Q_j):
        if Qj <= 0.0:
            continue
        load_per_length = float(Qj) / bearing.Lwe
        for race, r2 in (("inner", r2_inner), ("outer", r2_outer)):
            res = sc.hertz_full(
                r1=r1, r2=r2,
                moduli=[bearing.e1, bearing.e2], v=[bearing.nu1, bearing.nu2],
                load=load_per_length, line=True,
            )
            a = res["contact_radii"][0]
            print(f"  elem {j:>2} ({race:>5})  Q_j={Qj:8.2f} N  "
                  f"q'={load_per_length:7.2f} N/mm  p0={res['max_pressure']:9.1f} MPa  "
                  f"half-width a={a:.4f} mm")
            records.append(dict(label=label, contact="line", elem=j, race=race,
                                 Qj=float(Qj), result=res))
    return records


if __name__ == "__main__":
    ball_case_study(brg_dgb, "B1_dgb")
    ball_case_study(brg_ang, "B3_ang")
    roller_case_study(brg_roll1, "B2_roll")
    roller_case_study(brg_roll2, "B4_roll")
