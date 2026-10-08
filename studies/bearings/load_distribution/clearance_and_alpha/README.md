# clearance_and_alpha

**Question.** How do the radial internal clearance s and the free contact angle alpha_0 change the
load distribution, deflection, stiffness and reference life of a bearing at the nominal load?

For a ball bearing the two are one geometric quantity, cos(alpha_0) = 1 − s / (2A) with
A = ri + re − Dw, so each type sweeps the parameter it is defined by and reports both:

| Type | Swept axis | Also reported |
|---|---|---|
| `deep_groove/` | `s_mm` = 0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040 mm | `alpha0_deg` that s implies |
| `cylindrical_roller/` | `s_mm` (same values) | — (no contact angle) |
| `angular_contact/` | `alpha0_deg` = 10, 15, 20, 25, 30, 35, 40° | `s_mm` that alpha_0 implies |

| | |
|---|---|
| Load | nominal: 100 N m at 1000 rpm |
| Gears | spur (Fa = 0) |
| psi | FEM slope at the bearing (its own effect: `misalignment`) |
| AxisForge version | see `<type>/results/report.txt` |

**Angular contact with spur gears.** A single-row angular contact bearing has no external axial
load here, but every loaded ball produces an axial component Q sin α of the same sign and the
roller partner carries no axial load: axial equilibrium is not possible under a radial load. This
type is kept to see how the solver behaves on that case; its result does not describe a working
bearing. The physically meaningful case is the helical sub-case (planned).

**Ranges.** The s that alpha_0 = 10…40° implies is much larger than the 0…0.040 mm swept for the
deep groove bearing (for typical groove conformities A is a few tenths of a millimetre), so the
curves of the two ball bearings overlap only at the low end of the `vs s_mm` figure. Check the
`s_mm` column of `angular_contact/results/iso16281_rows.csv` after the first run and adjust the values if needed.

**Expected behaviour (to confirm, not an acceptance criterion).** s = 0 gives the widest loaded
zone and the lowest Q_max / (Fr / Z); clearance narrows the loaded zone, raises Q_max and lowers
L10r. delta_r includes the s / 2 needed to close the clearance, so the secant stiffness Fr / delta_r
falls with s.

**Checks.** Every row converged and in equilibrium (|Fr − Σ Fr_row| / Fr ≤ 1e-6); with spur gears
the bearing of the same type gives the same Q_max on both shafts (spread ≤ 1e-6).

## Files

| File | What it does |
|---|---|
| `study.py` | the axes of each type and `configure` (sets `s` or `alpha_0_deg`) |
| `<type>/run.py` | runs the question for one type; writes `results/` and `plots/` |
| `compare/compare.py` | the three types against `s_mm`, and the two ball bearings against `alpha0_deg` |

## Result

(to be written after the run with the corrected ball contact stiffness)
