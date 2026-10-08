# misalignment

**Question.** How does the inner-ring misalignment (tilt) psi change the load distribution of a
bearing, and how does the radial internal clearance s change that effect?

| | |
|---|---|
| Axis | `psi_mrad`: prescribed inner-ring tilt on the Fr plane [mrad] = 0, 0.5, 1, 1.5, 2, 3, 4, 5; the FEM slope of this system (about 2 mrad) is recorded in the RUN INFORMATION of `report.txt` as `psi_fem_mrad` |
| Load | nominal: 100 N m at 1000 rpm, spur gears (Fa = 0) |
| Second axis | `s_mm`: total radial internal clearance [mm] = 0, 0.010, 0.020 (one line per value in the figures) |
| Types | `deep_groove/`, `cylindrical_roller/` |
| AxisForge version | see `<type>/results/report.txt` |

**Sign of psi.** Only psi ≥ 0 is swept: the deep groove ball bearing and a symmetric roller are
mirror-symmetric, so −psi gives the same Q_j (delta_a and Mz change sign). This does not hold for
an angular contact bearing under Fa (planned question `psi_sign_vs_axial`).

**Figures.** `plots/summary/` has one line per s; `plots/load_distribution/` one polar drawing per
(psi, s); one overview per value of s; for the roller, one lamina profile per value of s.

**What to look at.** The roller is sensitive to tilt through edge loading: the lamina profile of
the most loaded roller (`lamina_peak_over_mean`, `cylindrical_roller/plots/roller_lamina_profile.png`)
and the drop of L10r. Mz is the tilting moment the supports would have to take: the shaft FEM with
pin supports ignores it.

## Files

| File | What it does |
|---|---|
| `study.py` | the two axes; psi is prescribed by the default `psi` hook, s is set by `configure` |
| `<type>/run.py` | runs the question for one type; writes `results/` and `plots/` |
| `compare/compare.py` | deep groove ball vs cylindrical roller, from the two `results/`: one folder of figures per value of s |

## Result

(to be written after the run with the corrected ball contact stiffness)
