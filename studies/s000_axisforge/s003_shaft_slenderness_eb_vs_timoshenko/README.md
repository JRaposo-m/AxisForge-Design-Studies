# slenderness_eb_vs_timoshenko

**Status:** planned.

**Question.** From which slenderness L/d (and support/load configuration) does
shear deformation stop being negligible, i.e. where does Euler–Bernoulli start
to under-predict deflection and bearing misalignment by more than a chosen
tolerance?

**Why.** The two FEM validation cases show EB vs Timoshenko at one L/d each;
this turns it into a design rule for choosing the beam model.

**Sketch.** Sweep L/d (`construction.py` parameterised by L and d), solve with
both theories on a converged mesh, plot the relative difference in v_max and
in the slope at the bearings. Reference for the trend: `references/beams/`.
