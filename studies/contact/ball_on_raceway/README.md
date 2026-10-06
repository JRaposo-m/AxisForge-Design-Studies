# ball_on_raceway — the AxisForge → SlipPY spike

**Status:** not started (after `examples/slippy/sphere_on_flat_vs_hertz`).

One bearing, one contact point: data from a solved AxisForge ISO/TS 16281
system, fed into SlipPY.

**Stages.**
1. AxisForge: build and solve; write the contact data of the most loaded ball
   to `artifacts/studies/contact/ball_on_raceway/`, stamped with the AxisForge commit.
2. SlipPY: read that file only (no AxisForge import), build the contact, solve.
3. Compare with Hertz (`references/contact/`) and with AxisForge's own Hertz.

**Interface fields (ADR-002):** `Q_j`, `alpha_j`, `phi_j`, `r1`, `r2` per ball and
raceway — **not** `delta_j` / `p_max` (they belong to the idealised Hertz law of
the equilibrium solve). Expect AxisForge's δ to differ from SlipPY's: that is
why `delta_j` is not in the interface (see also the `cp` factor-2 finding in the
slippy-study vault).

**Open decisions:** acceptance criteria; which Hertz is the reference; the
contract format between the two stages (option a/b/c).
