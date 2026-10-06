# sphere_on_flat_vs_hertz — smooth, elastic: SlipPY BEM vs Hertz

**Status:** not started. First SlipPY example; no AxisForge involved.

## Objective

Drive the SlipPY BEM path on the simplest geometry with a closed-form answer,
under both load control and displacement control, and compare with Hertz.

## Comparison criteria — to be fixed BEFORE running

| Quantity | Criterion | Mesh |
|---|---|---|
| p0, a | _to fix_ | 2a/h ≈ 40, error decreasing on two finer meshes |
| δ | _to fix_ | idem |
| W under displacement control | _to fix_ | idem |
| Always | solver converged; p = 0 on the two outer rows; complementarity residual reported | — |

Candidate values: the BEM study plan (3 % / 2 %) or the stricter BEM_PK guide
(2 % / 1 %, profile L2 ≤ 3 %). Decision pending.

## Reference

`references/contact/` (Hertz). Which implementation is the reference is still
open: own implementation, `slippy.contact.hertz_full` (with or without the
30/09 patch), or AxisForge's `contact_stiffness`.

## Next (after this one)

Ellipsoidal contact (two radii per body), then rough surfaces.
