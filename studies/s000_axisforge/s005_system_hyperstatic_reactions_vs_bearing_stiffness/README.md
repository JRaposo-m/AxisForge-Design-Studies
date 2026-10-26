# hyperstatic_reactions_vs_bearing_stiffness

**Status:** planned.

**Question.** On a shaft with three or more supports, how much do the support
reactions change when the bearing stiffness changes (i.e. when the bearing is
swapped)?

**Why.** Answers the open caveat in the project doc
`design_studies_architecture_next_steps.md` (decision 2): reactions are
invariant to the bearing choice only on an isostatic shaft. This study says by
how much that assumption fails on a hyperstatic one.

**To decide before starting.** Which AxisForge solver puts bearing stiffness in
the global matrix (the rigid-support solver used in `validation/` does not).
