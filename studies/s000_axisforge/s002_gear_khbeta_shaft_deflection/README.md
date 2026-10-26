# khbeta_shaft_deflection

**Status:** planned. After `validation/gears/iso6336_tr30_examples`.

**Question.** How does the face load factor K_Hβ change when the mesh
misalignment comes from the actual shaft deflection computed by the FEM,
instead of the simplified estimate of ISO 6336-1?

**Why.** This is a system-level result that only a coupled shaft + gear model
gives; it is where AxisForge adds something a gear-only calculation cannot.

**To decide before starting.** Which K_Hβ method AxisForge implements and how
the misalignment from the shaft solution is fed into it.
