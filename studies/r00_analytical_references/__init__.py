"""
references -- closed-form solutions used as independent judges.

Never call an AxisForge or SlipPY solver from here: a reference must not
reuse the code it checks. Reading AxisForge *input* objects (loads, supports
of a constructed ``ShaftSystem``) is allowed, so the reference sees exactly
the loads the solver saw.
"""
