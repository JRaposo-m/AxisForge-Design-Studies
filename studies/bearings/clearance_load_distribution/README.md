# clearance_load_distribution

**Status:** planned. After `validation/bearings/iso16281_harris_examples`.

**Question.** How does radial internal clearance change the load zone, the
maximum rolling-element load Q_max and the modified rating life (ISO/TS 16281)
for a given bearing and load?

**Why.** Clearance is the main operating variable ISO 281 ignores and ISO/TS
16281 captures; it also sets Q_max, which is the input of the contact study.

**Sketch.** `construction.py` takes the clearance as a parameter; sweep from
preload to large clearance; plot load zone angle, Q_max and L10mr.
