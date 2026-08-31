# Studies

Each folder below is one self-contained design problem, solved through AxisForge's capability pipeline (see [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)). Every study has its own `README.md` stating the problem, which capability stages it exercises, and its current status.

| Study | Problem | Stages exercised |
|---|---|---|
| [`01_stepped_shaft_static`](01_stepped_shaft_static) | A three-section stepped shaft under an external radial load and torque | Construction, Resolution (`shaft_fem`), ElementAnalysis (`shaft_static_report`) |
| [`02_deep_groove_bearing_life`](02_deep_groove_bearing_life) | A deep-groove ball bearing under combined radial/axial load | Construction, ElementAnalysis (`point_contact`) |
| [`03_helical_gearbox_drivetrain`](03_helical_gearbox_drivetrain) | A single-stage helical gearbox — two shafts, one mesh, four bearings | All four stages, end to end |

The studies are ordered by how much of the pipeline they exercise, not by difficulty — `01` is the simplest complete run, `03` is the full chain.
