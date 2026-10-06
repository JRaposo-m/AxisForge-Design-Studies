# examples

The libraries being **used**, without an acceptance criterion on AxisForge.
For learning a library and keeping a worked record of how it is driven.

```
examples/
└── slippy/        own SlipPY examples, beyond the ones in the SlipPY repository
```

## Rules

- One folder per topic, named by the topic. Inside: README + numbered scripts.
- Scripts are self-contained and call the library directly: no shared
  orchestration code, no wrappers. Only `references` may be imported.
- What a topic teaches about the library (a default that surprised you, an
  argument convention) goes in that topic's README, in words.
- Generated files go to `artifacts/examples/<path>/` (git-ignored).
