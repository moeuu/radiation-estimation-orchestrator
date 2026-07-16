# Architecture and dependency direction

The orchestrator depends on estimator contracts, never estimator implementations.

```text
orchestrator → subprocess CLI → pinned pure-PF checkout
             → subprocess CLI → pinned surface-MLE checkout

MeasurementLog → adapter input
PFResult/MLEResult → validation → evaluation
truth.json ────────────────────────┘  (evaluation phase only)
```

There is no Python import edge from `orchestrator` to `pf`,
`three_d_estimation`, detector/transport kernels, or runtime measurement packages.
The default commands are token arrays and run with `shell=False`. A reduced environment
allowlist is supplied to children; credentials are not propagated.

The benchmark runs in a fixed refuse-replace staging directory. Publication is one
atomic directory rename after input validation, all subprocesses, result validation,
truth-gated metrics, and manifest writing complete.

Dirty estimator checkouts are safe only when every changed path is beneath an explicit
artifact prefix such as `results/` or `logs/`. Dirty `src/`, tests, configs, or entry
points fail revision verification. The complete dirty inventory and file hashes are
recorded in every execution entry.

The benchmark production policy cannot be disabled in JSON: both exact revision
verification and clean-checkout enforcement must be true. Adapter command overrides
are rejected, so the pinned repository's public CLI remains the executable boundary.
Configured dirty prefixes may narrow, but never broaden, the fixed artifact/cache
allowlist.
