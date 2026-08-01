# Architecture and dependency direction

The orchestrator depends on estimator contracts, never estimator implementations.
Production simulation has one owner:

```text
Rotating-shield-particle-filter
  Geant4 + environment + detector/shield + action execution
                         │
                         ▼
             raw MeasurementLog v2
                ┌────────┼────────┐
                ▼        ▼        ▼
               PF       MLE     PF+MLE
```

This boundary removes simulator synchronization work. The estimator repositories may
change inference code independently, but they may not regenerate, reinterpret, or
silently project observations before consuming the shared log.

For a same-observation comparison, acquisition runs once and every estimator replays
the identical finalized log. For estimator-controlled planning, PF, MLE, and PF+MLE
may select different next actions, so their future observations cannot be identical.
Those missions share the same simulator implementation, resolved physical contract,
environment-generation rules, and seed policy, but each remains a separate causal run.

```text
orchestrator → subprocess CLI → pinned pure-PF checkout
             → subprocess CLI → pinned surface-MLE checkout

MeasurementLog → adapter input
PFResult/MLEResult → validation → evaluation
truth.json ────────────────────────┘  (evaluation phase only)
```

The separately versioned hybrid v1 path uses the same dependency boundary:

```text
MeasurementLog prefix ─► pinned PF replay ─► predictive record / PF receipt
                      └► pinned count MLE ─► MLESnapshot ─► PFDirective

future-only rows ─► frozen snapshot score ─► verification/quarantine
PF + pending/verified modes + attested poses ─► non-actuating DSS-PP recommendation

complete MeasurementLog ─► cold count MLE     (diagnostic)
                        └► cold spectral MLE  (authoritative final report)
```

The orchestrator owns scheduling, exact-prefix materialization, contracts, the
verification queue, and the observation-use ledger. The PF repository owns the
target-preserving fixed-cardinality MH kernel; the MLE repository owns every surface
fit and response calculation. No estimator physics is reimplemented here. See
[`hybrid_v1.md`](hybrid_v1.md) for causality and limitation details.

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

The pure benchmark and hybrid replay are independent active paths. Hybrid directives
cannot enter the pure PF command, and prefix warm starts cannot change the standalone
MLE objective or complete-surface candidate domain.

The default estimator registry pins the current MeasurementLog-v2 consumers. Archived
v1 benchmark/hybrid configs name a separate immutable v1 registry. Schema selection is
therefore explicit in the run config rather than inferred from whatever sibling
checkout happens to be present.
