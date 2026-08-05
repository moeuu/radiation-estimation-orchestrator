# Architecture and ownership

## Dependency direction

`Rotating-shield-simulation-runtime` is the only external research-code dependency.
It publishes immutable physics objects and truth-free observations. All statistical
inference and control after observation generation is implemented here.

```text
simulation runtime
  environment / obstacles / detector / shield / transport / spectrum
  action safety / actuation / observations / MeasurementLog v2
                              │
                              ▼
orchestrator.estimators.context + forward
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       ParticleFilter      SpectralMLE     action prediction
             │                │                │
             ├─ checkpoint    ├─ snapshot      └─ checkpoint planning
             ├─ exact RJ      └─ future scoring
             └────────────────┬────────────────┘
                              ▼
                    hybrid-v2 controller
```

Allowed dependency direction:

```text
hybrid_v2 -> local estimator services -> PF/MLE/RJ/planning
PF/MLE/RJ/planning -> authenticated runtime physics
live controller -> runtime protocol-v2 client
```

Forbidden directions:

- active code to `pf` or `three_d_estimation` packages;
- subprocess calls into sibling estimator repositories;
- copied detector, shield, obstacle, transport, spectrum, simulator, or
  MeasurementLog-writer implementations;
- truth flowing into inference, planning, or acquisition decisions.

## Physical context

`estimators/context.py` validates MeasurementLog v2, reconstructs the exact runtime
observation model from its resolved runtime configuration and model manifests, builds
runtime-owned surface charts, and checks the recorded energy axis. `estimators/forward.py`
then provides batched full-spectrum predictions for recorded rows and hypothetical
actions. PF, MLE, RJ, future scoring, and planning all use this one context.

The runtime remains authoritative for:

- line-resolved source transport and attenuation;
- detector response, additive scatter, background, and dead time;
- source-strength semantics;
- surface geometry and obstacle transport;
- collision/reachability/path attestation and physical action execution.

## Local estimators

The strict PF owns particle position charts, strengths, cardinality, weights, RNG state,
station-block likelihood updates, resampling, posterior summaries, pre-update spectral
predictions, and deterministic checkpoints. It has no batch-MLE call path.

The spectral MLE owns the complete-surface response matrix, nonnegative Poisson
optimization, L1 and graph-TV regularization, warm starts, surface density maps, and
hotspot clusters. Its API has no PF state or PF candidate argument.

The exact-RJ kernel is PF-owned. A verified MLE region defines a proposal distribution,
not an extra likelihood. Birth and death are paired; boundary move probabilities,
forward/reverse densities, the current full PF target, dimension matching, Jacobian,
draw, and decision are recorded in the receipt.

## Durable hybrid execution

Every hybrid artifact binds the source run, exact cutoff, covered step IDs, neutral
covered-record hash, resolved estimator configuration, and relevant input/output hashes.
The observation-use ledger rejects reused verification rows and pre-cutoff evidence.

The live state machine persists proposed, realized, appended, and estimator-updated
events separately. A runtime decision ID is exactly-once and receipt-queryable. After a
restart, an existing runtime receipt is recovered, the durable MeasurementLog prefix is
revalidated, and incomplete local inference is recomputed in a private output directory.

Planning receives only runtime-attested candidates. It may combine PF particle
hypotheses with pending/verified MLE regions at controlled mass, excludes quarantined
regions, scores full-spectrum separation and operational cost, and proves that the PF
checkpoint hash is unchanged.

## Active versus historical code

The public CLI exposes only runtime acquisition, local PF/MLE/scoring/RJ/planning,
benchmark v2, and hybrid v2. Historical v1 contracts remain parsable so old result
bundles can be audited, but v1 external-estimator execution is unsupported and is not a
production dependency.
