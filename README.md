# Radiation Estimation Orchestrator

This repository is the complete inference implementation for rotating-shield radiation
estimation. It owns:

- the strict sequential full-spectrum particle filter;
- the all-surface spectral MLE;
- causal checkpoint planning;
- future-only spectral candidate scoring;
- paired birth/death exact reversible-jump moves;
- offline PF+MLE replay and resumable closed-loop hybrid execution;
- estimator contracts, benchmarking, evaluation, and provenance.

The only external research-code dependency is
`Rotating-shield-simulation-runtime`. That package owns environment and obstacle
geometry, detector/shield/transport/spectral physics, observation generation,
collision/reachability checks, action execution, and MeasurementLog v2 production.
This repository never imports or invokes the sibling PF or MLE repositories.

```text
shared runtime
  ├─ immutable physical forward model
  ├─ safe 3-D action candidates and execution receipts
  └─ truth-free MeasurementLog v2
                 │
                 ▼
this repository
  ├─ strict PF ────────────────┐
  ├─ spectral surface MLE      │
  ├─ future spectral scoring   ├─ checkpoint planner
  ├─ exact paired RJ kernel    │
  └─ causal hybrid controller ─┘
                 │
                 ▼
      final cold spectral MLE report
```

## Implemented execution paths

Pure same-log comparison runs the local strict PF and local spectral MLE on the
same finalized MeasurementLog. Truth remains outside the log and is opened only after
both result bundles validate.

Offline hybrid replay processes station-complete prefixes. It checkpoints the PF,
runs prefix spectral MLE when scheduled, scores candidate regions only on later raw
spectra, verifies or quarantines them by independent station/height/shield blocks,
and optionally applies a PF-owned exact RJ transition.

Live hybrid execution is a durable protocol-v2 state machine:

```text
ready -> action_proposed -> action_realized -> observation_appended
      -> estimators_updated -> ready ... -> complete
```

The runtime authorizes and executes an action exactly once. The orchestrator binds the
decision to a PF checkpoint, the exact observation prefix, and the runtime-attested
candidate/path hashes. Restarting after a crash recovers receipts and completed local
estimator operations rather than repeating physical actions.

The authoritative hybrid result is a cold, converged, full-log spectral surface MLE.
MLE objective values are never added to PF weights, and MLE-directed hard pruning is
forbidden.

## Install and verify

Python 3.12+ and `uv` are required. The sibling runtime checkout is configured only as
the package source in `pyproject.toml` and pinned by `PINNED_RUNTIME.json`.

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run radiation-estimation-orchestrator --help
```

## Commands

Acquire a truth-free log through the shared runtime:

```bash
uv run radiation-estimation-orchestrator acquire --plan PRIVATE_PLAN.json
```

Run local estimators directly:

```bash
uv run radiation-estimation-orchestrator pf-checkpoint \
  --measurement-log RUN/measurement-log \
  --config configs/estimators/pf_strict_shared_small.json \
  --output-dir RESULTS/pf --seed 7

uv run radiation-estimation-orchestrator spectral-mle \
  --measurement-log RUN/measurement-log \
  --config configs/estimators/mle_spectral_shared_small.json \
  --output-dir RESULTS/mle
```

The `pf-checkpoint` command accepts `--checkpoint-in` for prefix continuation. Related
standalone commands are:

- `checkpoint-plan`: rank runtime-attested XYZ/shield actions without changing PF state;
- `future-spectral-score`: score frozen MLE candidates on post-cutoff spectra;
- `exact-rj`: apply one target-corrected birth/death transition to a PF checkpoint.

Run the versioned hybrid paths:

```bash
uv run radiation-estimation-orchestrator hybrid-v2-replay --config OFFLINE.json
uv run radiation-estimation-orchestrator hybrid-v2-live \
  --config configs/hybrid_v2/live_example.json
```

After a live mission completes, evaluation may open the separate truth artifact:

```bash
uv run radiation-estimation-orchestrator evaluate-live \
  --manifest RESULTS/live_hybrid_manifest.json \
  --truth EVALUATION_TRUTH.json \
  --output RESULTS/evaluation_metrics.json
```

## Statistical and causal boundaries

- The PF consumes raw spectra sequentially in complete station blocks. Its result is
  built only from the particle posterior; it does not call the MLE.
- The MLE always optimizes the complete runtime surface dictionary. It cannot accept PF
  particles or PF-selected candidate positions.
- A prefix MLE snapshot identifies its exact cutoff and covered-record hash.
- Candidate verification uses only steps strictly after that cutoff, once each.
- Exact RJ uses the current processed-prefix PF target, paired forward/reverse proposal
  densities, dimension matching, and an explicit acceptance receipt.
- Planning evaluates PF particles plus non-quarantined MLE hypotheses but never mutates
  the checkpoint.
- Truth is not a valid inference input.

See [architecture](docs/architecture.md), [hybrid v2](docs/hybrid_v2.md),
[contracts](docs/contracts.md), and the [evaluation protocol](docs/evaluation_protocol.md).

## Current limitations

- The spectral MLE materializes its response columns in memory; large patch/bin/action
  studies will need a matrix-free or disk-backed operator.
- Surface charts are those published by the shared runtime (currently room and box
  surfaces); arbitrary mesh/volume source dictionaries are not yet implemented here.
- Exact RJ currently changes source cardinality one particle and one source at a time.
- Verification thresholds require calibration on independent station-block holdouts and
  new Geant4/real-system runs; smoke fixtures are contract tests, not accuracy evidence.

Historical MeasurementLog-v1 schemas and adapter modules remain readable for artifact
validation only. They are not exposed by the production CLI and are not part of the
runtime-only inference architecture.
