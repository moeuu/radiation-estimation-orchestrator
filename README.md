# Rotating Shield Estimation Orchestrator

This repository is the reproducible ownership boundary between the pure online
particle filter in `moeuu/Rotating-shield-particle-filter` and the standalone
all-history surface MLE in `moeuu/3D_estimation`.

The repository now exposes two controlled v1 paths. The original same-log benchmark
remains unchanged, and a separate causal offline hybrid replay adds prefix MLE
proposals without changing either pure estimator entry point:

```text
truth-free MeasurementLog v1
    ├── pure PF sequential replay
    ├── count-domain surface MLE replay
    └── spectral-domain surface MLE replay
             ↓
    validate contracts → open separate truth → common evaluation → manifest

station-complete prefix 1:t
    ├── pure PF replay and frozen predictions
    └── count MLE → fixed-K MH position proposal
                         ↓
              future-only verification/quarantine
                         ↓
              non-actuating DSS-PP recommendation

complete log → cold spectral MLE → authoritative hybrid report
```

Estimator and physics source is not copied here. Every estimator is invoked through a
subprocess CLI at the exact commit in [`PINNED_ESTIMATORS.json`](PINNED_ESTIMATORS.json).
The orchestrator records commands, commits, input/config/output hashes, stdout/stderr
hashes, runtime, peak memory, and the complete artifact inventory.

## Current scope

Implemented:

- canonical, filesystem-aware MeasurementLog v1 validation;
- pure PFResult and standalone MLEResult v1 validation;
- exact-prefix MLESnapshot v2 with cutoff, covered-record, warm-start, and artifact
  lineage;
- pinned Git revision checks that reject dirty code while inventorying explicitly
  allowed result/cache paths;
- a non-bypassable production adapter policy: exact revision verification and clean
  checkout checks are mandatory, command overrides are forbidden, and dirty-prefix
  configuration may only narrow the built-in artifact allowlist;
- safe shell-free PF and MLE subprocess adapters;
- the fixed benchmark pipeline (PF, count MLE, spectral MLE);
- all requested point-source, operational, and surface-MLE metrics;
- truth isolation: truth is outside the log and first opened after all three result
  contracts validate;
- independent PF/MLE forward-response conformance at `rtol=1e-9`, `atol=1e-12`,
  with exact 40-character commit pins, mandatory clean-checkout verification, and
  command/config/stdout/stderr/artifact provenance;
- exact cross-repository line-model identity (energy, normalized weight, and
  line-specific Fe/Pb attenuation) bound into the forward manifest;
- a deterministic 12-record, three-isotope, multi-height contract smoke fixture.
- station-complete periodic/triggered count MLE with initialization-only warm starts;
- MLE-shaped, fixed-cardinality, target-preserving PF position relocation using full
  PF target and forward/reverse proposal correction;
- future-only frozen-model log-predictive-ratio verification and non-destructive
  quarantine;
- planner-only pending/verified external modes and a collision/reachability-attested,
  non-actuating DSS-PP XYZ/height/shield-program recommendation;
- a hash-chained observation-use ledger, aggregate per-candidate MH receipts, and
  execution evidence;
- independent cold full-log count diagnostics and converged cold spectral MLE as the
  authoritative hybrid result.

Deliberately not implemented are reversible-jump birth/death, MLE-directed hard
pruning, direct MLE-objective reweighting, or live robot actuation. See
[`docs/hybrid_v1.md`](docs/hybrid_v1.md) and
[`docs/future_hybrid.md`](docs/future_hybrid.md).

## Install and verify

Python 3.12+ and `uv` are required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run rotating-shield-orchestrator validate-log \
  --run-dir fixtures/shared_small_run/measurement_log
```

## Run the shared benchmark

The estimator checkouts must be at the pinned commits. The default config resolves the
local sibling paths from `PINNED_ESTIMATORS.json`.

The bundled `shared_small_run` is for contract, isolation, replay, and artifact-pipeline
testing. Its isotope counts are hand-authored and its spectra only distribute those
counts over the production line basis; they were not generated from the accompanying
truth through the production transport model. Do not interpret accuracy metrics from
this smoke fixture scientifically. Use a separately provenance-bound real-observation
or Geant4 MeasurementLog for scientific estimator comparisons.

```bash
uv run rotating-shield-orchestrator verify-pins
uv run rotating-shield-orchestrator benchmark \
  --config configs/benchmark/shared_small_run.json
```

The output directory contains:

```text
benchmark_manifest.json
benchmark_manifest.sha256
metrics.json
executions/{pf_strict,mle_count,mle_spectral}/{stdout.log,stderr.log}
results/pf_strict/{pf_posterior.json,pf_trace.jsonl,pf_diagnostics.json}
results/mle_count/{mle_estimate.npz,mle_diagnostics.json,hotspot_clusters.json}
results/mle_spectral/{mle_estimate.npz,mle_diagnostics.json,hotspot_clusters.json}
```

Outputs and staging directories are refuse-replace. A failed benchmark removes its
private staging directory and never publishes a partial manifest.

## Run causal hybrid replay

The shared hybrid smoke config invokes the pinned PF and MLE checkouts through their
real subprocess CLIs. Its planning request uses poses already realized in the fixture;
the environment artifact and ordered candidate set are hash-attested. The result is an
algorithmic recommendation only and explicitly cannot authorize robot actuation.

```bash
uv run rotating-shield-orchestrator verify-pins
uv run rotating-shield-orchestrator hybrid-replay \
  --config configs/hybrid/shared_small_run.json
```

The output includes exact-prefix logs, snapshots, directives, aggregate PF receipts,
future candidate scores, the verification queue, planning request/recommendation,
execution evidence, final cold MLE bundles, `hybrid_result.json`, and the run manifest.

## Contracts

The public JSON Schemas are packaged under [`src/orchestrator/contracts`](src/orchestrator/contracts).
Validation also covers NPZ dtype/shape/masks, causal row order, finite values,
quaternion normalization, covariance symmetry/PSD, metadata alignment, content hashes,
line-table identity, deterministic PF cardinality argmax, MLE scalar/patch/cluster
consistency, and result artifact mirrors—properties JSON Schema alone cannot prove.

MeasurementLog truth is deliberately absent. The shared fixture stores it at
[`fixtures/shared_small_run/evaluation/truth.json`](fixtures/shared_small_run/evaluation/truth.json),
outside the estimator input directory.
The validator recursively rejects realized-truth/source-layout pointers in runtime,
environment, run, and observation metadata, and `source_layout_path` is required to be
null. Source-rate and source-extent model semantics remain valid estimator inputs.

See [`docs/contracts.md`](docs/contracts.md) for exact arrays, hashing, and provenance.

## Forward-response conformance

[`fixtures/forward_response_conformance.json`](fixtures/forward_response_conformance.json)
covers Cs-137, Co-60, Eu-154, three detector poses, all 64 Fe/Pb orientation pairs,
floor/wall/ceiling/obstacle-top sources, and no-obstacle/one-box environments. Each
estimator owns a CLI that emits the neutral `case_ids` + `unit_response` NPZ contract.
The orchestrator compares the independent outputs; it contains no response physics.

See [`docs/forward_model_conformance.md`](docs/forward_model_conformance.md).

Run both production providers from the orchestrator root with:

```bash
uv run rotating-shield-orchestrator conformance \
  --fixture fixtures/forward_response_conformance.json \
  --pf-provider configs/conformance/pf_production.json \
  --mle-provider configs/conformance/mle_production.json \
  --output-dir conformance-results/production-v1
```

The provider JSON files pin the same estimator commits as
[`PINNED_ESTIMATORS.json`](PINNED_ESTIMATORS.json). Update both locations together
when advancing an estimator baseline. Production conformance rejects a missing,
abbreviated, non-commit, mismatched, or dirty provider checkout before execution.

## Repository map

```text
src/orchestrator/
  adapters/       pinned subprocess boundaries
  contracts/      JSON Schemas and filesystem/NPZ validators
  benchmark.py    fixed same-log execution pipeline
  evaluation.py   truth-gated common metrics
  conformance.py  independent forward-response comparison
  manifests.py    provenance and artifact hashing
configs/          benchmark, estimator, and scenario definitions
fixtures/         truth-free shared log, separate truth, conformance axes
tests/            contracts, pins, same-log run, isolation, metrics, conformance
```
