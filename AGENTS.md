# Repository instructions

This repository owns the PF, surface-MLE, reproducible estimator contracts, benchmarking,
and the versioned PF+MLE hybrid controller. The shared simulation-runtime package is the
only external research-code dependency. Keep the ownership boundaries below invariant.

- Use the shared simulation-runtime package for environment/obstacle geometry, detector,
  shield, spectrum and transport physics, observation generation, and MeasurementLog
  reading/writing. Do not copy those physical models into this repository.
- Implement particle state, PF likelihood/update/resampling, PF checkpoints, planning,
  spectral surface-MLE, future scoring, exact reversible jump, reporting, and resumable
  orchestration in this repository.
- Active estimator and hybrid paths must not invoke or import sibling PF or MLE
  repositories. Historical external-adapter artifacts may remain only as explicitly
  unsupported archive code and must not be reachable from production commands.
- Keep historical count-domain `hybrid-v1` contracts readable for artifact auditing,
  but do not expose their external-estimator execution path through production commands.
  New work belongs to the explicitly versioned raw-spectrum `hybrid-v2` milestone and
  must have causal-prefix, future-only observation-use, resume, and contract tests.
- `hybrid-v2` may schedule spectral prefix MLE, pass proposal-only directives to the PF,
  and perform closed-loop acquisition through the shared runtime. It must not add an MLE
  objective directly to PF weights or hard-prune a PF mode. Dimension-changing changes
  must be target-preserving PF-owned reversible-jump proposals.
- Estimator execution may read only the estimator-independent MeasurementLog bundle.
  Truth lives outside that bundle and is opened only after every estimator has completed,
  inside evaluation code.
- Record the shared-runtime revision, orchestrator revision, resolved configuration hash,
  input hash, deterministic random seed, and output-file hashes in run manifests.
- Every MLE-derived PF directive carries `data_cutoff_step`, `data_cutoff_station`, and
  the exact prefix identity. A consumer applies a snapshot once and uses only later
  observations for corroboration. Extend the observation-use ledger tests whenever this
  boundary changes.
- Contract changes require a schema-version change or a backwards-compatible schema
  extension plus conformance tests.
- Use deterministic JSON/NPZ serialization for fixtures and generated manifests.
