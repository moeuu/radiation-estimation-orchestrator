# Repository instructions

This repository owns reproducible cross-estimator contracts and benchmarking. It is
also the designated home of a later PF+MLE hybrid milestone, but no hybrid feedback
controller is implemented in the current active path. Keep these boundaries invariant.

- Never copy particle-filter, MLE, detector, shield, obstacle, spectrum, transport,
  simulation, environment, observation-generation, or MeasurementLog writer source
  into this repository. Invoke pinned estimator repositories through subprocess CLI
  adapters and invoke acquisition through the shared simulation-runtime package.
- The current milestone is pure comparison only. Do not add MLE-to-PF feedback,
  rejuvenation, pruning, periodic MLE execution, or hybrid planning without a new
  versioned milestone and explicit tests.
- Estimator execution may read only the estimator-independent MeasurementLog bundle.
  Truth lives outside that bundle and is opened only after every estimator subprocess
  has completed, inside evaluation code.
- Every external command, estimator revision, resolved configuration hash, input hash,
  stdout/stderr hash, and output-file hash belongs in the benchmark manifest.
- Any future MLE-derived PF directive will have to carry `data_cutoff_step` and
  `data_cutoff_station`. A consumer must apply a snapshot once and process only later
  observations afterward. Extend the observation-use ledger tests when changing this
  boundary. The current repository reserves only the MLESnapshot contract.
- Contract changes require a schema-version change or a backwards-compatible schema
  extension plus conformance tests.
- Use deterministic JSON/NPZ serialization for fixtures and generated manifests.
