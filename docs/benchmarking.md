# Benchmark and metrics

The command performs these steps in order: validate log, pure PF replay, count MLE,
spectral MLE, validate all results, open truth, calculate metrics, write manifest.

Shared point-source metrics are gated isotope-aware maximum-cardinality,
minimum-distance assignments. Reports include 3-D/XY/Z errors, exact source
cardinality, precision/recall, integrated strength error, and ceiling-source recall.
Strength reports include per-isotope totals, matched-pair errors, every unmatched truth
and estimated source as full missing/extra mass, and a complete assignment error.
Consequently, equal-and-opposite errors in different isotope channels cannot cancel.

Operational metrics are derived from MeasurementLog facts: measurement count, unique
XYZ/action counts, detector-height distribution, travel time, shield actuation time,
live time, estimator wall runtime, and subprocess-tree peak RSS.

Surface-MLE metrics include cluster centroid and strength errors, matched surface-kind
accuracy, reconstructed patch mass within the truth match radius, and held-out Poisson
deviance.

The manifest contains exact external commands, pinned/requested/observed commits,
allowed dirty inventories, all input/result/log hashes, package versions, and a sidecar
hash for the manifest itself. Estimator configuration provenance is split deliberately:
`estimator_config_file_sha256` hashes the raw files passed on the command line, while
`expected_resolved_estimator_config_sha256` is fixed independently in the benchmark
configuration for every run. Each PF/MLE result must match that expectation after
defaults and profile enforcement; only then is the observed
`resolved_estimator_config_sha256` recorded. The raw, expected-resolved, and
observed-resolved fields are never treated as aliases.

Execution stdout/stderr paths in the manifest are relative to the published benchmark
root. They remain valid after the staging directory is atomically renamed.
