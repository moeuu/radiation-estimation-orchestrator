# Benchmark and metrics

The active benchmark is schema v2:

```text
validate MeasurementLog v2
  -> local strict PF replay
  -> local spectral surface MLE
  -> validate both result bundles
  -> open separate truth
  -> common metrics
  -> hash-rich manifest
```

Both estimators use the same raw spectra, poses, shield states, live times, runtime
configuration, and physical forward model. No estimator subprocess or sibling checkout
is involved. The manifest identifies this repository's commit, dirty source snapshot,
runtime model/log identities, raw and resolved estimator configuration hashes, random
seed, execution durations, and every output-file hash.

Point-source metrics use isotope-aware gated assignment and report 3-D, XY, and Z
position error, exact cardinality, precision/recall, integrated strength error, and
ceiling-source recall. Unmatched truth and estimate mass is retained explicitly.

Operational metrics come only from the MeasurementLog: measurement/action count,
detector-height distribution, travel time, shield-actuation time, live time, estimator
runtime, and memory where measurable.

Surface-MLE metrics include hotspot centroid/strength error, surface-kind accuracy,
recovered mass near truth, and held-out deviance when available.

Truth must be outside the MeasurementLog directory. The benchmark opens it only after
PF and MLE result validation. Contract fixtures prove dataflow and serialization; their
accuracy metrics are not scientific results. Scientific comparisons require new,
provenance-bound runtime/Geant4 logs and equal observation/time budgets.

Historical schema-v1 result contracts remain validators for old artifacts. The active
benchmark CLI rejects v1 configs because they depended on external estimator checkouts.
