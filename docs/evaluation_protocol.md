# Evaluation protocol

Estimator quality and active-measurement quality must be separated.

## Same-log estimator comparison

Use one MeasurementLog v2 and run only:

1. strict raw-spectrum PF without batch refinement;
2. standalone spectral surface MLE.

The v2 benchmark fixes detector XYZ, heights, shield programs, dwell, spectra,
environment, and random realization. This isolates estimator differences. Archived
count MLE and historical mixed PF results belong to separately labelled v1 runs.

## Closed-loop system comparison

Use new, untuned Geant4 environments and compare at equal total mission time:

- strict PF with its active planner;
- spectral MLE using fixed or space-filling 3-D acquisition;
- hybrid verification only;
- hybrid exact RJ.

Report live, travel, height-change, shield-actuation, and estimator compute time
separately. A fixed number of steps is not an equal-cost comparison.

## Required scenarios

- single floor, wall, and ceiling source;
- same XY with different source heights;
- nearby same-isotope sources;
- strong plus weak source;
- multiple isotopes;
- obstacle-free and obstacle-attenuated scenes;
- high count, background drift, and model-mismatch stress cases.

## Required outcomes

- 3-D, XY, and Z error;
- cardinality exact match, precision, recall, and false source count;
- strength error and ceiling-source recall;
- hotspot centroid, surface-kind, integrated mass, and held-out deviance;
- detector-height distribution and source-elevation diversity;
- measurement count and all mission-time components;
- runtime, peak memory, snapshot stability, and verification outcomes.

Tune thresholds and regularization only on designated training scenes. Final claims use
new random holdout scenes. The hand-authored fixture is for contracts, never for
scientific accuracy claims.
