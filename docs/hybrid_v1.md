# Causal PF+MLE hybrid v1 (archived)

This document describes the historical projected-count/external-estimator milestone.
It is retained only to interpret old artifacts. The production CLI no longer exposes
this execution path; hybrid v2 owns local PF/MLE/RJ/planning and depends only on the
shared runtime.

Hybrid v1 adds a separately versioned orchestration path without changing either pure
baseline. The existing same-log benchmark remains the reference comparison path.
Estimator and response-physics code stays in the pinned PF and MLE repositories and is
invoked only through subprocess contracts.

## Scientific roles

```text
station-complete observations 1:t
        │
        ├─ sequential PF replay ──► online posterior, uncertainty, predictions
        │                              │
        │                              └─ fixed-K position proposal consumer
        │
        └─ count surface MLE on exact prefix 1:t
                  │
                  └─ MLESnapshot v2 ──► once-only PFDirective
                                              │
                         later observations t+1:… only
                                              │
                                      verification/quarantine
                                              │
                         pending/verified planner-only modes
                                              │
                         non-actuating DSS-PP recommendation

complete MeasurementLog
        ├─ final cold count MLE       (diagnostic)
        └─ final cold spectral MLE    (authoritative report and surface map)
```

The PF remains responsible for the online history, uncertainty, predictive monitoring,
and fixed-cardinality posterior state. Periodic count MLE fits use only a declared,
station-complete prefix. A previous MLE result may initialize the next optimizer, but
the complete current-prefix objective is always recomputed; a warm start is not a
partial-data update or an inherited likelihood.

An MLE cluster is candidate metadata, not an extra observation. In
`proposal_only_mh`, it defines a defensive truncated-Gaussian position proposal for an
existing PF source slot. The PF evaluates the complete target through the directive
cutoff and applies forward/reverse proposal-density correction. Particle weights are
not multiplied by the MLE objective. The move neither adds nor removes a source, so the
PF cardinality is unchanged.

Proposal means must lie in the PF/MeasurementLog physical domain. Surface centroids
that differ from a declared room boundary only by floating-point roundoff are clipped
to that boundary before the proposal density is defined; genuine domain mismatches are
excluded rather than silently projected.

Each snapshot, directive, PF receipt, and corroboration event is recorded in a
hash-chained observation-use ledger. A directive is applied at most once immediately
after its declared cutoff. Candidate verification can consume only observations with
`step_id > data_cutoff_step`; it cannot reuse prefix observations as another PF
likelihood. Verification and quarantine are evidence states and do not hard-prune PF
particles.

Future verification is a frozen count-model comparison: the snapshot's full model is
compared with the same model after setting one candidate cluster to zero. No parameter
is refit on future rows. The accumulated quantity is a log predictive likelihood
ratio, not a Bayes factor. Related shield views can be correlated, so scientific
thresholds must be calibrated at a station or shield-program-block level.

At configured boundaries, pending and verified MLE candidates may be exposed as
planner-only DSS-PP modes. Quarantined candidates are excluded and cannot affect the
normalization of included modes. The controller accepts only an ordered candidate set
with collision/reachability attestation, then emits an XYZ/height/shield-program
recommendation. It does not mutate PF state or authorize actuation.

At the end of a run, both standalone MLE modes are rerun cold over the complete log.
The count result is a diagnostic. The cold spectral MLE over the complete surface
dictionary is the authoritative final position, intensity-map, hotspot-cluster, and
source-count report. It accepts neither PF state nor PF candidate support.

## Result contract

`hybrid_result.json` is the canonical v1 result manifest. It binds:

- both the user-supplied truth-free MeasurementLog and the inference-log derivative
  containing only predeclared station-boundary completion markers;
- the unchanged source run ID and declared station-boundary schedule;
- final PF, cold count-MLE, cold spectral-MLE, and ledger result hashes;
- every snapshot, directive, and receipt identity and content hash;
- every frozen future-score artifact and its exact ledger corroboration rows;
- every planning request/recommendation hash, causal cutoff, candidate attestation,
  selected action, and `robot_actuation_authorized=false` assertion;
- the immutable execution-evidence hash covering estimator commands and outputs;
- pending, verified, and quarantined candidate counts;
- the authoritative spectral-MLE hotspot clusters;
- causality, no-reweight, no-prune, truth-isolation, and pure-baseline assertions;
- the explicit v1 limitations below.

The schema is packaged as `hybrid_result_schema.json`. Filesystem-aware validation
cross-checks reference IDs, cutoffs, once-only receipts, verification counts, ledger
identity, complete cold-MLE lineage, and the authoritative clusters against the
validated spectral MLE bundle. Canonical serialization is accompanied by
`hybrid_result.sha256`.

## Deliberate v1 limitations

Hybrid v1 does **not** implement:

- reversible-jump birth/death or any MLE-driven cardinality change;
- MLE-directed hard pruning;
- direct reweighting by an MLE objective or evidence score;
- arbitrary uncorrected rescue-particle injection;
- live closed-loop execution of a PF/MLE conflict planner.

The current controller is an offline, deterministic replay orchestrator. Planning
signals may be reported for later study, but v1 does not actuate a new detector pose,
height, or shield program during a live measurement run. A future live or
trans-dimensional milestone requires a new contract version and, for birth/death,
an explicit reversible-jump or equivalent target-preserving correction.

The current collision/reachability boundary validates an upstream attestation and its
candidate/environment hashes; it does not recreate `MeasurementWorkspace` or validate
a realized robot path inside the orchestrator. This is safe only because the artifact
is recommendation-only. Live control requires a workspace-generated path artifact and
independent environment/path validation.

## Preserved baselines

The public pure-PF replay still consumes only the MeasurementLog and emits a PF
posterior result. The public standalone MLE paths still consume only the MeasurementLog
and complete surface dictionary. The pre-existing benchmark still runs pure PF, count
MLE, and spectral MLE independently on one log. Hybrid-only snapshots, directives, and
receipts are carried by separate entry points and contracts.
