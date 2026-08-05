# Raw-spectrum PF–MLE hybrid v2

Hybrid v2 is the active PF+MLE method. The shared runtime generates observations and
owns the forward physics; this repository owns every inference and planning operation.

## Roles

| Concern | Owner |
| --- | --- |
| Environment, detector, shield, transport, spectrum, observations | shared runtime |
| Collision/reachability/path attestation and action receipts | shared runtime |
| Sequential posterior, checkpoints, cardinality, complete PF target | this repository |
| Surface spectral MLE, L1/TV map, clusters, warm starts | this repository |
| Future spectral scoring, verification queue, exact RJ | this repository |
| Checkpoint planning, scheduling, mission state, manifests | this repository |

No sibling PF or MLE repository is required.

## Offline causal replay

At each complete station:

1. Materialize and validate the exact MeasurementLog prefix.
2. Resume the strict PF checkpoint and save its pre-update spectral prediction.
3. Evaluate predictive mismatch using only a prediction made before each observation.
4. Run a prefix spectral MLE when the interval/mismatch scheduler triggers.
5. Register snapshot clusters as proposal candidates.
6. Score frozen full-versus-candidate-zero spectra only on later, unconsumed rows.
7. Aggregate evidence by station, height group, and shield-program block.
8. Verify or quarantine candidates according to configured independent-block counts.
9. In exact-RJ mode, issue a proposal-only directive against the current PF checkpoint.

The final authoritative output is a cold, converged spectral MLE over the complete log.

## Resumable live protocol

`hybrid-v2-live` connects the same inference pipeline to a runtime adaptive session:

```text
ready
  -> action_proposed
  -> action_realized
  -> observation_appended
  -> estimators_updated
  -> ready ... -> complete
```

The runtime publishes a candidate snapshot with collision, reachability, travel-cost,
and path hashes. The local checkpoint planner selects detector XYZ and a shield pair.
The controller sends a decision containing the exact candidate/path identity. Runtime
protocol v2 executes it exactly once and supports receipt lookup by decision ID.

A crash at any boundary is recoverable:

- an already realized action is found by receipt lookup, not executed again;
- the appended MeasurementLog prefix is revalidated before inference;
- an authenticated completed local estimator operation is reused;
- an incomplete private estimator output is removed and deterministically recomputed;
- the hash-chained mission ledger repairs a lagging state file.

Changing the live config or runtime scenario after mission creation changes the resume
identity and is rejected.

## Future spectral scoring

A spectral snapshot is immutable and includes its exact data cutoff. For each candidate,
the scorer compares:

```text
frozen complete snapshot model
vs.
the same frozen model with that candidate cluster set to zero
```

It computes raw-spectrum Poisson log-likelihood ratios on steps strictly greater than
the snapshot cutoff. It never refits either model. Each row is used once and belongs to
one independent verification block. Promotion may require distinct stations, detector
heights, and shield programs; this prevents many correlated views at one station from
masquerading as independent support.

If the runtime supplies `metadata.shield_program_id`, that identifier defines the
block. Current protocol-v2 runtime logs need not supply it, so the scorer otherwise
derives a deterministic program ID from the ordered measured Fe/Pb orientation pairs
within the station/height group. A group that mixes present and absent explicit IDs is
rejected instead of being guessed.

## Exact reversible jump

Exact-RJ uses two cutoffs:

- `proposal_data_cutoff_*`: the older prefix that produced the MLE region;
- `data_cutoff_*`: the current PF target after future verification.

The directive binds the current log prefix, covered-record hash, PF checkpoint, input
state hash, verified regions, and dimension-matching transform. The PF kernel then:

1. determines whether birth, death, or both are reversible at the current state;
2. samples a region-conditioned surface chart and bounded log-strength auxiliary for
   birth, or a reverse-supported final active slot for death;
3. evaluates the complete processed-prefix PF target before and after the move;
4. evaluates marginal forward and reverse proposal densities and move probabilities;
5. includes the Jacobian term and performs Metropolis–Hastings acceptance;
6. writes a receipt whose acceptance arithmetic and state transition are independently
   validated.

The MLE objective is not added to PF weights. Rejected moves leave the deterministic PF
state bytes unchanged. There is no MLE-directed hard prune.

## Checkpoint planning

Planning operates on a validated PF checkpoint plus a runtime-attested ordered candidate
set. It evaluates every candidate XYZ/shield-pair combination using the same runtime
spectral forward model. The score combines posterior spectral separation, expected
counts, vertical diversity, and travel cost. Pending candidates have less configurable
mass than verified candidates; quarantined candidates are excluded.

Planning is read-only. The recommendation records identical before/after PF state hashes
and cannot authorize actuation by itself; only the live controller and runtime receipt
protocol can do that.

## Causal invariants

1. A prediction cutoff is earlier than the observation it scores.
2. A snapshot covers exactly its declared prefix.
3. Verification uses only later rows, once each.
4. MLE output is proposal/planning information, never a second PF likelihood update.
5. Exact RJ evaluates the current PF target, not the old snapshot objective.
6. Proposed, realized, appended, and estimator-updated events are separate ledger facts.
7. Truth is absent from inference logs/configs and opened only after completion.
8. Runtime physics and safety attestations are consumed, never recreated locally.

## CLI

```bash
uv run radiation-estimation-orchestrator hybrid-v2-replay --config OFFLINE.json
uv run radiation-estimation-orchestrator hybrid-v2-live --config LIVE.json
uv run radiation-estimation-orchestrator evaluate-live \
  --manifest OUTPUT/live_hybrid_manifest.json \
  --truth TRUTH.json \
  --output OUTPUT/evaluation_metrics.json
```

Standalone protocol endpoints are `pf-checkpoint`, `spectral-mle`,
`future-spectral-score`, `exact-rj`, and `checkpoint-plan`.
