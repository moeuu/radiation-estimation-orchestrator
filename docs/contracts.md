# Runtime and local-estimator contracts

## MeasurementLog

Schema v1 is the archived projected-count contract. Schema v2 is the current common
contract produced by `Rotating-shield-simulation-runtime`. V2 carries raw integer
full spectra and the complete producer-owned forward-model identity; it deliberately
omits projected isotope counts, fitted count variances, and isotope covariance arrays.
The local PF and spectral MLE therefore see the same recorded photons without
maintaining a second observation-generation implementation.

Required files:

- `run_manifest.json`
- `runtime_config.resolved.json`
- `environment.json`
- `forward_model_manifest.json`
- `observations.npz`
- `observation_metadata.jsonl`
- `repository_commit.txt`

Truth files and truth-named NPZ/metadata fields are forbidden. Every artifact path is
checked component-by-component for truth, source-layout, source-position, or
point-source indicators. Run metadata, environment data, resolved runtime config, and
every observation metadata record are recursively checked for realized-truth keys and
string pointers. Generic realized `sources` lists are forbidden; `source_rate*` and
`source_extent*` model semantics are explicitly allowed. `source_layout_path` is a
required null value and can never name a file.

The public manifest requires a generic `repository_commit`, `run_id`, resolved-config
and forward-manifest hashes, source-rate semantics, six model identifiers/hashes
(detector, shield, environment, obstacle, transport, spectrum), causal index
conventions, per-artifact hashes, isotope order, and record/bin counts. Legacy
`upstream_pf_commit` and `runtime_config_sha256` are optional aliases only.

`runtime_config.resolved.json` uses the canonical JSON encoding described below. This
makes its raw artifact SHA-256 equal to `resolved_config_sha256`; the per-artifact hash
therefore proves the exact bytes while the resolved-config field proves the same
normalized configuration semantics.

### Forward model identity

`forward_model_manifest.json` is validated by the schema matching its MeasurementLog.
The v1 manifest explicitly carries the line attenuation table described below. The
MeasurementLog-v2 reader accepts forward-manifest schema 2 and the runtime's current
backwards-compatible schema 4. Schema 4 additionally binds the detector-response,
shield-pose, obstacle-material, and transport-physics contracts by identifier and/or
SHA-256. Consumers verify that immutable identity instead of rebuilding simulation
physics locally. In v1, in addition to the
six component identifiers, it requires `units.linear_attenuation="cm^-1"` and a
`line_mu_by_isotope` table. Every line entry has exactly `energy_keV`, normalized
`weight`, and line-specific `fe`/`pb` linear attenuation coefficients. Energies are
strictly increasing within each isotope and weights sum to one.

The shield model SHA-256 is the canonical JSON hash of the complete
`line_mu_by_isotope` table. The spectrum model SHA-256 is the canonical JSON hash of
the isotope-keyed table after projecting every line to only `energy_keV` and `weight`.
For the production Cs-137/Co-60/Eu-154 table these hashes are respectively
`c5e24ded41d8f15b59cbcb08d37c41d281a3867aa39e5fde4bf1bfb6004160f3`
and `49cc8ee41dea713ed6dcae459d676ffe78e6b70cacbfea2eba6df2eb732ace73`.
Eu-154 uses all six production lines at 723.3, 873.2, 996.3, 1274.5, 1494.0,
and 1596.5 keV—not the older three- or four-line detection subsets.

Canonical v1 NPZ arrays (`N` records, `B` bins, `I` isotopes):

| Array | dtype | shape |
|---|---:|---:|
| `step_id`, `action_id`, `station_id` | int64 | `(N,)` |
| `detector_pose_xyz` | float64 | `(N,3)` |
| `detector_quat_wxyz` | float64 | `(N,4)` |
| `fe_orientation_index`, `pb_orientation_index` | int64 | `(N,)` |
| `live_time_s`, `travel_time_s`, `shield_actuation_time_s` | float64 | `(N,)` |
| `energy_bin_edges_keV` | float64 | `(B+1,)` |
| `spectrum_counts`, `spectrum_variance` | float64 | `(N,B)` |
| `spectrum_variance_present` | bool | `(N,)` |
| `isotope_counts`, `isotope_counts_present` | float64/bool | `(N,I)` |
| `isotope_counts_record_present` | bool | `(N,)` |
| `isotope_count_covariance`, presence mask | float64/bool | `(N,I,I)` |
| `isotope_count_covariance_record_present` | bool | `(N,)` |

Absent optional numeric values use NaN and a false presence mask; present values are
finite. Metadata JSONL rows contain exactly aligned `run_id`, `array_index`, `step_id`,
`action_id`, `station_id`, and an estimator-independent `metadata` object.

Canonical v2 uses the same step/action/station, pose, quaternion, Fe/Pb orientation,
timing, and energy-edge arrays. `spectrum_counts` is nonnegative `int64`; no other
observation-derived numeric arrays are permitted. Steps and actions equal zero-based
row order, and station IDs form contiguous nondecreasing zero-based groups.

`measurement_log_sha256` is the SHA-256 of canonical JSON encoding of the sorted map
`{relative POSIX file path: raw file SHA-256}` for every regular file in the log. The
encoding uses UTF-8, `indent=2`, sorted keys, `(',', ': ')` separators, and one trailing
newline. Symlinks are forbidden.

## PFResult

The bundle contains `pf_posterior.json`, `pf_trace.jsonl`, and
`pf_diagnostics.json`. `pf_posterior.json` identifies a pure PF posterior, requires all
batch/surface/model-order flags false, and represents isotope estimates as MAP
cardinality, normalized cardinality distribution, and posterior `modes` with position
mean/covariance, strength mean, and posterior mass.
`map_cardinality` must be the distribution argmax; ties deterministically select the
smallest cardinality, matching the pure-PF contract.

Provenance binds this repository commit, MeasurementLog schema/hash, input and resolved
config hashes, seed, allowed PF planner belief sources (`pf_posterior`,
`pf_tentative`, or the v2 `joint_pf_particles` source), and
`batch_feedback_applied=false`.

During a same-log benchmark, PF posterior isotope keys must equal the MeasurementLog
isotope set and the trace `step_id` sequence must equal the log's causal `step_id`
sequence exactly. A matching length alone is insufficient.

## MLEResult

The bundle contains `mle_estimate.npz`, `mle_diagnostics.json`, and
`hotspot_clusters.json`. NPZ-to-JSON hashes and mirrors are validated. Provenance binds
spectral mode, this repository commit, log hash, input and resolved MLE config hashes,
`uses_pf_state=false`, `uses_pf_candidates=false`, and
`candidate_domain=complete_surface_dictionary`.
Both diagnostics and NPZ isotope order must equal the MeasurementLog isotope order;
channel subsets or reorderings are rejected.
NPZ objective, deviance, iteration, convergence, and patch-count scalars must exactly
mirror diagnostics. Patch IDs are unique and nonnegative; patch centroids and strengths
are finite; and every cluster isotope, patch membership, surface kind, integrated
strength, and strength-weighted centroid is cross-checked against the NPZ arrays.

## Hybrid contracts

The original MLESnapshot v1 remains readable for compatibility. The executable hybrid
uses MLESnapshot v2, which binds a surface-MLE result to the exact station-complete
MeasurementLog prefix through `source_run_id`, `covered_step_ids`, prefix-log hash,
neutral covered-record hash, and covered station-boundary hash. Predicted observations
must cover exactly the declared prefix. Warm-start lineage names the prior result and
does not change the requirement to optimize the full current-prefix objective.

PFDirective v1 binds proposals to one validated snapshot and one resolved PF
configuration. `verification_only` directives register candidates without a proposal
kernel. `proposal_only_mh` directives require a density-defined defensive truncated
Gaussian position kernel and target-preserving MH correction. All directives forbid
direct MLE-objective reweighting and hard pruning. A PFDirectiveReceipt accounts for
every proposal and records the before/after state hashes, MH decision evidence, and
the first legal future-observation step. Candidate outcomes contain aggregate
attempt/accept/reject/not-sampled counts across eligible PF particles; mixed decisions
are never compressed to a representative particle. Scalar MH ratio/draw evidence is
present only when exactly one particle attempted that proposal.

FutureCandidateScore v1 binds a frozen prefix snapshot and exact later prefix. It
records per-candidate, per-step full-versus-cluster-zero count log predictive ratios;
future rows are never refit and every ledger corroboration is hash-bound to one score
row.

HybridPlanningRequest/Recommendation v1 binds a prefix-only PF state to an ordered,
collision/reachability-attested XYZ candidate set. The recommendation echoes the exact
request hash, excludes quarantined external modes, proves PF state bytes are unchanged,
and requires `robot_actuation_authorized=false`.

HybridLedgerSummary v1 is an append-only hash chain over snapshot registration,
directive issuance, exactly one receipt per directive, and candidate corroboration.
Corroboration may use only a unique step strictly after the source snapshot cutoff.

HybridResult v1 (`hybrid_result.json`) records separate hashes for the user-supplied
source log and its inference-log derivative, whose only permitted change is adding
predeclared station-boundary completion markers. Record count, source run ID, step
sequence, and station schedule are checked across that derivation. It then binds the
complete run to validated final PF, cold count-MLE, cold spectral-MLE, ledger,
snapshot, directive, receipt, future-score, planning-request/recommendation, queue, and
execution-evidence hashes. The
cold spectral full-history MLE is explicitly authoritative and its hotspot clusters
are mirrored in the manifest. Semantic validation checks final cold-fit lineage,
reference/cutoff consistency, proposal/verification counts, once-only receipts, the
ledger tail, and the final spectral cluster mirror. The result contract also records
that v1 has no RJ birth/death, hard prune, direct MLE reweight, or live closed-loop
planner actuation. See [`hybrid_v1.md`](hybrid_v1.md).

## Hybrid-v2 contracts

SpectralMLESnapshot v3 binds a converged spectral surface fit, predicted-spectrum
artifact, complete-surface cluster geometry/covariance, and cold/warm lineage to one
exact MeasurementLog-v2 prefix.

FutureSpectralScoreRequest v1 partitions not-yet-scored post-cutoff steps.
FutureSpectralCandidateScore v2 reports frozen full-versus-candidate-zero raw-spectrum
LLRs in station/height/shield-program blocks. Validation rejects pre-cutoff rows,
overlapping blocks, missing requested steps, refitting, and repeated scoring. The
shield-program identity uses an explicit metadata ID when uniformly available and
otherwise hashes the ordered observed Fe/Pb orientation sequence; mixed explicit and
implicit identity inside one station/height block is invalid.

PFCheckpoint v1 binds the locally owned PF state artifact and RNG state to an exact
prefix. Generic controller code transports it by hash; only the local PF/RJ/planning
implementation may interpret its particle arrays.

The exact-RJ directive binds two times: `proposal_data_cutoff_*` identifies the older
spectral snapshot, while `data_cutoff_*` identifies the current PF target after future
verification. It also binds current log, covered records, checkpoint, and input PF
state hashes. The receipt exposes complete target/proposal/Jacobian terms; validation
recomputes acceptance, cardinality, state transition, candidate eligibility, transform
identity, and output-checkpoint state identity.

The live runtime protocol is stricter than a recommendation contract. Candidate sets
carry per-candidate path hashes plus aggregate collision/reachability/path attestation.
Action decisions and receipts must agree on realized XYZ, shield pair, station marker,
and exact selected path. The mission ledger durably orders proposed, realized,
appended, and estimator-updated events and repairs a state file lagging an fsynced
ledger after a crash.
