# Cross-repository contracts v1

## MeasurementLog

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

`forward_model_manifest.json` follows its own packaged v1 schema. In addition to the
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

Canonical NPZ arrays (`N` records, `B` bins, `I` isotopes):

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

Provenance binds estimator commit, MeasurementLog schema/hash, input and resolved
config hashes, seed, allowed PF planner belief sources (`pf_posterior` and
`pf_tentative` only), and
`batch_feedback_applied=false`.

During a same-log benchmark, PF posterior isotope keys must equal the MeasurementLog
isotope set and the trace `step_id` sequence must equal the log's causal `step_id`
sequence exactly. A matching length alone is insufficient.

## MLEResult

The bundle contains `mle_estimate.npz`, `mle_diagnostics.json`, and
`hotspot_clusters.json`. NPZ-to-JSON hashes and mirrors are validated. Provenance binds
count/spectral mode, estimator commit, log hash, input and resolved MLE config hashes,
`uses_pf_state=false`, `uses_pf_candidates=false`, and
`candidate_domain=complete_surface_dictionary`.
Both diagnostics and NPZ isotope order must equal the MeasurementLog isotope order;
channel subsets or reorderings are rejected.
NPZ objective, deviance, iteration, convergence, and patch-count scalars must exactly
mirror diagnostics. Patch IDs are unique and nonnegative; patch centroids and strengths
are finite; and every cluster isotope, patch membership, surface kind, integrated
strength, and strength-weighted centroid is cross-checked against the NPZ arrays.

## Reserved MLESnapshot

The schema reserves `data_cutoff_step`, `data_cutoff_station`, sorted unique
`covered_step_ids`, result/input hashes, clusters, and fit diagnostics. It is not called
by the v1 benchmark. Any later hybrid milestone must ensure coverage ends exactly at
the cutoff and that a snapshot cannot cause observations at or before the cutoff to be
applied to PF weights a second time.
