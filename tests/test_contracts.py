from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from orchestrator.contracts import (
    validate_measurement_log,
    validate_mle_result,
    validate_mle_snapshot,
    validate_pf_result,
)
from orchestrator.errors import ContractError, TruthIsolationError
from orchestrator.hashing import load_json, sha256_file, write_json_atomic


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays.update(updates)
    np.savez(path, **arrays)


def _rewrite_mle_diagnostics(result: Path, mutate: Callable[[list[object]], None]) -> None:
    diagnostics_path = result / "mle_diagnostics.json"
    diagnostics = load_json(diagnostics_path)
    nested = diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    clusters = nested["hotspot_clusters"]
    assert isinstance(clusters, list)
    mutate(clusters)
    write_json_atomic(diagnostics_path, diagnostics, overwrite=True)
    write_json_atomic(
        result / "hotspot_clusters.json",
        {"schema_version": 1, "hotspot_clusters": clusters},
        overwrite=True,
    )
    _rewrite_npz(
        result / "mle_estimate.npz",
        diagnostics_sha256=np.asarray(sha256_file(diagnostics_path)),
    )


def test_shared_measurement_log_v1_is_complete_and_truth_free(
    measurement_log_path: Path,
) -> None:
    info = validate_measurement_log(measurement_log_path)
    assert info.record_count == 12
    assert info.step_ids == tuple(range(12))
    assert info.isotopes == ("Cs-137", "Co-60", "Eu-154")
    assert info.measurement_log_sha256 == (
        "2ed73e665983ab1deffdb8e867b92e6becf962089e4c2ccad1640649f50bdc5d"
    )
    assert not any("truth" in path.name.lower() for path in measurement_log_path.rglob("*"))


def test_forward_manifest_binds_exact_production_line_table(measurement_log_path: Path) -> None:
    info = validate_measurement_log(measurement_log_path)
    forward = info.forward_model_manifest
    assert forward["units"]["linear_attenuation"] == "cm^-1"  # type: ignore[index]
    table = forward["line_mu_by_isotope"]
    assert isinstance(table, dict)
    eu_lines = table["Eu-154"]
    assert isinstance(eu_lines, list)
    assert [row["energy_keV"] for row in eu_lines] == [  # type: ignore[index]
        723.3,
        873.2,
        996.3,
        1274.5,
        1494.0,
        1596.5,
    ]
    identifiers = forward["model_identifiers"]
    assert isinstance(identifiers, dict)
    assert identifiers["shield"]["sha256"] == (  # type: ignore[index]
        "c5e24ded41d8f15b59cbcb08d37c41d281a3867aa39e5fde4bf1bfb6004160f3"
    )
    assert identifiers["spectrum"]["sha256"] == (  # type: ignore[index]
        "49cc8ee41dea713ed6dcae459d676ffe78e6b70cacbfea2eba6df2eb732ace73"
    )


def test_forward_manifest_rejects_line_table_hash_drift(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    forward_path = copied / "forward_model_manifest.json"
    forward = load_json(forward_path)
    line_table = forward["line_mu_by_isotope"]
    assert isinstance(line_table, dict)
    eu_lines = line_table["Eu-154"]
    assert isinstance(eu_lines, list)
    eu_lines[0]["weight"] += 0.001  # type: ignore[index,operator]
    eu_lines[1]["weight"] -= 0.001  # type: ignore[index,operator]
    write_json_atomic(forward_path, forward, overwrite=True)

    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    new_forward_hash = sha256_file(forward_path)
    manifest["forward_model_manifest_sha256"] = new_forward_hash
    artifact_hashes = manifest["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["forward_model_manifest.json"] = new_forward_hash
    write_json_atomic(manifest_path, manifest, overwrite=True)
    with pytest.raises(ContractError, match="hash must bind"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_embedded_truth(measurement_log_path: Path, tmp_path: Path) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    (copied / "truth.json").write_text("{}\n")
    with pytest.raises(TruthIsolationError):
        validate_measurement_log(copied)


def test_measurement_log_rejects_truth_indicating_relative_path(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    nested = copied / "auxiliary" / "source_layout"
    nested.mkdir(parents=True)
    (nested / "data.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TruthIsolationError, match="artifact path"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_recursive_realized_truth_metadata(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    metadata["nested"] = {"source_positions": [[1.0, 2.0, 3.0]]}
    write_json_atomic(manifest_path, manifest, overwrite=True)
    with pytest.raises(TruthIsolationError, match="source_positions"):
        validate_measurement_log(copied)


@pytest.mark.parametrize("key", ["sourcePositions", "sourceLayoutPath", "pointSources"])
def test_measurement_log_rejects_camel_case_truth_keys(
    measurement_log_path: Path,
    tmp_path: Path,
    key: str,
) -> None:
    """Case style must not let realized source truth evade isolation."""
    copied = tmp_path / f"camel-{key}"
    shutil.copytree(measurement_log_path, copied)
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    metadata[key] = [[1.0, 2.0, 3.0]]
    write_json_atomic(manifest_path, manifest, overwrite=True)

    with pytest.raises(TruthIsolationError):
        validate_measurement_log(copied)


def test_measurement_log_rejects_recursive_runtime_config_truth(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    config_path = copied / "runtime_config.resolved.json"
    config = load_json(config_path)
    config["nested"] = {"point_sources": []}
    write_json_atomic(config_path, config, overwrite=True)
    with pytest.raises(TruthIsolationError, match="point_sources"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_recursive_environment_truth(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    environment_path = copied / "environment.json"
    environment = load_json(environment_path)
    environment["nested"] = {"evaluation_truth": "hidden.json"}
    write_json_atomic(environment_path, environment, overwrite=True)
    with pytest.raises(TruthIsolationError, match="evaluation_truth"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_realized_truth_string_values(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    metadata["external_pointer"] = "evaluation/ground_truth.json"
    write_json_atomic(manifest_path, manifest, overwrite=True)
    with pytest.raises(TruthIsolationError, match="realized-truth value"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_recursive_observation_source_list(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    metadata_path = copied / "observation_metadata.jsonl"
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["metadata"]["nested"] = {"sources": [{"position": [1, 2, 3]}]}
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    artifact_hashes = manifest["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["observation_metadata.jsonl"] = sha256_file(metadata_path)
    write_json_atomic(manifest_path, manifest, overwrite=True)
    with pytest.raises(TruthIsolationError, match=r"\.sources"):
        validate_measurement_log(copied)


def test_measurement_log_allows_source_rate_and_extent_model_semantics(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    metadata["source_rate_model"] = "detector_cps_1m"
    metadata["source_extent_semantics"] = "finite_support_model"
    write_json_atomic(manifest_path, manifest, overwrite=True)
    validate_measurement_log(copied)


def test_measurement_log_source_layout_path_is_always_null(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    manifest_path = copied / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["source_layout_path"] = "outside.json"
    write_json_atomic(manifest_path, manifest, overwrite=True)
    with pytest.raises(ContractError, match="source_layout_path"):
        validate_measurement_log(copied)


def test_measurement_log_rejects_symlink_root(measurement_log_path: Path, tmp_path: Path) -> None:
    link = tmp_path / "measurement-log-link"
    link.symlink_to(measurement_log_path, target_is_directory=True)
    with pytest.raises(ContractError, match="must not be a symlink"):
        validate_measurement_log(link)


def test_measurement_log_rejects_artifact_hash_drift(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "log"
    shutil.copytree(measurement_log_path, copied)
    with (copied / "environment.json").open("ab") as handle:
        handle.write(b" \n")
    with pytest.raises(ContractError, match=r"environment|artifact|hash"):
        validate_measurement_log(copied)


def test_result_contracts_accept_complete_subprocess_outputs(benchmark_output: Path) -> None:
    pf = validate_pf_result(benchmark_output / "results" / "pf_strict")
    count = validate_mle_result(benchmark_output / "results" / "mle_count", expected_mode="count")
    spectral = validate_mle_result(
        benchmark_output / "results" / "mle_spectral", expected_mode="spectral"
    )
    assert pf.posterior["final_estimate_source"] == "pf_posterior"
    assert count.mode == "count"
    assert spectral.mode == "spectral"


def test_pf_result_accepts_raw_measurement_log_v2_provenance(
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    """Current PF replay artifacts remain valid behind the shared v2 log."""
    copied = tmp_path / "pf-result-v2"
    shutil.copytree(benchmark_output / "results" / "pf_strict", copied)
    posterior_path = copied / "pf_posterior.json"
    posterior = load_json(posterior_path)
    provenance = posterior["provenance"]
    assert isinstance(provenance, dict)
    provenance["measurement_log_schema_version"] = 2
    provenance["planner_belief_sources"] = ["joint_pf_particles"]
    write_json_atomic(posterior_path, posterior, overwrite=True)

    trace_path = copied / "pf_trace.jsonl"
    transformed_trace = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["schema_version"] = 2
        row["estimator_family"] = "pure_particle_filter"
        transformed_trace.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    trace_path.write_text("\n".join(transformed_trace) + "\n", encoding="utf-8")

    diagnostics_path = copied / "pf_diagnostics.json"
    diagnostics = load_json(diagnostics_path)
    diagnostics["schema_version"] = 2
    diagnostics["measurement_log_schema_version"] = 2
    write_json_atomic(diagnostics_path, diagnostics, overwrite=True)

    result = validate_pf_result(copied)

    assert result.posterior["provenance"]["measurement_log_schema_version"] == 2


def test_mle_result_accepts_raw_measurement_log_v2_provenance(
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    """Spectral MLE results may bind directly to a shared raw v2 log."""
    copied = tmp_path / "mle-result-v2"
    shutil.copytree(benchmark_output / "results" / "mle_spectral", copied)
    diagnostics_path = copied / "mle_diagnostics.json"
    diagnostics = load_json(diagnostics_path)
    nested = diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    provenance = nested["provenance"]
    assert isinstance(provenance, dict)
    provenance["measurement_log_schema_version"] = 2
    write_json_atomic(diagnostics_path, diagnostics, overwrite=True)

    with np.load(copied / "mle_estimate.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["diagnostics_sha256"] = np.asarray(sha256_file(diagnostics_path))
    np.savez(copied / "mle_estimate.npz", **arrays)

    result = validate_mle_result(copied, expected_mode="spectral")

    assert result.diagnostics["diagnostics"]["provenance"][
        "measurement_log_schema_version"
    ] == 2


def test_pf_result_must_match_requested_profile(benchmark_output: Path, tmp_path: Path) -> None:
    copied = tmp_path / "pf-result"
    shutil.copytree(benchmark_output / "results" / "pf_strict", copied)
    posterior_path = copied / "pf_posterior.json"
    posterior = load_json(posterior_path)
    posterior["estimator_variant"] = "pf_profiled"
    provenance = posterior["provenance"]
    assert isinstance(provenance, dict)
    provenance["estimator_variant"] = "pf_profiled"
    write_json_atomic(posterior_path, posterior, overwrite=True)
    with pytest.raises(ContractError, match="does not match requested variant"):
        validate_pf_result(copied, expected_variant="pf_strict")


def test_pf_map_cardinality_is_deterministic_argmax_with_smallest_tie(
    benchmark_output: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "pf-result"
    shutil.copytree(benchmark_output / "results" / "pf_strict", copied)
    posterior_path = copied / "pf_posterior.json"
    posterior = load_json(posterior_path)
    isotopes = posterior["isotopes"]
    assert isinstance(isotopes, dict)
    estimate = isotopes["Cs-137"]
    assert isinstance(estimate, dict)
    estimate["cardinality_distribution"] = {"0": 0.5, "1": 0.5}
    write_json_atomic(posterior_path, posterior, overwrite=True)
    with pytest.raises(ContractError, match="smallest-cardinality tie breaking"):
        validate_pf_result(copied)


def test_result_resolved_config_hash_must_match_independent_expectation(
    benchmark_output: Path,
) -> None:
    with pytest.raises(ContractError, match="resolved-config hash"):
        validate_pf_result(
            benchmark_output / "results" / "pf_strict",
            expected_resolved_config_sha256="0" * 64,
        )
    with pytest.raises(ContractError, match="resolved estimator config hash"):
        validate_mle_result(
            benchmark_output / "results" / "mle_count",
            expected_resolved_config_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("objective_value", np.asarray(0.0, dtype=np.float64), "objective_value differs"),
        ("poisson_deviance", np.asarray(0.0, dtype=np.float64), "poisson_deviance differs"),
        ("iterations", np.asarray(43, dtype=np.int64), "iterations differs"),
        ("converged", np.asarray(0, dtype=np.uint8), "converged differs"),
        ("patch_count", np.asarray(7, dtype=np.int64), "patch_count differs"),
    ],
)
def test_mle_npz_summary_scalars_must_match_diagnostics(
    benchmark_output: Path,
    tmp_path: Path,
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    copied = tmp_path / f"mle-result-{field}"
    shutil.copytree(benchmark_output / "results" / "mle_count", copied)
    estimate_path = copied / "mle_estimate.npz"
    _rewrite_npz(estimate_path, **{field: replacement})
    with pytest.raises(ContractError, match=message):
        validate_mle_result(copied)


def test_mle_patch_centroids_and_strengths_must_be_finite(
    benchmark_output: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "mle-result"
    shutil.copytree(benchmark_output / "results" / "mle_count", copied)
    estimate_path = copied / "mle_estimate.npz"
    with np.load(estimate_path, allow_pickle=False) as archive:
        centroids = np.array(archive["patch_centroids_xyz"], copy=True)
    centroids[0, 0] = np.nan
    _rewrite_npz(estimate_path, patch_centroids_xyz=centroids)
    with pytest.raises(ContractError, match="centroids must be finite"):
        validate_mle_result(copied)

    copied_strength = tmp_path / "mle-result-strength"
    shutil.copytree(benchmark_output / "results" / "mle_count", copied_strength)
    strength_path = copied_strength / "mle_estimate.npz"
    with np.load(strength_path, allow_pickle=False) as archive:
        strengths = np.array(archive["patch_strength_by_isotope"], copy=True)
    strengths[0, 0] = np.inf
    _rewrite_npz(strength_path, patch_strength_by_isotope=strengths)
    with pytest.raises(ContractError, match="strengths must be finite"):
        validate_mle_result(copied_strength)


def test_mle_cluster_isotope_must_exist_in_result_order(
    benchmark_output: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "mle-result"
    shutil.copytree(benchmark_output / "results" / "mle_count", copied)

    def mutate(clusters: list[object]) -> None:
        cluster = clusters[0]
        assert isinstance(cluster, dict)
        cluster["isotope"] = "I-131"

    _rewrite_mle_diagnostics(copied, mutate)
    with pytest.raises(ContractError, match="absent from the result isotope order"):
        validate_mle_result(copied)


def test_mle_cluster_patch_ids_must_be_valid_and_consistent(
    benchmark_output: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "mle-result"
    shutil.copytree(benchmark_output / "results" / "mle_count", copied)

    def mutate(clusters: list[object]) -> None:
        cluster = clusters[0]
        assert isinstance(cluster, dict)
        cluster["patch_ids"] = [999999]

    _rewrite_mle_diagnostics(copied, mutate)
    with pytest.raises(ContractError, match="unknown patch ID"):
        validate_mle_result(copied)


def test_result_channels_and_pf_steps_must_match_measurement_log(
    benchmark_output: Path, measurement_log_path: Path
) -> None:
    log = validate_measurement_log(measurement_log_path)
    pf_path = benchmark_output / "results" / "pf_strict"
    count_path = benchmark_output / "results" / "mle_count"
    with pytest.raises(ContractError, match="isotope keys"):
        validate_pf_result(pf_path, expected_isotopes=("Cs-137", "Co-60"))
    with pytest.raises(ContractError, match="isotope order"):
        validate_mle_result(
            count_path,
            expected_mode="count",
            expected_isotopes=tuple(reversed(log.isotopes)),
        )
    shifted_steps = tuple(step + 1 for step in log.step_ids)
    with pytest.raises(ContractError, match="causal steps"):
        validate_pf_result(pf_path, expected_step_ids=shifted_steps)


def test_future_mle_snapshot_contract_enforces_exact_cutoff(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "snapshot_id": "snapshot-1",
        "estimator_family": "surface_mle",
        "estimator_variant": "count",
        "data_cutoff_step": 5,
        "data_cutoff_station": 1,
        "covered_step_ids": [0, 1, 2, 3, 4, 5],
        "measurement_log_sha256": "a" * 64,
        "mle_result_sha256": "b" * 64,
        "clusters": [],
        "fit_diagnostics": {},
        "provenance": {"future_contract_only": True},
    }
    path = write_json_atomic(tmp_path / "snapshot.json", payload)
    assert validate_mle_snapshot(path)["data_cutoff_step"] == 5
    payload["covered_step_ids"] = [0, 1, 2, 3, 4]
    invalid = write_json_atomic(tmp_path / "invalid.json", payload)
    with pytest.raises(ContractError, match="end exactly"):
        validate_mle_snapshot(invalid)
