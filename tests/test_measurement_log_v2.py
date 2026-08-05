"""Compatibility tests for the PF-owned raw full-spectrum log contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from orchestrator.contracts import validate_measurement_log
from orchestrator.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON for one local contract fixture."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_v2_log(target: Path) -> Path:
    """Convert the bundled estimator-neutral fixture to canonical v2 storage."""
    source = ROOT / "fixtures" / "shared_small_run" / "measurement_log"
    shutil.copytree(source, target)
    manifest_path = target / "run_manifest.json"
    forward_path = target / "forward_model_manifest.json"
    observation_path = target / "observations.npz"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_forward = json.loads(forward_path.read_text(encoding="utf-8"))
    with np.load(observation_path, allow_pickle=False) as loaded:
        names = (
            "step_id",
            "action_id",
            "station_id",
            "detector_pose_xyz",
            "detector_quat_wxyz",
            "fe_orientation_index",
            "pb_orientation_index",
            "live_time_s",
            "travel_time_s",
            "shield_actuation_time_s",
            "energy_bin_edges_keV",
            "spectrum_counts",
        )
        arrays = {name: np.array(loaded[name], copy=True) for name in names}
    arrays["spectrum_counts"] = arrays["spectrum_counts"].astype(np.int64)
    with observation_path.open("wb") as handle:
        np.savez(handle, **arrays)

    source_semantics = {
        "quantity": "expected_pre_dead_time_detector_pulse_rate",
        "unit": "cps",
        "normalization_distance_m": 1.0,
    }
    forward = {
        "schema_version": 2,
        "repository_commit": old_manifest["repository_commit"],
        "resolved_config_sha256": old_manifest["resolved_config_sha256"],
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": source_semantics,
        "units": old_forward["units"],
        "response_semantics": {
            **old_forward["response_semantics"],
            "observation_distribution": (
                "joint_renewal_total_and_conditional_energy_marks"
            ),
        },
        "line_mu_by_isotope": old_forward["line_mu_by_isotope"],
        "model_identifiers": old_manifest["model_identifiers"],
    }
    _write_json(forward_path, forward)
    edges = arrays["energy_bin_edges_keV"]
    bin_width = float(edges[1] - edges[0])
    manifest = {
        "schema_version": 2,
        "run_id": old_manifest["run_id"],
        "record_count": int(arrays["step_id"].size),
        "repository_commit": old_manifest["repository_commit"],
        "resolved_config_sha256": old_manifest["resolved_config_sha256"],
        "forward_model_manifest_sha256": sha256_file(forward_path),
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": source_semantics,
        "isotopes": old_manifest["isotopes"],
        "environment": old_manifest["environment"],
        "obstacle_layout_path": None,
        "source_layout_path": None,
        "sim_backend": old_manifest["sim_backend"],
        "observation_model": "joint_full_spectrum_generative",
        "energy_bin_count": int(arrays["spectrum_counts"].shape[1]),
        "energy_min_keV": float(edges[0]),
        "energy_max_keV": float(edges[-2]),
        "bin_width_keV": bin_width,
        "full_spectrum_contract_hash_sha256": "a" * 64,
        "full_spectrum_contract_schema_version": 3,
        "model_identifiers": old_manifest["model_identifiers"],
        "index_conventions": old_manifest["index_conventions"],
        "artifact_hashes": {},
        "metadata": {
            "closed_loop": False,
            "estimator_independent": True,
            "fixture": "shared_small_run_v2",
        },
    }
    manifest["artifact_hashes"] = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in target.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json(manifest_path, manifest)
    return target


def test_raw_full_spectrum_v2_log_is_accepted(tmp_path: Path) -> None:
    """One PF v2 log can be shared without projected isotope-count arrays."""
    info = validate_measurement_log(_materialize_v2_log(tmp_path / "log-v2"))

    assert info.manifest["schema_version"] == 2
    assert info.record_count == 12
    assert info.arrays["spectrum_counts"].dtype == np.dtype(np.int64)
    assert set(info.arrays) == {
        "step_id",
        "action_id",
        "station_id",
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "fe_orientation_index",
        "pb_orientation_index",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "energy_bin_edges_keV",
        "spectrum_counts",
    }


def test_current_runtime_forward_manifest_v4_is_accepted(tmp_path: Path) -> None:
    """MeasurementLog v2 accepts the runtime's strengthened v4 physics identity."""
    log_dir = _materialize_v2_log(tmp_path / "log-v2-forward-v4")
    forward_path = log_dir / "forward_model_manifest.json"
    manifest_path = log_dir / "run_manifest.json"
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    forward.update(
        {
            "schema_version": 4,
            "detector_response_contract_sha256": "1" * 64,
            "shield_pose_contract_id": "shield-pose-contract-v1",
            "shield_pose_contract_sha256": "2" * 64,
            "obstacle_material_contract_id": "obstacle-material-contract-v1",
            "obstacle_material_contract_sha256": "3" * 64,
            "transport_physics_table_contract_id": "transport-table-contract-v1",
            "transport_physics_table_contract_sha256": "4" * 64,
        }
    )
    _write_json(forward_path, forward)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forward_hash = sha256_file(forward_path)
    manifest["forward_model_manifest_sha256"] = forward_hash
    manifest["artifact_hashes"]["forward_model_manifest.json"] = forward_hash
    _write_json(manifest_path, manifest)

    info = validate_measurement_log(log_dir)

    assert info.forward_model_manifest["schema_version"] == 4
