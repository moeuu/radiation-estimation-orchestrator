from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.contracts import validate_pf_checkpoint_v1
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import sha256_file, write_json_atomic


def _checkpoint(tmp_path: Path) -> Path:
    state = tmp_path / "pf_state.npz"
    state.write_bytes(b"opaque-pf-owned-state")
    return write_json_atomic(
        tmp_path / "pf_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_family": "pure_pf_causal_state",
            "checkpoint_id": "checkpoint-step-2",
            "source_run_id": "run-v2",
            "measurement_log_schema_version": 2,
            "data_cutoff_step": 2,
            "data_cutoff_station": 0,
            "covered_step_ids": [0, 1, 2],
            "covered_records_sha256": "a" * 64,
            "prefix_measurement_log_sha256": "b" * 64,
            "pf_repository_commit": "c" * 40,
            "resolved_config_sha256": "d" * 64,
            "random_seed": 7,
            "state_artifact": "pf_state.npz",
            "state_artifact_sha256": sha256_file(state),
            "rng_state_sha256": "e" * 64,
            "safety": {
                "prefix_causal": True,
                "truth_read": False,
                "batch_feedback_applied": False,
            },
        },
    )


def test_pf_checkpoint_binds_opaque_state_and_exact_prefix(tmp_path: Path) -> None:
    info = validate_pf_checkpoint_v1(
        _checkpoint(tmp_path),
        expected_source_run_id="run-v2",
        expected_prefix_measurement_log_sha256="b" * 64,
    )

    assert info.cutoff_step == 2


def test_pf_checkpoint_rejects_nonprefix_steps(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["covered_step_ids"] = [0, 2]
    write_json_atomic(path, payload, overwrite=True)

    with pytest.raises(DataReuseError, match="exact causal prefix"):
        validate_pf_checkpoint_v1(path)


def test_pf_checkpoint_rejects_modified_state(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path)
    (tmp_path / "pf_state.npz").write_bytes(b"changed")

    with pytest.raises(ContractError, match="hash mismatch"):
        validate_pf_checkpoint_v1(path)
