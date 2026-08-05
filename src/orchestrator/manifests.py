"""Hash-rich benchmark manifest construction and atomic persistence."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import MeasurementLogInfo, MLEResultInfo, PFResultInfo
from .errors import ContractError
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic

if TYPE_CHECKING:
    from .adapters import AdapterExecution, EstimatorPin


def _git_value(repository: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *args),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def orchestrator_provenance(repository: str | Path) -> dict[str, object]:
    """Record this orchestrator checkout, including an intentional dirty state."""
    root = Path(repository).resolve()
    status = _git_value(root, "status", "--porcelain=v1", "--untracked-files=all")
    listed = _git_value(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    source_inventory: dict[str, str] = {}
    for relative in sorted(item for item in (listed or "").split("\0") if item):
        path = root / relative
        source_inventory[relative] = sha256_file(path) if path.is_file() else "deleted"
    return {
        "repository_path": root.as_posix(),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "tracked_tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git_value(root, "branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status_sha256": sha256_bytes((status or "").encode("utf-8")),
        "source_snapshot_sha256": sha256_bytes(canonical_json_bytes(source_inventory)),
        "source_file_count": len(source_inventory),
    }


def environment_provenance() -> dict[str, object]:
    """Capture runtime versions that can affect numeric/replay serialization."""
    packages: dict[str, str] = {}
    for name in ("numpy", "jsonschema", "psutil"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def pin_payload(pin: EstimatorPin) -> dict[str, object]:
    return {
        "repository": pin.repository,
        "revision": pin.revision,
        "revision_type": pin.revision_type,
        "release_tag": pin.release_tag,
        "expected_measurement_log_schema_version": pin.expected_measurement_log_schema_version,
        "expected_result_schema_version": pin.expected_result_schema_version,
    }


def _resolved_estimator_config_hashes(
    pf_result: PFResultInfo,
    mle_count_result: MLEResultInfo | None,
    mle_spectral_result: MLEResultInfo,
) -> dict[str, str]:
    """Extract estimator-resolved hashes only from already validated provenance."""
    pf_provenance = pf_result.posterior["provenance"]
    assert isinstance(pf_provenance, dict)

    def mle_provenance(result: MLEResultInfo) -> dict[str, object]:
        diagnostics = result.diagnostics["diagnostics"]
        assert isinstance(diagnostics, dict)
        provenance = diagnostics["provenance"]
        assert isinstance(provenance, dict)
        return provenance

    resolved = {
        "pf_strict": str(pf_provenance["resolved_config_sha256"]),
        "mle_spectral": str(
            mle_provenance(mle_spectral_result)["resolved_estimator_config_sha256"]
        ),
    }
    if mle_count_result is not None:
        resolved["mle_count"] = str(
            mle_provenance(mle_count_result)["resolved_estimator_config_sha256"]
        )
    return resolved


def build_benchmark_manifest(
    *,
    benchmark_id: str,
    started_at_utc: str,
    completed_at_utc: str,
    orchestrator_root: Path,
    pin_registry_path: Path,
    pins: Mapping[str, EstimatorPin],
    benchmark_config_path: Path,
    estimator_config_file_hashes: Mapping[str, str],
    expected_resolved_config_hashes: Mapping[str, str],
    measurement_log: MeasurementLogInfo,
    truth_path: Path,
    metrics_path: Path,
    executions: Mapping[str, AdapterExecution],
    pf_result: PFResultInfo,
    mle_count_result: MLEResultInfo | None,
    mle_spectral_result: MLEResultInfo,
) -> dict[str, object]:
    """Build a complete manifest without reading estimator source code."""
    resolved_config_hashes = _resolved_estimator_config_hashes(
        pf_result,
        mle_count_result,
        mle_spectral_result,
    )
    if dict(resolved_config_hashes) != dict(expected_resolved_config_hashes):
        raise ContractError(
            "Validated resolved estimator hashes differ from benchmark expectations."
        )
    pipeline_order = ["validate_measurement_log", "pure_pf_replay"]
    if mle_count_result is not None:
        pipeline_order.append("count_mle_replay")
    pipeline_order.extend(
        [
            "spectral_mle_replay",
            "validate_result_contracts",
            "open_evaluation_truth",
            "compute_metrics",
            "write_manifest",
        ]
    )
    validated_outputs: dict[str, object] = {
        "pf_strict": {
            "sha256": pf_result.result_sha256,
            "artifact_inventory": dict(pf_result.artifact_inventory),
        },
        "mle_spectral": {
            "sha256": mle_spectral_result.result_sha256,
            "artifact_inventory": dict(mle_spectral_result.artifact_inventory),
        },
    }
    if mle_count_result is not None:
        validated_outputs["mle_count"] = {
            "sha256": mle_count_result.result_sha256,
            "artifact_inventory": dict(mle_count_result.artifact_inventory),
        }
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "status": "complete",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "pipeline_order": pipeline_order,
        "truth_isolation": {
            "truth_path": truth_path.resolve().as_posix(),
            "truth_sha256": sha256_file(truth_path),
            "opened_only_after_all_result_validation": True,
            "passed_to_estimator_commands": False,
        },
        "contracts": {
            "measurement_log": measurement_log.schema_version,
            "pf_result": 1,
            "mle_result": 1,
            "mle_snapshot": 1,
        },
        "orchestrator": orchestrator_provenance(orchestrator_root),
        "runtime_environment": environment_provenance(),
        "pin_registry": {
            "path": pin_registry_path.resolve().as_posix(),
            "sha256": sha256_file(pin_registry_path),
            "estimators": {name: pin_payload(pin) for name, pin in sorted(pins.items())},
        },
        "benchmark_config": {
            "path": benchmark_config_path.resolve().as_posix(),
            "sha256": sha256_file(benchmark_config_path),
            "estimator_config_file_sha256": dict(sorted(estimator_config_file_hashes.items())),
            "expected_resolved_estimator_config_sha256": dict(
                sorted(expected_resolved_config_hashes.items())
            ),
            "resolved_estimator_config_sha256": dict(sorted(resolved_config_hashes.items())),
        },
        "execution_paths_relative_to_benchmark_root": True,
        "measurement_log": {
            "path": measurement_log.root.as_posix(),
            "schema_version": measurement_log.manifest["schema_version"],
            "run_id": measurement_log.manifest["run_id"],
            "sha256": measurement_log.measurement_log_sha256,
            "artifact_inventory": dict(measurement_log.artifact_inventory),
        },
        "executions": {name: execution.to_dict() for name, execution in sorted(executions.items())},
        "validated_outputs": validated_outputs,
        "metrics": {
            "path": metrics_path.name,
            "sha256": sha256_file(metrics_path),
        },
    }


def write_manifest_bundle(
    output_directory: str | Path, manifest: Mapping[str, object]
) -> tuple[Path, Path]:
    """Write the canonical manifest and a non-self-referential SHA-256 sidecar."""
    root = Path(output_directory)
    manifest_path = write_json_atomic(root / "benchmark_manifest.json", dict(manifest))
    digest = sha256_file(manifest_path)
    sidecar = root / "benchmark_manifest.sha256"
    payload = f"{digest}  benchmark_manifest.json\n".encode("ascii")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise
    if sha256_bytes(canonical_json_bytes(dict(manifest))) != digest:
        raise RuntimeError("Manifest canonical serialization changed unexpectedly.")
    return manifest_path, sidecar
