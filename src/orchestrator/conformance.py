"""Cross-repository unit-strength forward-response conformance boundary."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .adapters.base import EstimatorPin, expand_command, verify_repository_revision
from .errors import AdapterExecutionError, ContractError
from .hashing import load_json, sha256_bytes, sha256_file, write_json_atomic


@dataclass(frozen=True, slots=True)
class ForwardResponseProvenance:
    """Execution provenance supplied by a production response provider."""

    provider_revision: str
    expanded_command: tuple[str, ...]
    provider_config_sha256: str
    stdout_sha256: str
    stderr_sha256: str


@dataclass(frozen=True, slots=True)
class ForwardResponseArtifact:
    provider: str
    path: Path
    case_ids: tuple[str, ...]
    unit_response: NDArray[np.float64]
    sha256: str
    provenance: ForwardResponseProvenance | None = None


@runtime_checkable
class ForwardResponseProvider(Protocol):
    """Estimator-owned CLI or test double producing no shared physics code."""

    @property
    def name(self) -> str: ...

    def compute(self, fixture_path: Path, output_path: Path) -> ForwardResponseArtifact: ...


def load_forward_response(provider: str, path: str | Path) -> ForwardResponseArtifact:
    """Load the small provider-neutral response artifact."""
    supplied = Path(path)
    if supplied.is_symlink():
        raise ContractError(f"Forward-response artifact must not be a symlink: {supplied}")
    source = supplied.resolve()
    if not source.is_file():
        raise ContractError(f"Forward-response artifact is invalid: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"case_ids", "unit_response"}:
                raise ContractError(
                    "Forward-response NPZ must contain exactly case_ids and unit_response."
                )
            case_ids_array = np.asarray(archive["case_ids"])
            response = np.asarray(archive["unit_response"])
    except (OSError, ValueError) as exc:
        raise ContractError(f"Could not load forward-response artifact {source}.") from exc
    if case_ids_array.ndim != 1 or case_ids_array.dtype.kind != "U":
        raise ContractError("Forward-response case_ids must be a one-dimensional Unicode array.")
    if response.dtype != np.dtype(np.float64):
        raise ContractError("Forward-response unit_response must have exact float64 dtype.")
    case_ids = tuple(str(value) for value in case_ids_array.tolist())
    if len(set(case_ids)) != len(case_ids) or response.shape != (len(case_ids),):
        raise ContractError(
            "Forward-response unit_response must be a one-dimensional vector aligned "
            "exactly with unique case_ids."
        )
    if not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ContractError("Unit-strength responses must be finite and nonnegative.")
    response = np.array(response, copy=True)
    response.setflags(write=False)
    return ForwardResponseArtifact(
        provider=provider,
        path=source,
        case_ids=case_ids,
        unit_response=response,
        sha256=sha256_file(source),
    )


class CLIForwardResponseProvider:
    """Run an estimator-owned conformance CLI without importing its physics."""

    def __init__(
        self,
        name: str,
        *,
        repository_path: str | Path,
        revision: str,
        command_template: tuple[str, ...],
        provider_config_sha256: str,
        timeout_s: float = 300.0,
    ) -> None:
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ContractError(
                f"{name} conformance provider must pin an exact 40-character lowercase commit."
            )
        if len(provider_config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in provider_config_sha256
        ):
            raise ContractError("Provider config SHA-256 must be a lowercase 64-character digest.")
        self._name = name
        self.repository_path = Path(repository_path).resolve()
        self.revision = revision
        self.command_template = tuple(command_template)
        self.provider_config_sha256 = provider_config_sha256
        self.timeout_s = float(timeout_s)
        self._pin = EstimatorPin(
            name=name,
            repository=self.repository_path.as_posix(),
            revision=revision,
            revision_type="commit",
            release_tag=None,
            local_path_hint=None,
            expected_measurement_log_schema_version=1,
            expected_result_schema_version=1,
        )

    @property
    def name(self) -> str:
        return self._name

    def compute(self, fixture_path: Path, output_path: Path) -> ForwardResponseArtifact:
        observed_revision, _ = verify_repository_revision(
            self.repository_path,
            self._pin,
            require_clean=True,
            allowed_dirty_prefixes=(),
        )
        output = output_path.resolve()
        if output.exists():
            raise FileExistsError(f"Conformance output exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = expand_command(
            self.command_template,
            {
                "repository": self.repository_path,
                "config": fixture_path.resolve(),
                "output_dir": output,
                "log_dir": fixture_path.resolve(),
                "seed": 0,
                "profile": "forward-response-conformance",
                "mode": "forward-response-conformance",
            },
        )
        try:
            environment = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "UV_CACHE_DIR")
                if key in os.environ
            }
            environment.update(
                {
                    "PYTHONHASHSEED": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": (self.repository_path / "src").as_posix(),
                    "RSE_TRUTH_ACCESS": "forbidden",
                }
            )
            completed = subprocess.run(
                command,
                cwd=self.repository_path,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AdapterExecutionError(
                f"Could not run {self.name} conformance CLI: {exc}"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
            raise AdapterExecutionError(
                f"{self.name} conformance CLI exited {completed.returncode}: {stderr}"
            )
        artifact = load_forward_response(self.name, output)
        return replace(
            artifact,
            provenance=ForwardResponseProvenance(
                provider_revision=observed_revision,
                expanded_command=command,
                provider_config_sha256=self.provider_config_sha256,
                stdout_sha256=sha256_bytes(completed.stdout),
                stderr_sha256=sha256_bytes(completed.stderr),
            ),
        )


def _provider_report(artifact: ForwardResponseArtifact) -> dict[str, object]:
    """Serialize provider provenance, using explicit neutral values for test doubles."""
    provenance = artifact.provenance
    return {
        "provider_name": artifact.provider,
        "provider_revision": (None if provenance is None else provenance.provider_revision),
        "expanded_command": ([] if provenance is None else list(provenance.expanded_command)),
        "provider_config_sha256": (
            None if provenance is None else provenance.provider_config_sha256
        ),
        "stdout_sha256": None if provenance is None else provenance.stdout_sha256,
        "stderr_sha256": None if provenance is None else provenance.stderr_sha256,
        "artifact_sha256": artifact.sha256,
    }


def expected_conformance_case_count(fixture_path: str | Path) -> int:
    fixture = load_json(fixture_path)
    if fixture.get("schema_version") != 1:
        raise ContractError("Forward conformance fixture schema_version must be 1.")
    required = ("isotopes", "detector_poses", "shield_program", "source_points", "obstacles")
    if any(not isinstance(fixture.get(key), list | dict) for key in required):
        raise ContractError("Forward conformance fixture is missing required axes.")
    shield = fixture["shield_program"]
    assert isinstance(shield, dict)
    fe = shield.get("fe_orientation_indices")
    pb = shield.get("pb_orientation_indices")
    if fe != list(range(8)) or pb != list(range(8)) or shield.get("pairing") != "cartesian_product":
        raise ContractError("Conformance fixture must exercise all 64 Fe/Pb orientation pairs.")
    axes = [
        fixture["isotopes"],
        fixture["detector_poses"],
        fixture["source_points"],
        fixture["obstacles"],
    ]
    assert all(isinstance(value, list) for value in axes)
    return 64 * int(np.prod([len(value) for value in axes]))


def canonical_conformance_case_ids(fixture_path: str | Path) -> tuple[str, ...]:
    """Expand the mandated axes into the provider-neutral stable case order."""
    fixture = load_json(fixture_path)
    expected_conformance_case_count(fixture_path)
    shield = fixture["shield_program"]
    assert isinstance(shield, dict)
    isotopes = fixture["isotopes"]
    poses = fixture["detector_poses"]
    sources = fixture["source_points"]
    obstacles = fixture["obstacles"]
    assert all(isinstance(value, list) for value in (isotopes, poses, sources, obstacles))
    result: list[str] = []
    for isotope in isotopes:
        for pose in poses:
            assert isinstance(pose, dict)
            for fe_index in shield["fe_orientation_indices"]:  # type: ignore[union-attr]
                for pb_index in shield["pb_orientation_indices"]:  # type: ignore[union-attr]
                    for source in sources:
                        assert isinstance(source, dict)
                        for obstacle in obstacles:
                            assert isinstance(obstacle, dict)
                            result.append(
                                f"{isotope}|pose={pose['pose_id']}|fe={int(fe_index):02d}|"
                                f"pb={int(pb_index):02d}|source={source['source_id']}|"
                                f"obstacle={obstacle['obstacle_id']}"
                            )
    return tuple(result)


def run_forward_response_conformance(
    *,
    fixture_path: str | Path,
    pf_provider: ForwardResponseProvider,
    mle_provider: ForwardResponseProvider,
    output_directory: str | Path,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Run both independent implementations and require numerical agreement."""
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("Conformance tolerances must be finite and nonnegative.")
    fixture = Path(fixture_path).resolve()
    expected_count = expected_conformance_case_count(fixture)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Conformance output directory exists: {output}")
    output.mkdir(parents=True)
    pf = pf_provider.compute(fixture, output / "pf_forward_response.npz")
    mle = mle_provider.compute(fixture, output / "mle_forward_response.npz")
    canonical_case_ids = canonical_conformance_case_ids(fixture)
    if len(canonical_case_ids) != expected_count:
        raise RuntimeError("Canonical conformance expansion disagrees with its case count.")
    if pf.case_ids != canonical_case_ids:
        raise ContractError("PF forward-response case IDs differ from the canonical fixture order.")
    if mle.case_ids != canonical_case_ids:
        raise ContractError(
            "MLE forward-response case IDs differ from the canonical fixture order."
        )
    expected_shape = (expected_count,)
    if pf.unit_response.shape != expected_shape or mle.unit_response.shape != expected_shape:
        raise ContractError(
            f"PF and MLE forward responses must both have exact scalar shape {expected_shape}."
        )
    delta = np.abs(pf.unit_response - mle.unit_response)
    scale = np.maximum(np.abs(mle.unit_response), atol)
    report = {
        "schema_version": 1,
        "fixture_path": fixture.as_posix(),
        "fixture_sha256": sha256_file(fixture),
        "case_count": expected_count,
        "response_shape": list(pf.unit_response.shape),
        "rtol": float(rtol),
        "atol": float(atol),
        "pf_artifact_sha256": pf.sha256,
        "mle_artifact_sha256": mle.sha256,
        "providers": {
            "particle_filter": _provider_report(pf),
            "surface_mle": _provider_report(mle),
        },
        "max_absolute_difference": float(np.max(delta, initial=0.0)),
        "max_relative_difference": float(np.max(delta / scale, initial=0.0)),
        "passed": bool(np.allclose(pf.unit_response, mle.unit_response, rtol=rtol, atol=atol)),
    }
    write_json_atomic(output / "conformance_report.json", report)
    if not report["passed"]:
        worst = int(np.argmax(delta)) if delta.size else 0
        raise ContractError(
            "PF/MLE unit-strength responses differ beyond tolerance; "
            f"flat worst index={worst}, max_abs={report['max_absolute_difference']}."
        )
    return report
