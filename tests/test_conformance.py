from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from orchestrator.cli import _provider
from orchestrator.conformance import (
    CLIForwardResponseProvider,
    ForwardResponseArtifact,
    ForwardResponseProvenance,
    canonical_conformance_case_ids,
    expected_conformance_case_count,
    load_forward_response,
    run_forward_response_conformance,
)
from orchestrator.errors import ContractError, RevisionError
from orchestrator.hashing import load_json, sha256_bytes, sha256_file, write_json_atomic


@dataclass
class AnalyticProvider:
    name: str
    bias: float = 0.0
    reverse_order: bool = False
    provenance: ForwardResponseProvenance | None = None

    def compute(self, fixture_path: Path, output_path: Path) -> ForwardResponseArtifact:
        case_ids = canonical_conformance_case_ids(fixture_path)
        index = np.arange(len(case_ids), dtype=np.float64)
        response = 1.0e-6 + (index % 97) * 1.0e-8 + self.bias
        if self.reverse_order:
            case_ids = tuple(reversed(case_ids))
            response = response[::-1]
        np.savez(output_path, case_ids=np.asarray(case_ids), unit_response=response)
        return replace(load_forward_response(self.name, output_path), provenance=self.provenance)


def _git(repository: Path, *arguments: str) -> str:
    """Run one deterministic Git command for a temporary provider checkout."""
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_provider_repository(repository: Path) -> str:
    """Create a clean exact-revision CLI provider repository."""
    repository.mkdir()
    script = repository / "provider.py"
    script.write_text(
        """from pathlib import Path
import sys

import numpy as np

output = Path(sys.argv[1])
np.savez(
    output,
    case_ids=np.asarray([\"case-0\"]),
    unit_response=np.asarray([1.25], dtype=np.float64),
)
print(\"provider stdout\")
print(\"provider stderr\", file=sys.stderr)
""",
        encoding="utf-8",
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "conformance@example.invalid")
    _git(repository, "config", "user.name", "Conformance Test")
    _git(repository, "add", "provider.py")
    _git(repository, "commit", "-m", "provider")
    return _git(repository, "rev-parse", "HEAD")


def test_conformance_fixture_covers_all_required_axes(repository_root: Path) -> None:
    fixture = repository_root / "fixtures" / "forward_response_conformance.json"
    assert expected_conformance_case_count(fixture) == 3 * 3 * 64 * 4 * 2
    case_ids = canonical_conformance_case_ids(fixture)
    assert len(case_ids) == 4608
    assert len(set(case_ids)) == len(case_ids)
    assert "fe=00|pb=00" in case_ids[0]
    assert any("fe=07|pb=07" in value for value in case_ids)


def test_independent_forward_response_interfaces_must_agree(
    repository_root: Path, tmp_path: Path
) -> None:
    report = run_forward_response_conformance(
        fixture_path=repository_root / "fixtures" / "forward_response_conformance.json",
        pf_provider=AnalyticProvider("pf"),
        mle_provider=AnalyticProvider("mle"),
        output_directory=tmp_path / "pass",
    )
    assert report["passed"] is True
    pf_provenance = report["providers"]["particle_filter"]
    assert pf_provenance["provider_revision"] is None
    assert pf_provenance["expanded_command"] == []
    assert pf_provenance["artifact_sha256"] == report["pf_artifact_sha256"]
    with pytest.raises(ContractError, match="differ beyond tolerance"):
        run_forward_response_conformance(
            fixture_path=repository_root / "fixtures" / "forward_response_conformance.json",
            pf_provider=AnalyticProvider("pf"),
            mle_provider=AnalyticProvider("mle", bias=1e-5),
            output_directory=tmp_path / "fail",
        )


def test_conformance_report_persists_provider_execution_provenance(
    repository_root: Path, tmp_path: Path
) -> None:
    provenance = ForwardResponseProvenance(
        provider_revision="a" * 40,
        expanded_command=("provider", "--output", "/tmp/artifact.npz"),
        provider_config_sha256="b" * 64,
        stdout_sha256="c" * 64,
        stderr_sha256="d" * 64,
    )
    output = tmp_path / "provenance"
    report = run_forward_response_conformance(
        fixture_path=repository_root / "fixtures" / "forward_response_conformance.json",
        pf_provider=AnalyticProvider("pf", provenance=provenance),
        mle_provider=AnalyticProvider("mle", provenance=provenance),
        output_directory=output,
    )
    persisted = load_json(output / "conformance_report.json")
    assert persisted == report
    for provider_name in ("particle_filter", "surface_mle"):
        provider = report["providers"][provider_name]
        assert provider["provider_revision"] == "a" * 40
        assert provider["expanded_command"] == [
            "provider",
            "--output",
            "/tmp/artifact.npz",
        ]
        assert provider["provider_config_sha256"] == "b" * 64
        assert provider["stdout_sha256"] == "c" * 64
        assert provider["stderr_sha256"] == "d" * 64
        assert provider["artifact_sha256"] in {
            report["pf_artifact_sha256"],
            report["mle_artifact_sha256"],
        }


def test_cli_provider_verifies_clean_exact_revision_and_captures_hashes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "provider-repository"
    revision = _create_provider_repository(repository)
    fixture = tmp_path / "fixture.json"
    write_json_atomic(fixture, {"schema_version": 1})
    config_sha256 = "e" * 64
    provider = CLIForwardResponseProvider(
        "particle_filter",
        repository_path=repository,
        revision=revision,
        command_template=(sys.executable, "{repository}/provider.py", "{output_dir}"),
        provider_config_sha256=config_sha256,
    )
    artifact = provider.compute(fixture, tmp_path / "response.npz")
    assert artifact.provenance is not None
    assert artifact.provenance.provider_revision == revision
    assert artifact.provenance.provider_config_sha256 == config_sha256
    assert artifact.provenance.expanded_command == (
        sys.executable,
        (repository / "provider.py").as_posix(),
        (tmp_path / "response.npz").as_posix(),
    )
    assert artifact.provenance.stdout_sha256 == sha256_bytes(b"provider stdout\n")
    assert artifact.provenance.stderr_sha256 == sha256_bytes(b"provider stderr\n")
    assert artifact.sha256 == sha256_file(tmp_path / "response.npz")

    (repository / "provider.py").write_text("raise AssertionError('must not run')\n")
    with pytest.raises(RevisionError, match="dirty code/config paths"):
        provider.compute(fixture, tmp_path / "must-not-exist.npz")
    assert not (tmp_path / "must-not-exist.npz").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "surface_mle"),
        ("revision", "a" * 39),
        ("revision", "A" * 40),
        ("revision_type", "tag"),
        ("require_clean", False),
        ("require_clean", None),
    ],
)
def test_production_provider_config_requires_exact_commit_and_clean_checkout(
    tmp_path: Path, field: str, value: object
) -> None:
    payload: dict[str, object] = {
        "provider": "particle_filter",
        "repository_path": ".",
        "revision": "a" * 40,
        "revision_type": "commit",
        "require_clean": True,
        "command": ["provider", "--output", "{output_dir}"],
    }
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    path = write_json_atomic(tmp_path / f"bad-{field}.json", payload)
    with pytest.raises(ContractError, match="pin an exact 40-character"):
        _provider(path, name="particle_filter")


def test_bundled_production_provider_configs_pin_registry_commits(
    repository_root: Path,
) -> None:
    pins = load_json(repository_root / "PINNED_ESTIMATORS.json")
    for provider_name, pin_name, filename in (
        ("particle_filter", "particle_filter", "pf_production.json"),
        ("surface_mle", "surface_mle", "mle_production.json"),
    ):
        path = repository_root / "configs" / "conformance" / filename
        provider = _provider(path, name=provider_name)
        pin = pins[pin_name]
        assert isinstance(pin, dict)
        assert provider.revision == pin["revision"]
        assert provider.provider_config_sha256 == sha256_file(path)


def test_matching_providers_cannot_share_the_same_wrong_case_order(
    repository_root: Path, tmp_path: Path
) -> None:
    with pytest.raises(ContractError, match="canonical fixture order"):
        run_forward_response_conformance(
            fixture_path=repository_root / "fixtures" / "forward_response_conformance.json",
            pf_provider=AnalyticProvider("pf", reverse_order=True),
            mle_provider=AnalyticProvider("mle", reverse_order=True),
            output_directory=tmp_path / "wrong-order",
        )


def test_forward_response_rejects_non_scalar_vectors(tmp_path: Path) -> None:
    path = tmp_path / "two-dimensional.npz"
    np.savez(
        path,
        case_ids=np.asarray(["case-0", "case-1"]),
        unit_response=np.ones((2, 3), dtype=np.float64),
    )
    with pytest.raises(ContractError, match="one-dimensional vector"):
        load_forward_response("bad-shape", path)


def test_forward_response_rejects_implicit_dtype_coercion(tmp_path: Path) -> None:
    path = tmp_path / "float32.npz"
    np.savez(
        path,
        case_ids=np.asarray(["case-0"]),
        unit_response=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(ContractError, match="exact float64"):
        load_forward_response("bad-dtype", path)
