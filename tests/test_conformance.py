from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from orchestrator.conformance import (
    ForwardResponseArtifact,
    ForwardResponseProvenance,
    canonical_conformance_case_ids,
    expected_conformance_case_count,
    load_forward_response,
    run_forward_response_conformance,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import load_json


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


def test_conformance_fixture_covers_all_required_axes(repository_root: Path) -> None:
    fixture = repository_root / "fixtures" / "forward_response_conformance.json"
    assert expected_conformance_case_count(fixture) == 3 * 3 * 64 * 4 * 2
    case_ids = canonical_conformance_case_ids(fixture)
    assert len(case_ids) == 4608
    assert len(set(case_ids)) == len(case_ids)
    assert "fe=00|pb=00" in case_ids[0]
    assert any("fe=07|pb=07" in value for value in case_ids)


def test_forward_response_comparison_engine_detects_mismatch(
    repository_root: Path, tmp_path: Path
) -> None:
    fixture = repository_root / "fixtures" / "forward_response_conformance.json"
    report = run_forward_response_conformance(
        fixture_path=fixture,
        pf_provider=AnalyticProvider("pf"),
        mle_provider=AnalyticProvider("mle"),
        output_directory=tmp_path / "pass",
    )
    assert report["passed"] is True
    with pytest.raises(ContractError, match="differ beyond tolerance"):
        run_forward_response_conformance(
            fixture_path=fixture,
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
    assert load_json(output / "conformance_report.json") == report
    assert report["providers"]["particle_filter"]["provider_revision"] == "a" * 40


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


@pytest.mark.parametrize(("dtype", "shape"), [(np.float32, (2,)), (np.float64, (2, 3))])
def test_forward_response_requires_float64_vector(
    tmp_path: Path, dtype: type[np.floating], shape: tuple[int, ...]
) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        case_ids=np.asarray(["case-0", "case-1"]),
        unit_response=np.ones(shape, dtype=dtype),
    )
    with pytest.raises(ContractError):
        load_forward_response("invalid", path)
