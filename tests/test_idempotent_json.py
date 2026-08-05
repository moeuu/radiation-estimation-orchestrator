from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_line, write_json_idempotent


def test_idempotent_json_accepts_only_exact_retry(tmp_path: Path) -> None:
    path = tmp_path / "directive.json"
    first = write_json_idempotent(path, {"cutoff": 3, "safe": True})
    second = write_json_idempotent(path, {"safe": True, "cutoff": 3})

    assert first == second
    with pytest.raises(ContractError, match="retry payload differs"):
        write_json_idempotent(path, {"cutoff": 4, "safe": True})


def test_canonical_json_line_is_one_strict_deterministic_line() -> None:
    assert canonical_json_line({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    with pytest.raises(ContractError, match="strict JSON"):
        canonical_json_line({"invalid": float("nan")})
