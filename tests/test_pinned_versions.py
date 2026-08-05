from __future__ import annotations

from pathlib import Path


def test_only_shared_runtime_has_a_repository_pin(repository_root: Path) -> None:
    pins = sorted(path.name for path in repository_root.glob("PINNED_*.json"))
    assert pins == ["PINNED_RUNTIME.json"]


def test_no_sibling_estimator_production_configs(repository_root: Path) -> None:
    assert not list((repository_root / "configs" / "conformance").glob("*.json"))
    assert not list((repository_root / "configs" / "hybrid").glob("*.json"))
