from __future__ import annotations

import subprocess
from pathlib import Path

from orchestrator.adapters import load_estimator_pins


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_estimator_registry_pins_full_existing_local_commits(repository_root: Path) -> None:
    pins = load_estimator_pins(repository_root / "PINNED_ESTIMATORS.json")
    expected = {
        "particle_filter": repository_root.parent / "Rotating-shield-particle-filter",
        "surface_mle": repository_root.parent / "radiation-surface-mle-estimator",
    }
    for name, checkout in expected.items():
        pin = pins[name]
        assert len(pin.revision) == 40
        assert _git(checkout, "cat-file", "-t", pin.revision) == "commit"
        assert _git(checkout, "rev-parse", "HEAD") == pin.revision
        assert pin.release_tag is None
        assert pin.revision_type == "commit"


def test_registry_repositories_are_upstream_urls(repository_root: Path) -> None:
    pins = load_estimator_pins(repository_root / "PINNED_ESTIMATORS.json")
    assert pins["particle_filter"].repository.endswith("Rotating-shield-particle-filter.git")
    assert pins["surface_mle"].repository.endswith(
        "radiation-surface-mle-estimator.git"
    )
