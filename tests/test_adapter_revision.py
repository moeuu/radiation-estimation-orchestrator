from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.adapters import EstimatorPin
from orchestrator.adapters.base import verify_repository_revision
from orchestrator.errors import RevisionError


def _run(root: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)


def test_revision_check_allows_only_inventoried_artifact_prefixes(tmp_path: Path) -> None:
    repository = tmp_path / "estimator"
    repository.mkdir()
    _run(repository, "init", "-b", "main")
    _run(repository, "config", "user.email", "test@example.invalid")
    _run(repository, "config", "user.name", "Test")
    (repository / "src").mkdir()
    (repository / "src" / "model.py").write_text("VALUE = 1\n")
    _run(repository, "add", "src/model.py")
    _run(repository, "commit", "-m", "initial")
    revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pin = EstimatorPin("particle_filter", "fake", revision, "commit", None, None, 1, 1)
    (repository / "results").mkdir()
    (repository / "results" / "output.json").write_text("{}\n")
    observed, dirty = verify_repository_revision(
        repository, pin, require_clean=True, allowed_dirty_prefixes=("results/",)
    )
    assert observed == revision
    assert "results/output.json" in dirty
    (repository / "src" / "model.py").write_text("VALUE = 2\n")
    with pytest.raises(RevisionError, match="outside the explicit allowlist"):
        verify_repository_revision(
            repository, pin, require_clean=True, allowed_dirty_prefixes=("results/",)
        )


def test_revision_check_cannot_hide_code_as_rename_from_allowed_output(tmp_path: Path) -> None:
    repository = tmp_path / "estimator"
    repository.mkdir()
    _run(repository, "init", "-b", "main")
    _run(repository, "config", "user.email", "test@example.invalid")
    _run(repository, "config", "user.name", "Test")
    (repository / "results").mkdir()
    (repository / "results" / "tracked.py").write_text("VALUE = 1\n")
    _run(repository, "add", "results/tracked.py")
    _run(repository, "commit", "-m", "initial")
    revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pin = EstimatorPin("particle_filter", "fake", revision, "commit", None, None, 1, 1)
    (repository / "src").mkdir()
    _run(repository, "mv", "results/tracked.py", "src/tracked.py")
    with pytest.raises(RevisionError, match=r"src/tracked\.py"):
        verify_repository_revision(
            repository,
            pin,
            require_clean=True,
            allowed_dirty_prefixes=("results/",),
        )
