from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORT_PREFIXES = (
    "pf",
    "three_d_estimation",
    "measurement.continuous_kernels",
    "runtime.measurement_log",
)


def test_orchestrator_imports_no_estimator_or_physics_packages(repository_root: Path) -> None:
    for path in (repository_root / "src" / "orchestrator").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for module in modules
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            ), f"forbidden estimator import in {path}: {modules}"


def test_repository_contains_no_estimator_or_physics_source_copy(repository_root: Path) -> None:
    forbidden_names = {
        "particle_filter.py",
        "continuous_kernels.py",
        "surface_patches.py",
        "response_builder.py",
        "spectral_response_builder.py",
        "estimator.py",
    }
    source_files = list((repository_root / "src").rglob("*.py"))
    assert not forbidden_names.intersection(path.name for path in source_files)
    source_text = "\n".join(path.read_text() for path in source_files)
    for signature in (
        "class ParticleFilter",
        "class SurfaceMLEEstimator",
        "class ContinuousKernel",
    ):
        assert signature not in source_text
    ignored_roots = {".git", ".venv"}
    assert not any(
        path.is_symlink()
        for path in repository_root.rglob("*")
        if path.relative_to(repository_root).parts[0] not in ignored_roots
    )
