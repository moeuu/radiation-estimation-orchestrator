from __future__ import annotations

import ast
from pathlib import Path

SIBLING_ESTIMATOR_IMPORTS = ("pf", "three_d_estimation")
RUNTIME_PHYSICS_IMPORTS = ("measurement", "spectrum", "runtime")
RUNTIME_BOUNDARY_FILES = {
    "orchestrator/acquisition.py",
    "orchestrator/estimators/context.py",
    "orchestrator/estimators/forward.py",
}


def _imports(path: Path) -> tuple[tuple[str, bool], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((alias.name, True) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append((node.module, node.level == 0))
    return tuple(modules)


def test_active_code_has_no_sibling_estimator_dependency(repository_root: Path) -> None:
    source = repository_root / "src"
    for path in (source / "orchestrator").rglob("*.py"):
        for module, absolute in _imports(path):
            if not absolute:
                continue
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in SIBLING_ESTIMATOR_IMPORTS
            ), f"sibling estimator import in {path}: {module}"


def test_shared_runtime_physics_enters_through_narrow_boundary(
    repository_root: Path,
) -> None:
    source = repository_root / "src"
    for path in (source / "orchestrator").rglob("*.py"):
        relative = path.relative_to(source).as_posix()
        runtime_imports = [
            module
            for module, absolute in _imports(path)
            if absolute
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in RUNTIME_PHYSICS_IMPORTS
            )
        ]
        if runtime_imports:
            assert relative in RUNTIME_BOUNDARY_FILES, (
                f"runtime physics import escaped the owned boundary in {path}: "
                f"{runtime_imports}"
            )


def test_repository_owns_estimators_but_not_runtime_physics(repository_root: Path) -> None:
    estimator_root = repository_root / "src" / "orchestrator" / "estimators"
    assert (estimator_root / "pf.py").is_file()
    assert (estimator_root / "mle.py").is_file()
    assert (estimator_root / "rj.py").is_file()
    assert not (repository_root / "src" / "measurement").exists()
    assert not (repository_root / "src" / "spectrum").exists()
    assert not (repository_root / "src" / "runtime").exists()
