from __future__ import annotations

import re
import subprocess
from pathlib import Path

from orchestrator.hashing import load_json


def test_shared_runtime_pin_is_exact_and_locally_resolvable(repository_root: Path) -> None:
    payload = load_json(repository_root / "PINNED_RUNTIME.json")

    assert payload["schema_version"] == 1
    revision = payload["revision"]
    assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    runtime = (repository_root / str(payload["local_path_hint"])).resolve()
    completed = subprocess.run(
        ("git", "cat-file", "-e", f"{revision}^{{commit}}"),
        cwd=runtime,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert payload["required_measurement_log_schema_version"] == 2
    assert payload["required_live_protocol_version"] == 2
