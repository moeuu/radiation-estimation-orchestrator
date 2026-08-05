"""Strict deterministic serialization and SHA-256 helpers."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ContractError


def _reject_constant(value: str) -> None:
    raise ContractError(f"Non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json(path: str | Path) -> dict[str, object]:
    """Load one strict UTF-8 JSON object, rejecting duplicates and NaN."""
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read strict JSON object {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{source} must contain a JSON object.")
    return payload


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize strict JSON with stable key ordering and a trailing newline."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Payload is not strict JSON data: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_json_line(payload: object) -> str:
    """Serialize one strict, deterministic JSON object on exactly one line."""
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Payload is not strict JSON data: {exc}") from exc


def write_json_atomic(path: str | Path, payload: object, *, overwrite: bool = False) -> Path:
    """Atomically write canonical JSON without exposing a partial manifest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing file: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_json_idempotent(path: str | Path, payload: object) -> Path:
    """Create deterministic JSON once, accepting only byte-identical retries."""
    target = Path(path)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise ContractError(f"Existing deterministic JSON path is invalid: {target}")
        if target.read_bytes() != encoded:
            raise ContractError(
                f"Deterministic retry payload differs from existing file: {target}"
            )
        return target
    return write_json_atomic(target, payload)


def write_npz_atomic(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic, pickle-free NPZ archive atomically.

    NumPy's convenience writer embeds current ZIP timestamps, which makes
    checkpoint and result hashes change across otherwise identical replays.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing file: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(arrays):
                    if not name or "/" in name or "\\" in name:
                        raise ContractError(f"Invalid NPZ array name: {name!r}")
                    buffer = BytesIO()
                    np.lib.format.write_array(
                        buffer,
                        np.asarray(arrays[name]),
                        version=(2, 0),
                        allow_pickle=False,
                    )
                    entry = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.create_system = 3
                    entry.external_attr = 0o600 << 16
                    archive.writestr(entry, buffer.getvalue())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without following a final symlink."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractError(f"Expected a non-symlink regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(root: str | Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash an explicit, normalized artifact allowlist."""
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for name in sorted(set(relative_paths)):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"Artifact path must be relative and contained: {name!r}")
        target = base / relative
        try:
            target.resolve().relative_to(base)
        except ValueError as exc:
            raise ContractError(f"Artifact escapes root: {name!r}") from exc
        result[relative.as_posix()] = sha256_file(target)
    return result


def directory_inventory(root: str | Path) -> dict[str, str]:
    """Return SHA-256 hashes for every regular file below a directory."""
    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise ContractError(f"Expected a non-symlink directory: {base}")
    names: list[str] = []
    for path in base.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"Symlinks are forbidden in artifact directories: {path}")
        if path.is_file():
            names.append(path.relative_to(base).as_posix())
    return artifact_hashes(base, names)


def inventory_digest(inventory: Mapping[str, str]) -> str:
    """Hash a filename-to-content-hash inventory, preserving path identity."""
    normalized = {str(key): str(value) for key, value in sorted(inventory.items())}
    return sha256_bytes(canonical_json_bytes(normalized))


def hash_directory(root: str | Path) -> str:
    """Return a path-sensitive digest for all files below ``root``."""
    return inventory_digest(directory_inventory(root))


def hash_json_file(path: str | Path) -> str:
    """Hash canonical JSON semantics rather than irrelevant source whitespace."""
    return sha256_bytes(canonical_json_bytes(load_json(path)))


def json_safe(value: Any) -> object:
    """Convert common scalar containers to strict JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ContractError("Non-finite values cannot be serialized.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return [json_safe(child) for child in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    raise ContractError(f"Unsupported JSON value: {type(value).__name__}")
