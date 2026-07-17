"""Materialize deterministic, truth-free MeasurementLog prefixes."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from orchestrator.contracts import MeasurementLogInfo, validate_measurement_log
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import (
    canonical_json_bytes,
    directory_inventory,
    inventory_digest,
    load_json,
    sha256_bytes,
    sha256_file,
)

from .prefix import MeasurementPrefix, StationBoundarySchedule

_RECORD_INDEPENDENT_ARRAYS = frozenset({"energy_bin_edges_keV"})


def measurement_records_sha256(
    log: MeasurementLogInfo,
    *,
    record_count: int | None = None,
) -> str:
    """Hash complete ordered estimator-input rows using the shared neutral keys."""
    count = log.record_count if record_count is None else int(record_count)
    if count < 1 or count > log.record_count:
        raise ContractError("record_count must select a non-empty MeasurementLog prefix.")
    metadata_lines = (
        (log.root / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    )
    rows: list[dict[str, object]] = []
    isotopes = log.isotopes
    arrays = log.arrays
    for index in range(count):
        metadata_payload = json.loads(metadata_lines[index])
        metadata = metadata_payload["metadata"]
        variance = (
            np.asarray(arrays["spectrum_variance"][index], dtype=float).tolist()
            if bool(arrays["spectrum_variance_present"][index])
            else None
        )
        isotope_counts: dict[str, float] | None = None
        if bool(arrays["isotope_counts_record_present"][index]):
            isotope_counts = {
                isotope: float(arrays["isotope_counts"][index, isotope_index])
                for isotope_index, isotope in enumerate(isotopes)
                if bool(arrays["isotope_counts_present"][index, isotope_index])
            }
        covariance: dict[str, dict[str, float]] | None = None
        if bool(arrays["isotope_count_covariance_record_present"][index]):
            covariance = {
                row_isotope: {
                    column_isotope: float(
                        arrays["isotope_count_covariance"][index, row_index, column_index]
                    )
                    for column_index, column_isotope in enumerate(isotopes)
                    if bool(
                        arrays["isotope_count_covariance_present"][index, row_index, column_index]
                    )
                }
                for row_index, row_isotope in enumerate(isotopes)
            }
        rows.append(
            {
                "station_id": int(arrays["station_id"][index]),
                "step_id": int(arrays["step_id"][index]),
                "action_id": int(arrays["action_id"][index]),
                "detector_pose_xyz": np.asarray(
                    arrays["detector_pose_xyz"][index], dtype=float
                ).tolist(),
                "detector_quat_wxyz": np.asarray(
                    arrays["detector_quat_wxyz"][index], dtype=float
                ).tolist(),
                "fe_orientation_index": int(arrays["fe_orientation_index"][index]),
                "pb_orientation_index": int(arrays["pb_orientation_index"][index]),
                "live_time_s": float(arrays["live_time_s"][index]),
                "travel_time_s": float(arrays["travel_time_s"][index]),
                "shield_actuation_time_s": float(arrays["shield_actuation_time_s"][index]),
                "spectrum_counts": np.asarray(
                    arrays["spectrum_counts"][index], dtype=float
                ).tolist(),
                "spectrum_variance": variance,
                "energy_bin_edges_keV": np.asarray(
                    arrays["energy_bin_edges_keV"], dtype=float
                ).tolist(),
                "isotope_counts": isotope_counts,
                "isotope_count_covariance": covariance,
                "metadata": metadata,
            }
        )
    return sha256_bytes(canonical_json_bytes(rows))


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a pickle-free NPZ whose bytes do not depend on wall-clock time."""
    with path.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asarray(arrays[name]),
                    version=(2, 0),
                    allow_pickle=False,
                )
                entry = zipfile.ZipInfo(
                    f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_STORED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())


def _copy_all_invariant_artifacts(source: Path, target: Path) -> None:
    """Copy every regular truth-free artifact except prefix-dependent files."""
    excluded = {
        "run_manifest.json",
        "observations.npz",
        "observation_metadata.jsonl",
    }
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"MeasurementLog prefix source contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.as_posix() in excluded:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _prefix_invariant_artifact_paths(source: Path) -> tuple[str, ...]:
    """Resolve only physical-model files referenced by prefix-safe manifests."""
    required = {
        "environment.json",
        "forward_model_manifest.json",
        "repository_commit.txt",
        "runtime_config.resolved.json",
    }
    if (source / "upstream_pf_commit.txt").is_file():
        required.add("upstream_pf_commit.txt")
    queue = [
        source / "environment.json",
        source / "forward_model_manifest.json",
        source / "runtime_config.resolved.json",
    ]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited or not path.is_file() or path.suffix.lower() != ".json":
            continue
        visited.add(path)
        try:
            payload = load_json(path)
        except ContractError:
            continue
        values: list[object] = [payload]
        while values:
            value = values.pop()
            if isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)
            elif isinstance(value, str):
                relative = Path(value)
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                candidate = source / relative
                if candidate.is_file() and not candidate.is_symlink():
                    name = relative.as_posix()
                    if name not in required:
                        required.add(name)
                        queue.append(candidate)
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise ContractError(f"MeasurementLog lacks prefix invariant artifacts: {missing}")
    return tuple(sorted(required))


def _copy_prefix_invariant_artifacts(source: Path, target: Path) -> None:
    """Copy a fail-closed allowlist so arbitrary suffix artifacts cannot affect IDs."""
    for name in _prefix_invariant_artifact_paths(source):
        path = source / name
        if path.is_symlink():
            raise ContractError(f"MeasurementLog prefix source contains a symlink: {path}")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _prefix_manifest(
    source_log: MeasurementLogInfo,
    *,
    record_count: int,
    observations_sha256: str,
    observation_metadata_sha256: str,
) -> dict[str, object]:
    """Build the identical causal-prefix manifest for preview and publication."""
    manifest = load_json(source_log.root / "run_manifest.json")
    manifest["record_count"] = int(record_count)
    metadata = manifest.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("run_manifest.metadata must be an object.")
    # The full replay schedule is a controller preflight artifact.  Carrying its
    # digest into a shorter prefix would make causal identities depend on an
    # unseen suffix.  Prefixes attest only the station-end markers in their own
    # selected metadata rows.
    metadata.pop("station_boundary_attestation", None)
    metadata.pop("station_boundary_schedule_sha256", None)
    metadata.pop("station_boundary_schedule", None)
    covered_boundaries = _covered_station_boundaries(source_log, record_count=record_count)
    metadata["measurement_log_prefix"] = {
        "schema_version": 1,
        "source_run_id": str(source_log.manifest["run_id"]),
        "data_cutoff_step": int(source_log.step_ids[record_count - 1]),
        "data_cutoff_station": int(source_log.station_ids[record_count - 1]),
        "station_boundary_attestation": "covered_prefix_markers_v1",
        "covered_station_boundaries_sha256": covered_boundaries.schedule_sha256,
    }
    declared = manifest.get("artifact_hashes")
    if not isinstance(declared, dict):
        raise ContractError("MeasurementLog artifact_hashes must be an object.")
    invariant_names = set(_prefix_invariant_artifact_paths(source_log.root))
    manifest["artifact_hashes"] = {
        **{name: digest for name, digest in declared.items() if name in invariant_names},
        "observations.npz": observations_sha256,
        "observation_metadata.jsonl": observation_metadata_sha256,
    }
    return manifest


def _covered_station_boundaries(
    source_log: MeasurementLogInfo,
    *,
    record_count: int,
) -> StationBoundarySchedule:
    """Derive only explicit station-end markers contained in a selected prefix."""
    lines = (
        (source_log.root / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    )
    boundaries: list[tuple[int, int]] = []
    for index, line in enumerate(lines[:record_count]):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid observation metadata line {index + 1}.") from exc
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ContractError("Observation metadata rows must contain metadata objects.")
        metadata = row["metadata"]
        if metadata.get("station_complete") is True:
            boundaries.append((int(row["station_id"]), int(row["step_id"])))
    cutoff = int(source_log.step_ids[record_count - 1])
    if not boundaries or boundaries[-1][1] != cutoff:
        raise DataReuseError("A causal prefix cutoff must be its declared station terminal step.")
    return StationBoundarySchedule.create(
        source_run_id=str(source_log.manifest["run_id"]),
        station_end_steps=tuple(boundaries),
    )


def materialize_station_marked_log(
    source_log: MeasurementLogInfo,
    station_boundaries: StationBoundarySchedule,
    output_directory: str | Path,
) -> tuple[MeasurementLogInfo, StationBoundarySchedule]:
    """Publish a full log carrying a predeclared station-end marker per row.

    This is an offline replay adapter for logs written before the marker became
    mandatory.  The boundary schedule is supplied and hash-bound before replay;
    this function never infers a boundary from a later observation.
    """
    source_run_id = str(source_log.manifest["run_id"])
    if station_boundaries.source_run_id != source_run_id:
        raise DataReuseError("Station-boundary schedule is bound to a different source run.")
    terminal_by_station = dict(station_boundaries.station_end_steps)
    target = Path(output_directory).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace station-marked log {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale station-marked staging path: {temporary}")
    temporary.mkdir()
    try:
        _copy_all_invariant_artifacts(source_log.root, temporary)
        shutil.copyfile(source_log.root / "observations.npz", temporary / "observations.npz")
        raw_lines = (
            (source_log.root / "observation_metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        marked_lines: list[bytes] = []
        for index, line in enumerate(raw_lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"Invalid observation metadata line {index + 1}.") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
                raise ContractError("Observation metadata rows must contain metadata objects.")
            step_id = int(payload.get("step_id", -1))
            station_id = int(payload.get("station_id", -1))
            if station_id not in terminal_by_station:
                raise DataReuseError(f"Station {station_id} is absent from boundary schedule.")
            payload["metadata"]["station_complete"] = step_id == int(
                terminal_by_station[station_id]
            )
            marked_lines.append(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        (temporary / "observation_metadata.jsonl").write_bytes(b"\n".join(marked_lines) + b"\n")
        manifest = load_json(source_log.root / "run_manifest.json")
        metadata = manifest.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ContractError("run_manifest.metadata must be an object.")
        metadata["station_boundary_attestation"] = {
            "kind": "predeclared_schedule_v1",
            "source_run_id": source_run_id,
            "schedule_sha256": station_boundaries.schedule_sha256,
        }
        hashes = manifest.get("artifact_hashes")
        if not isinstance(hashes, dict):
            raise ContractError("MeasurementLog artifact_hashes must be an object.")
        hashes["observation_metadata.jsonl"] = sha256_file(temporary / "observation_metadata.jsonl")
        manifest["artifact_hashes"] = hashes
        (temporary / "run_manifest.json").write_bytes(canonical_json_bytes(manifest))
        marked = validate_measurement_log(temporary)
        os.replace(temporary, target)
        published = validate_measurement_log(target)
        if published.measurement_log_sha256 != marked.measurement_log_sha256:
            raise ContractError("Station-marked log changed while being published.")
        return published, station_boundaries
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def materialize_measurement_prefix(
    source_log: MeasurementLogInfo,
    prefix: MeasurementPrefix,
    output_directory: str | Path,
) -> MeasurementLogInfo:
    """Publish the exact declared leading rows as a valid MeasurementLog.

    ``prefix`` must already be bound to the source log and to a predeclared
    station boundary.  No suffix row is consulted when selecting content.
    """
    if prefix.source_run_id != str(source_log.manifest["run_id"]):
        raise DataReuseError("Measurement prefix is bound to a different source run.")
    source_log_steps = source_log.step_ids[: len(prefix.covered_step_ids)]
    if source_log_steps != prefix.covered_step_ids:
        raise DataReuseError("Measurement prefix is not the exact source-log prefix.")
    if not prefix.cutoff_station_complete:
        raise DataReuseError("Measurement prefix cutoff must be station-complete.")

    target = Path(output_directory).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace MeasurementLog prefix {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale MeasurementLog prefix staging path: {temporary}")
    temporary.mkdir()
    try:
        source = source_log.root
        _copy_prefix_invariant_artifacts(source, temporary)
        record_count = len(prefix.covered_step_ids)
        sliced_arrays = {
            name: (
                np.asarray(value).copy()
                if name in _RECORD_INDEPENDENT_ARRAYS
                else np.asarray(value)[:record_count].copy()
            )
            for name, value in source_log.arrays.items()
        }
        _write_deterministic_npz(temporary / "observations.npz", sliced_arrays)

        metadata_lines = (source / "observation_metadata.jsonl").read_bytes().splitlines()
        if len(metadata_lines) < record_count:
            raise ContractError("MeasurementLog metadata is shorter than its declared prefix.")
        metadata_bytes = b"\n".join(metadata_lines[:record_count]) + b"\n"
        (temporary / "observation_metadata.jsonl").write_bytes(metadata_bytes)

        manifest = _prefix_manifest(
            source_log,
            record_count=record_count,
            observations_sha256=sha256_file(temporary / "observations.npz"),
            observation_metadata_sha256=sha256_file(temporary / "observation_metadata.jsonl"),
        )
        (temporary / "run_manifest.json").write_bytes(canonical_json_bytes(manifest))

        validated = validate_measurement_log(temporary)
        if validated.step_ids != prefix.covered_step_ids:
            raise DataReuseError("Materialized MeasurementLog has unexpected step coverage.")
        if validated.station_ids != prefix.covered_station_ids:
            raise DataReuseError("Materialized MeasurementLog has unexpected station coverage.")
        if validated.measurement_log_sha256 != prefix.prefix_measurement_log_sha256:
            raise DataReuseError(
                "Materialized prefix digest differs from its declared prefix digest."
            )
        os.replace(temporary, target)
        return validate_measurement_log(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def preview_measurement_prefix_sha256(
    source_log: MeasurementLogInfo,
    *,
    record_count: int,
    temporary_parent: str | Path,
) -> str:
    """Materialize a private preview and return its canonical directory digest.

    The helper is intentionally separate from publication so a caller can bind
    ``MeasurementPrefix`` to the exact prefix hash before publishing it.
    """
    count = int(record_count)
    if count < 1 or count > source_log.record_count:
        raise ContractError("record_count must select a non-empty source-log prefix.")
    # A minimal local declaration is used only to compute deterministic bytes;
    # station completeness remains the caller's separately validated authority.
    parent = Path(temporary_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    preview = parent / f".prefix-preview-{os.getpid()}-{count}"
    if preview.exists():
        raise FileExistsError(f"Prefix preview already exists: {preview}")
    preview.mkdir()
    try:
        _copy_prefix_invariant_artifacts(source_log.root, preview)
        arrays = {
            name: (
                np.asarray(value).copy()
                if name in _RECORD_INDEPENDENT_ARRAYS
                else np.asarray(value)[:count].copy()
            )
            for name, value in source_log.arrays.items()
        }
        _write_deterministic_npz(preview / "observations.npz", arrays)
        lines = (source_log.root / "observation_metadata.jsonl").read_bytes().splitlines()
        (preview / "observation_metadata.jsonl").write_bytes(b"\n".join(lines[:count]) + b"\n")
        manifest = _prefix_manifest(
            source_log,
            record_count=count,
            observations_sha256=sha256_file(preview / "observations.npz"),
            observation_metadata_sha256=sha256_file(preview / "observation_metadata.jsonl"),
        )
        (preview / "run_manifest.json").write_bytes(canonical_json_bytes(manifest))
        validate_measurement_log(preview)
        return inventory_digest(directory_inventory(preview))
    finally:
        shutil.rmtree(preview, ignore_errors=True)


def build_and_materialize_measurement_prefix(
    source_log: MeasurementLogInfo,
    *,
    cutoff_step: int,
    station_boundaries: StationBoundarySchedule,
    station_complete_marker: bool,
    output_directory: str | Path,
) -> tuple[MeasurementPrefix, MeasurementLogInfo]:
    """Bind and publish one exact prefix without a hash/serialization gap."""
    try:
        cutoff_index = source_log.step_ids.index(int(cutoff_step))
    except ValueError as exc:
        raise ContractError(f"Cutoff step {cutoff_step} is absent from MeasurementLog.") from exc
    output = Path(output_directory).resolve()
    digest = preview_measurement_prefix_sha256(
        source_log,
        record_count=cutoff_index + 1,
        temporary_parent=output.parent,
    )
    prefix = MeasurementPrefix.from_measurement_log(
        source_log,
        cutoff_step=int(cutoff_step),
        prefix_measurement_log_sha256=digest,
        covered_records_sha256=measurement_records_sha256(
            source_log,
            record_count=cutoff_index + 1,
        ),
        station_boundaries=station_boundaries,
        station_complete_marker=bool(station_complete_marker),
    )
    return prefix, materialize_measurement_prefix(source_log, prefix, output)


__all__ = [
    "build_and_materialize_measurement_prefix",
    "materialize_measurement_prefix",
    "materialize_station_marked_log",
    "measurement_records_sha256",
    "preview_measurement_prefix_sha256",
]
