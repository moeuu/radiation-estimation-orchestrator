"""Contract-complete subprocess test double; it imports no orchestrator code."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_hash(root: Path) -> str:
    inventory = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    payload = (
        json.dumps(
            inventory,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    return sha256(payload).hexdigest()


def json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def write_json(path: Path, payload: object) -> None:
    path.write_bytes(json_bytes(payload))


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), version=(2, 0), allow_pickle=False
                )
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())


def assert_truth_isolated(log: Path) -> None:
    if os.environ.get("RSE_TRUTH_ACCESS") != "forbidden":
        raise RuntimeError("Orchestrator did not set the truth-access guard.")
    forbidden = [path for path in log.rglob("*") if path.is_file() and "truth" in path.name]
    if forbidden:
        raise RuntimeError(f"Estimator received truth files: {forbidden}")


def provenance(args: argparse.Namespace, *, family: str, variant: str) -> dict[str, object]:
    file_digest = file_hash(args.config)
    resolved_payload = json.loads(args.config.read_text(encoding="utf-8"))
    log_manifest = json.loads((args.log / "run_manifest.json").read_text(encoding="utf-8"))
    resolved_digest = sha256(json_bytes(resolved_payload)).hexdigest()
    base = {
        "estimator_family": family,
        "estimator_variant": variant,
        "estimator_repository": f"fake://{family}",
        "estimator_commit": args.commit,
        "measurement_log_schema_version": int(log_manifest["schema_version"]),
        "measurement_log_sha256": log_hash(args.log),
        "config_sha256": file_digest,
        "command": list(os.sys.argv),
    }
    if family == "particle_filter":
        base.update(
            {
                "resolved_config_sha256": resolved_digest,
                "random_seed": int(args.seed),
                "planner_belief_sources": [
                    "pf_posterior",
                    "pf_tentative",
                ],
                "batch_feedback_applied": False,
            }
        )
    else:
        base.update(
            {
                "resolved_estimator_config_sha256": resolved_digest,
                "uses_pf_state": False,
                "uses_pf_candidates": False,
                "candidate_domain": "complete_surface_dictionary",
            }
        )
    return base


def pf_result(args: argparse.Namespace) -> None:
    manifest = json.loads((args.log / "run_manifest.json").read_text())
    with np.load(args.log / "observations.npz", allow_pickle=False) as archive:
        steps = np.asarray(archive["step_id"], dtype=int)
        stations = np.asarray(archive["station_id"], dtype=int)
    prov = provenance(args, family="particle_filter", variant=args.profile)
    measurement_log_schema_version = int(manifest["schema_version"])
    modes = {
        "Cs-137": ([1.08, 0.95, 0.04], 118000.0),
        "Co-60": ([4.92, 4.55, 2.94], 92500.0),
        "Eu-154": ([3.10, 2.92, 1.25], 63000.0),
    }
    isotopes = {
        isotope: {
            "map_cardinality": 1,
            "cardinality_distribution": {"0": 0.03, "1": 0.97},
            "modes": [
                {
                    "position_mean_xyz": position,
                    "position_covariance_xyz": [
                        [0.04, 0.0, 0.0],
                        [0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.06],
                    ],
                    "strength_mean_cps_1m": strength,
                    "posterior_mass": 0.9,
                }
            ],
        }
        for isotope, (position, strength) in modes.items()
    }
    posterior = {
        "schema_version": 1,
        "estimator_family": "particle_filter",
        "estimator_variant": args.profile,
        "final_estimate_source": "pf_posterior",
        "uses_all_history_batch_fit": False,
        "uses_surface_map": False,
        "uses_batch_model_order": False,
        "isotopes": isotopes,
        "provenance": prov,
    }
    write_json(args.output / "pf_posterior.json", posterior)
    with (args.output / "pf_trace.jsonl").open("x", encoding="utf-8") as handle:
        for index, (step, station) in enumerate(zip(steps, stations, strict=True)):
            hypotheses = [
                {
                    "hypothesis_id": f"{isotope}-mode-0",
                    "isotope": isotope,
                    "position_mean_xyz": value[0],
                    "posterior_mass": min(0.9, 0.2 + 0.06 * index),
                }
                for isotope, value in modes.items()
            ]
            line = {
                "schema_version": 2 if measurement_log_schema_version == 2 else 1,
                "estimator_family": (
                    "pure_particle_filter"
                    if measurement_log_schema_version == 2
                    else "particle_filter"
                ),
                "step_id": int(step),
                "station_id": int(station),
                "predictive_deviance": float(max(0, 18 - index)),
                "pf_hypotheses": hypotheses,
            }
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(
        args.output / "pf_diagnostics.json",
        {
            "schema_version": 2 if measurement_log_schema_version == 2 else 1,
            "record_count": int(manifest["record_count"]),
            "measurement_log_schema_version": int(manifest["schema_version"]),
            "measurement_log_sha256": prov["measurement_log_sha256"],
            "resolved_config_sha256": prov["resolved_config_sha256"],
            "estimator_commit": args.commit,
            "truth_read": False,
        },
    )


def mle_result(args: argparse.Namespace, mode: str) -> None:
    config = json.loads(args.config.read_text())
    manifest = json.loads((args.log / "run_manifest.json").read_text(encoding="utf-8"))
    with np.load(args.log / "observations.npz", allow_pickle=False) as observation_archive:
        spectrum_shape = np.asarray(observation_archive["spectrum_counts"]).shape
    isotope_names = tuple(config.get("isotope_names", manifest["isotopes"]))
    prov = provenance(args, family="surface_mle", variant=mode)
    clusters = [
        {
            "isotope": "Cs-137",
            "cluster_id": 0,
            "patch_ids": [0],
            "centroid_xyz": [1.1, 1.0, 0.0],
            "integrated_strength_cps_1m": 117500.0,
            "peak_density_cps_1m_m2": 117500.0,
            "surface_kinds": ["floor"],
        },
        {
            "isotope": "Co-60",
            "cluster_id": 1,
            "patch_ids": [1],
            "centroid_xyz": [5.0, 4.4, 3.0],
            "integrated_strength_cps_1m": 91500.0,
            "peak_density_cps_1m_m2": 91500.0,
            "surface_kinds": ["ceiling"],
        },
        {
            "isotope": "Eu-154",
            "cluster_id": 2,
            "patch_ids": [2],
            "centroid_xyz": [3.05, 3.05, 1.2],
            "integrated_strength_cps_1m": 64000.0,
            "peak_density_cps_1m_m2": 64000.0,
            "surface_kinds": ["obstacle_top"],
        },
    ]
    diagnostics = {
        "schema_version": 1,
        "isotope_names": list(isotope_names),
        "patch_count": 6,
        "predicted_spectra_present": mode == "spectral",
        "predicted_isotope_counts_present": True,
        "objective_value": 123.5 if mode == "count" else 121.0,
        "poisson_deviance": 8.5 if mode == "count" else 7.8,
        "iterations": 42,
        "converged": True,
        "diagnostics": {
            "mode": mode,
            "held_out_poisson_deviance": 4.25 if mode == "count" else 3.9,
            "hotspot_clusters": clusters,
            "provenance": prov,
        },
        "config": config,
    }
    diagnostic_bytes = json_bytes(diagnostics)
    (args.output / "mle_diagnostics.json").write_bytes(diagnostic_bytes)
    write_json(
        args.output / "hotspot_clusters.json",
        {"schema_version": 1, "hotspot_clusters": clusters},
    )
    centroids = np.asarray(
        [
            [1.1, 1.0, 0.0],
            [5.0, 4.4, 3.0],
            [3.05, 3.05, 1.2],
            [0.0, 2.0, 1.0],
            [6.0, 2.0, 1.0],
            [2.0, 6.0, 1.0],
        ],
        dtype=np.float64,
    )
    strengths = np.zeros((3, 6), dtype=np.float64)
    strengths[0, 0] = 117500.0
    strengths[1, 1] = 91500.0
    strengths[2, 2] = 64000.0
    arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "diagnostics_sha256": np.asarray(sha256(diagnostic_bytes).hexdigest()),
            "isotope_names": np.asarray(isotope_names),
            "patch_ids": np.arange(6, dtype=np.int64),
            "patch_centroids_xyz": centroids,
            "patch_surface_kinds": np.asarray(
                ["floor", "ceiling", "obstacle_top", "wall", "wall", "wall"]
            ),
            "patch_strength_by_isotope": strengths,
            "objective_value": np.asarray(diagnostics["objective_value"], dtype=np.float64),
            "poisson_deviance": np.asarray(diagnostics["poisson_deviance"], dtype=np.float64),
            "iterations": np.asarray(42, dtype=np.int64),
            "converged": np.asarray(1, dtype=np.uint8),
            "patch_count": np.asarray(6, dtype=np.int64),
    }
    if mode == "spectral":
        arrays["predicted_spectra"] = np.ones(spectrum_shape, dtype=np.float64)
    write_npz(args.output / "mle_estimate.npz", arrays)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--kind", choices=("pf", "mle"), required=True)
    result.add_argument("--mode", default="replay")
    result.add_argument("--log", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--profile", default="pf_strict")
    result.add_argument("--seed", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    args.log = args.log.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    assert_truth_isolated(args.log)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.kind == "pf":
        pf_result(args)
    else:
        mode = "spectral" if args.mode in {"spectral", "fit-spectrum"} else "count"
        mle_result(args, mode)


if __name__ == "__main__":
    main()
