"""Typed construction of exact-prefix MLE snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, json_safe, sha256_bytes

from .prefix import MeasurementPrefix
from .scheduler import HybridTrigger


@dataclass(frozen=True, slots=True)
class SnapshotCluster:
    """One complete-surface MLE cluster exposed as a candidate, not a PF fact."""

    snapshot_candidate_id: str
    cluster_id: int
    isotope: str
    centroid_xyz: tuple[float, float, float]
    integrated_strength_cps_1m: float
    surface_kinds: tuple[str, ...]
    patch_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized_id = self.snapshot_candidate_id.replace("-", "").replace("_", "")
        normalized_id = normalized_id.replace(".", "")
        if not self.snapshot_candidate_id or not normalized_id.isalnum():
            raise ContractError("snapshot_candidate_id contains unsupported characters.")
        if self.cluster_id < 0 or not self.isotope:
            raise ContractError("Snapshot cluster ID and isotope are invalid.")
        if len(self.centroid_xyz) != 3 or not all(isfinite(value) for value in self.centroid_xyz):
            raise ContractError("Snapshot cluster centroid must contain three finite values.")
        if not isfinite(self.integrated_strength_cps_1m) or self.integrated_strength_cps_1m < 0:
            raise ContractError("Snapshot cluster strength must be finite and nonnegative.")
        if not self.surface_kinds or len(set(self.surface_kinds)) != len(self.surface_kinds):
            raise ContractError("Snapshot cluster surface kinds must be nonempty and unique.")
        if not self.patch_ids or any(value < 0 for value in self.patch_ids):
            raise ContractError("Snapshot cluster patch IDs must be nonempty and nonnegative.")
        if len(set(self.patch_ids)) != len(self.patch_ids):
            raise ContractError("Snapshot cluster patch IDs must be unique.")

    def to_dict(self) -> dict[str, object]:
        """Return the v2 contract representation."""
        return {
            "snapshot_candidate_id": self.snapshot_candidate_id,
            "cluster_id": self.cluster_id,
            "isotope": self.isotope,
            "centroid_xyz": list(self.centroid_xyz),
            "integrated_strength_cps_1m": self.integrated_strength_cps_1m,
            "surface_kinds": list(self.surface_kinds),
            "patch_ids": list(self.patch_ids),
        }


@dataclass(frozen=True, slots=True)
class SnapshotPrediction:
    """MLE prediction for one already-covered row, used for diagnostics only."""

    step_id: int
    isotope_counts: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.step_id < 0 or not self.isotope_counts:
            raise ContractError("Snapshot prediction requires a nonnegative step and counts.")
        normalized: dict[str, float] = {}
        for isotope, raw_value in self.isotope_counts.items():
            value = float(raw_value)
            if not isotope or not isfinite(value) or value < 0:
                raise ContractError("Snapshot predicted counts must be finite and nonnegative.")
            normalized[str(isotope)] = value
        object.__setattr__(self, "isotope_counts", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, object]:
        """Return the v2 contract representation."""
        return {"step_id": self.step_id, "isotope_counts": dict(self.isotope_counts)}


@dataclass(frozen=True, slots=True)
class MLESnapshot:
    """All-history MLE output bound to one exact station-complete prefix."""

    snapshot_id: str
    trigger_id: str
    estimator_variant: str
    prefix: MeasurementPrefix
    mle_result_sha256: str
    warm_start_snapshot_id: str | None
    warm_start_mle_result_sha256: str | None
    clusters: tuple[SnapshotCluster, ...]
    predictions: tuple[SnapshotPrediction, ...]
    fit_diagnostics: Mapping[str, object]
    provenance: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        trigger: HybridTrigger,
        prefix: MeasurementPrefix,
        estimator_variant: str,
        mle_result_sha256: str,
        clusters: Sequence[SnapshotCluster],
        predictions: Sequence[SnapshotPrediction],
        fit_diagnostics: Mapping[str, object],
        provenance: Mapping[str, object],
        warm_start_snapshot_id: str | None = None,
        warm_start_mle_result_sha256: str | None = None,
    ) -> MLESnapshot:
        """Create a snapshot after checking trigger, prefix, and prediction coverage."""
        if estimator_variant not in {"count", "spectral"}:
            raise ContractError("MLESnapshot estimator_variant must be count or spectral.")
        if trigger.data_cutoff_step != prefix.data_cutoff_step:
            raise DataReuseError("Trigger and MeasurementPrefix cutoff steps differ.")
        if trigger.data_cutoff_station != prefix.data_cutoff_station:
            raise DataReuseError("Trigger and MeasurementPrefix cutoff stations differ.")
        if not trigger.station_complete or not prefix.cutoff_station_complete:
            raise DataReuseError("MLESnapshot cutoff must carry a station-complete marker.")
        if len(mle_result_sha256) != 64:
            raise ContractError("mle_result_sha256 must be a SHA-256 digest.")
        normalized_clusters = tuple(clusters)
        candidate_ids = [cluster.snapshot_candidate_id for cluster in normalized_clusters]
        cluster_ids = [cluster.cluster_id for cluster in normalized_clusters]
        if len(candidate_ids) != len(set(candidate_ids)) or len(cluster_ids) != len(
            set(cluster_ids)
        ):
            raise ContractError("Snapshot candidate and cluster IDs must be unique.")
        normalized_predictions = tuple(predictions)
        prediction_steps = tuple(prediction.step_id for prediction in normalized_predictions)
        prefix.assert_exact_coverage(prediction_steps)
        warm_used = warm_start_snapshot_id is not None or warm_start_mle_result_sha256 is not None
        if warm_used and (warm_start_snapshot_id is None or warm_start_mle_result_sha256 is None):
            raise ContractError(
                "Warm-start snapshot ID and result digest must be supplied together."
            )
        identity = {
            "schema_version": 2,
            "trigger_id": trigger.trigger_id,
            "source_run_id": prefix.source_run_id,
            "estimator_variant": estimator_variant,
            "data_cutoff_step": prefix.data_cutoff_step,
            "prefix_measurement_log_sha256": prefix.prefix_measurement_log_sha256,
            "covered_records_sha256": prefix.covered_records_sha256,
            "covered_station_boundaries_sha256": prefix.covered_station_boundaries_sha256,
            "mle_result_sha256": mle_result_sha256,
        }
        snapshot_id = f"snapshot-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        return cls(
            snapshot_id=snapshot_id,
            trigger_id=trigger.trigger_id,
            estimator_variant=estimator_variant,
            prefix=prefix,
            mle_result_sha256=mle_result_sha256,
            warm_start_snapshot_id=warm_start_snapshot_id,
            warm_start_mle_result_sha256=warm_start_mle_result_sha256,
            clusters=normalized_clusters,
            predictions=normalized_predictions,
            fit_diagnostics=MappingProxyType(dict(json_safe(fit_diagnostics))),  # type: ignore[arg-type]
            provenance=MappingProxyType(dict(json_safe(provenance))),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical MLESnapshot v2 JSON data."""
        warm_used = self.warm_start_snapshot_id is not None
        return {
            "schema_version": 2,
            "snapshot_id": self.snapshot_id,
            "trigger_id": self.trigger_id,
            "estimator_family": "surface_mle",
            "estimator_variant": self.estimator_variant,
            "data_cutoff_step": self.prefix.data_cutoff_step,
            "data_cutoff_station": self.prefix.data_cutoff_station,
            "cutoff_station_complete": self.prefix.cutoff_station_complete,
            "covered_step_ids": list(self.prefix.covered_step_ids),
            "source_run_id": self.prefix.source_run_id,
            "prefix_measurement_log_sha256": self.prefix.prefix_measurement_log_sha256,
            "covered_records_sha256": self.prefix.covered_records_sha256,
            "covered_station_boundaries_sha256": (self.prefix.covered_station_boundaries_sha256),
            "mle_result_sha256": self.mle_result_sha256,
            "warm_start": {
                "used": warm_used,
                "snapshot_id": self.warm_start_snapshot_id,
                "mle_result_sha256": self.warm_start_mle_result_sha256,
            },
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "predicted_observations": [prediction.to_dict() for prediction in self.predictions],
            "fit_diagnostics": dict(self.fit_diagnostics),
            "safety": {
                "direct_mle_objective_reweight": False,
                "hard_prune_authorized": False,
            },
            "provenance": dict(self.provenance),
        }

    @property
    def sha256(self) -> str:
        """Hash canonical snapshot semantics."""
        return sha256_bytes(canonical_json_bytes(self.to_dict()))
