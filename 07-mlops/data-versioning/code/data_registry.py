"""
Data Versioning & Feature Store Engine:
1. Content-addressable dataset hashing and DVC-style manifest generation.
2. Feature Store with Online low-latency lookup and Offline Point-in-Time correctness join (leakage-free time-travel).
3. Data lineage DAG tracking dependency changes between stages.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 1. Content-Addressable Data Versioning (DVC Pattern)
# --------------------------------------------------------------------------- #
def compute_content_hash(data_bytes: bytes) -> str:
    """Computes SHA-256 hash representing unique data content."""
    return hashlib.sha256(data_bytes).hexdigest()


@dataclass
class DVCManifest:
    path: str
    content_hash: str
    size_bytes: int
    remote_storage_key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "hash": self.content_hash,
            "size": self.size_bytes,
            "remote_key": self.remote_storage_key
        }


class DataVersionControl:
    """Simulates DVC: tracks data files via pointers while storing actual payloads in content-addressed cache."""
    def __init__(self):
        self.remote_cache: Dict[str, bytes] = {}  # hash -> data bytes

    def track_file(self, file_path: str, data: bytes) -> DVCManifest:
        chash = compute_content_hash(data)
        # Store in remote cache indexed by hash
        self.remote_cache[chash] = data
        remote_key = f"s3://ml-data-bucket/{chash[:2]}/{chash[2:]}"
        return DVCManifest(
            path=file_path,
            content_hash=chash,
            size_bytes=len(data),
            remote_storage_key=remote_key
        )

    def verify_integrity(self, manifest: DVCManifest, current_data: bytes) -> bool:
        """Verifies if local file matches the tracked manifest hash."""
        return compute_content_hash(current_data) == manifest.content_hash


# --------------------------------------------------------------------------- #
# 2. Feature Store (Feast Pattern: Point-in-Time Correctness)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureRecord:
    entity_id: str
    timestamp: float
    features: Dict[str, Any]


class OnlineFeatureStore:
    """Ultra-low latency in-memory KV store serving the latest feature values for real-time inference."""
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def write_features(self, entity_id: str, features: Dict[str, Any]) -> None:
        if entity_id not in self.store:
            self.store[entity_id] = {}
        self.store[entity_id].update(features)

    def get_online_features(self, entity_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return [dict(self.store.get(eid, {})) for eid in entity_ids]


class OfflineFeatureStore:
    """Historical event log supporting point-in-time time-travel joins without target leakage."""
    def __init__(self):
        self.records: List[FeatureRecord] = []

    def log_features(self, entity_id: str, timestamp: float, features: Dict[str, Any]) -> None:
        self.records.append(FeatureRecord(entity_id=entity_id, timestamp=timestamp, features=features))

    def point_in_time_join(
        self,
        observation_events: Sequence[Dict[str, Any]],
        entity_key: str = "user_id",
        timestamp_key: str = "timestamp"
    ) -> List[Dict[str, Any]]:
        """
        Point-in-Time Join (AS-OF join):
        For each observation (e.g. user checkout at time T), retrieves the most recent feature record
        strictly recorded at or before time T (t <= T). Prevents future data leakage during training!
        """
        joined_results = []
        for obs in observation_events:
            eid = obs[entity_key]
            obs_time = obs[timestamp_key]

            # Filter candidates for this entity that occurred <= obs_time
            candidates = [r for r in self.records if r.entity_id == eid and r.timestamp <= obs_time]
            if candidates:
                # Pick the latest valid candidate
                latest = max(candidates, key=lambda r: r.timestamp)
                merged = {**obs, **latest.features, "feature_timestamp": latest.timestamp}
            else:
                merged = {**obs, "feature_timestamp": None}
            joined_results.append(merged)
        return joined_results


# --------------------------------------------------------------------------- #
# 3. Data Lineage & Pipeline Stages (DVC DAG)
# --------------------------------------------------------------------------- #
@dataclass
class PipelineStage:
    name: str
    inputs: Dict[str, str]   # path -> hash
    outputs: Dict[str, str]  # path -> hash


class DataLineageDAG:
    """Tracks upstream data hashes to determine if downstream pipeline stages must re-execute."""
    def __init__(self):
        self.stages: Dict[str, PipelineStage] = {}

    def register_stage(self, stage: PipelineStage) -> None:
        self.stages[stage.name] = stage

    def is_stage_stale(self, stage_name: str, current_input_hashes: Dict[str, str]) -> bool:
        """Returns True if any input hash has diverged from the registered stage inputs."""
        if stage_name not in self.stages:
            return True
        registered_inputs = self.stages[stage_name].inputs
        for path, chash in current_input_hashes.items():
            if registered_inputs.get(path) != chash:
                return True
        return False
