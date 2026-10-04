import pytest
from data_registry import (
    compute_content_hash,
    DataVersionControl,
    OnlineFeatureStore,
    OfflineFeatureStore,
    PipelineStage,
    DataLineageDAG
)


def test_dvc_content_addressable_tracking():
    dvc = DataVersionControl()
    dataset = b"user_id,amount,timestamp\nu1,100.5,1600000000\nu2,50.0,1600000100"

    manifest = dvc.track_file("data/raw/transactions.csv", dataset)
    assert manifest.path == "data/raw/transactions.csv"
    assert manifest.size_bytes == len(dataset)
    assert manifest.content_hash in dvc.remote_cache

    # Verify integrity passes on exact data
    assert dvc.verify_integrity(manifest, dataset) is True

    # Modified data fails integrity check
    assert dvc.verify_integrity(manifest, dataset + b"\nu3,99.0,1600000200") is False


def test_online_feature_store():
    online = OnlineFeatureStore()
    online.write_features("u100", {"avg_spend_30d": 150.2, "credit_score": 720})
    online.write_features("u200", {"avg_spend_30d": 45.0, "credit_score": 680})

    res = online.get_online_features(["u100", "u200", "u999"])
    assert len(res) == 3
    assert res[0]["credit_score"] == 720
    assert res[1]["avg_spend_30d"] == 45.0
    assert res[2] == {}


def test_point_in_time_join_prevents_leakage():
    offline = OfflineFeatureStore()

    # User 1 feature updates over time
    offline.log_features("u1", timestamp=10.0, features={"credit_score": 650})
    offline.log_features("u1", timestamp=20.0, features={"credit_score": 700})
    offline.log_features("u1", timestamp=30.0, features={"credit_score": 750})

    # Observation 1: User 1 applied for a loan at time 15.0 -> should receive score 650, NOT 700 or 750!
    # Observation 2: User 1 applied for a loan at time 25.0 -> should receive score 700!
    observations = [
        {"user_id": "u1", "timestamp": 15.0, "loan_approved": 0},
        {"user_id": "u1", "timestamp": 25.0, "loan_approved": 1}
    ]

    joined = offline.point_in_time_join(observations, entity_key="user_id", timestamp_key="timestamp")
    assert len(joined) == 2

    # Verify no future leakage
    assert joined[0]["credit_score"] == 650
    assert joined[0]["feature_timestamp"] == 10.0

    assert joined[1]["credit_score"] == 700
    assert joined[1]["feature_timestamp"] == 20.0


def test_data_lineage_staleness():
    dag = DataLineageDAG()
    h_train_raw = "hash_raw_v1"
    h_train_feat = "hash_feat_v1"

    dag.register_stage(PipelineStage(
        name="feature_engineering",
        inputs={"data/raw/data.csv": h_train_raw},
        outputs={"data/features/data.parquet": h_train_feat}
    ))

    # Same input hash -> not stale
    assert dag.is_stage_stale("feature_engineering", {"data/raw/data.csv": "hash_raw_v1"}) is False

    # New raw data hash -> stage is stale and must rerun
    assert dag.is_stage_stale("feature_engineering", {"data/raw/data.csv": "hash_raw_v2_updated"}) is True
