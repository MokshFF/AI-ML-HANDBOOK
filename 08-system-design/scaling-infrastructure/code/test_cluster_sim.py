import pytest
from cluster_sim import (
    ClusterAutoScaler,
    calculate_distributed_memory,
    MultiTierCache
)


def test_autoscaler_scaling_and_cooldown():
    scaler = ClusterAutoScaler(
        min_replicas=2,
        max_replicas=10,
        target_concurrency_per_replica=10.0,
        scale_up_cooldown_s=20.0,
        scale_down_cooldown_s=60.0
    )

    # Initial state: 2 replicas
    assert scaler.current_replicas == 2

    # Spike: 50 concurrent requests at t=0 -> needs ceil(50/10) = 5 replicas
    d1 = scaler.evaluate(current_time=0.0, active_load=50.0)
    assert d1.scaled is True
    assert d1.current_replicas == 5

    # Another spike immediately at t=5s -> scale up cooldown prevents flapping
    d2 = scaler.evaluate(current_time=5.0, active_load=80.0)
    assert d2.scaled is False
    assert "Scale-up cooldown active" in d2.reason
    assert d2.current_replicas == 5

    # At t=25s, cooldown has passed -> scales to 8 replicas
    d3 = scaler.evaluate(current_time=25.0, active_load=80.0)
    assert d3.scaled is True
    assert d3.current_replicas == 8

    # Traffic drops to 10 at t=30s -> scale down cooldown (60s) prevents immediate premature downscaling
    d4 = scaler.evaluate(current_time=30.0, active_load=10.0)
    assert d4.scaled is False
    assert "Scale-down cooldown active" in d4.reason


def test_distributed_memory_budgeting():
    # 70B model in fp16 (140 GB weights) on single 80 GB GPU -> DOES NOT FIT
    b_single = calculate_distributed_memory(param_count_billions=70.0, precision_bytes=2, tp_degree=1, gpu_memory_gb=80.0)
    assert b_single.fits_in_memory is False

    # 70B model sharded with Tensor Parallelism TP=4 across 4x 80GB GPUs -> FITS!
    # 140 GB / 4 = 35 GB weights * 1.25 = 43.75 GB <= 80 GB
    b_tp4 = calculate_distributed_memory(param_count_billions=70.0, precision_bytes=2, tp_degree=4, gpu_memory_gb=80.0)
    assert b_tp4.fits_in_memory is True
    assert b_tp4.total_gpus == 4
    assert b_tp4.memory_per_gpu_gb < 50.0


def test_multi_tier_cache():
    cache = MultiTierCache(l1_max_items=2)

    # First get -> MISS
    val, status = cache.get("key1")
    assert status == "MISS"

    # Put in cache
    cache.put("key1", "val1")

    # Second get -> L1_HIT
    v1, s1 = cache.get("key1")
    assert s1 == "L1_HIT"
    assert v1 == "val1"

    # Fill L1 cache past capacity to evict key1 from L1 into L2 only
    cache.put("key2", "val2")
    cache.put("key3", "val3")

    # key1 is now evicted from L1, but present in L2 -> L2_HIT
    v_l2, s_l2 = cache.get("key1")
    assert s_l2 == "L2_HIT"
    assert v_l2 == "val1"
