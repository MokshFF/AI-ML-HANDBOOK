"""
Scaling ML Infrastructure & Distributed Systems:
1. Kubernetes Horizontal Pod Autoscaler (HPA / KEDA) concurrency simulation with anti-flapping cooldowns.
2. Distributed Model Sharding & Memory Calculator (Tensor Parallelism, Pipeline Parallelism, ZeRO-3).
3. Multi-tier Cache (L1 Local Memory + L2 Distributed Store).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 1. Autoscaler with Anti-Flapping Cooldowns (HPA / KEDA Pattern)
# --------------------------------------------------------------------------- #
@dataclass
class AutoscalerDecision:
    timestamp: float
    current_replicas: int
    desired_replicas: int
    scaled: bool
    reason: str


class ClusterAutoScaler:
    """Simulates Kubernetes HPA / KEDA scaling based on queue depth / concurrency."""
    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 20,
        target_concurrency_per_replica: float = 10.0,
        scale_up_cooldown_s: float = 30.0,
        scale_down_cooldown_s: float = 120.0
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_concurrency = target_concurrency_per_replica
        self.scale_up_cooldown = scale_up_cooldown_s
        self.scale_down_cooldown = scale_down_cooldown_s

        self.current_replicas = min_replicas
        self.last_scale_time = -1e9

    def evaluate(self, current_time: float, active_load: float) -> AutoscalerDecision:
        raw_desired = math.ceil(active_load / self.target_concurrency)
        desired = max(self.min_replicas, min(self.max_replicas, raw_desired))

        if desired == self.current_replicas:
            return AutoscalerDecision(current_time, self.current_replicas, desired, False, "Load matches capacity")

        elapsed = current_time - self.last_scale_time
        # Scaling up
        if desired > self.current_replicas:
            if elapsed < self.scale_up_cooldown:
                return AutoscalerDecision(current_time, self.current_replicas, desired, False, "Scale-up cooldown active")
            self.current_replicas = desired
            self.last_scale_time = current_time
            return AutoscalerDecision(current_time, self.current_replicas, desired, True, "Scaled up due to high load")

        # Scaling down
        if elapsed < self.scale_down_cooldown:
            return AutoscalerDecision(current_time, self.current_replicas, desired, False, "Scale-down cooldown active (anti-flapping)")
        self.current_replicas = desired
        self.last_scale_time = current_time
        return AutoscalerDecision(current_time, self.current_replicas, desired, True, "Scaled down due to low load")


# --------------------------------------------------------------------------- #
# 2. Distributed Model Sharder (TP, PP, ZeRO)
# --------------------------------------------------------------------------- #
@dataclass
class ShardingBudget:
    param_count_billions: float
    precision_bytes: int
    tp_degree: int
    pp_degree: int
    zero_stage: int
    total_gpus: int
    memory_per_gpu_gb: float
    fits_in_memory: bool
    notes: str


def calculate_distributed_memory(
    param_count_billions: float,
    precision_bytes: int = 2,
    tp_degree: int = 1,
    pp_degree: int = 1,
    zero_stage: int = 0,
    gpu_memory_gb: float = 80.0
) -> ShardingBudget:
    """
    Computes per-GPU memory consumption during inference/training:
    Model Weight GB = Params (in billions) * precision_bytes
    """
    total_model_gb = param_count_billions * precision_bytes
    total_gpus = tp_degree * pp_degree

    if zero_stage == 3:
        # ZeRO-3 shards weights evenly across all data-parallel ranks
        weight_per_gpu = total_model_gb / max(1, total_gpus)
    else:
        # Tensor/Pipeline parallelism shards weights across TP * PP
        weight_per_gpu = total_model_gb / max(1, (tp_degree * pp_degree))

    # Overhead estimate: 20% for activations and KV cache buffers
    estimated_total_per_gpu = weight_per_gpu * 1.25
    fits = estimated_total_per_gpu <= gpu_memory_gb

    notes = f"Weight: {weight_per_gpu:.1f} GB/GPU, Estimated Total: {estimated_total_per_gpu:.1f} GB/GPU"
    return ShardingBudget(
        param_count_billions=param_count_billions,
        precision_bytes=precision_bytes,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        zero_stage=zero_stage,
        total_gpus=total_gpus,
        memory_per_gpu_gb=round(estimated_total_per_gpu, 2),
        fits_in_memory=fits,
        notes=notes
    )


# --------------------------------------------------------------------------- #
# 3. Multi-Tier Cache (L1 Memory + L2 Distributed Store)
# --------------------------------------------------------------------------- #
class MultiTierCache:
    """Two-tier caching architecture: L1 in-process RAM + L2 simulated Redis."""
    def __init__(self, l1_max_items: int = 50):
        self.l1_max = l1_max_items
        self.l1_cache: Dict[str, Any] = {}
        self.l2_cache: Dict[str, Any] = {}
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}

    def get(self, key: str) -> Tuple[Optional[Any], str]:
        # Check L1
        if key in self.l1_cache:
            self.stats["l1_hits"] += 1
            return self.l1_cache[key], "L1_HIT"
        # Check L2
        if key in self.l2_cache:
            val = self.l2_cache[key]
            self.stats["l2_hits"] += 1
            # Populate L1
            if len(self.l1_cache) >= self.l1_max:
                self.l1_cache.pop(next(iter(self.l1_cache)))
            self.l1_cache[key] = val
            return val, "L2_HIT"
        self.stats["misses"] += 1
        return None, "MISS"

    def put(self, key: str, value: Any) -> None:
        self.l2_cache[key] = value
        if len(self.l1_cache) >= self.l1_max:
            self.l1_cache.pop(next(iter(self.l1_cache)))
        self.l1_cache[key] = value
