"""
Python for ML - Core Patterns & Idioms
Implements production-grade patterns: typing, generators, context managers,
custom dataset abstractions, and defensive tensor validations.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, Sequence, Any, Callable, TypeVar
import numpy as np

T = TypeVar("T")


class Timer:
    """
    Context manager for micro-benchmarking execution latency.
    
    Example:
        with Timer("Matrix multiplication"):
            _ = np.dot(A, B)
    """
    def __init__(self, description: str = "Execution"):
        self.description = description
        self.elapsed: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
        print(f"[{self.description}] Elapsed: {self.elapsed * 1000:.3f} ms")


@dataclass
class DatasetConfig:
    """Configuration schema for ML datasets."""
    name: str
    feature_dim: int
    num_samples: int
    normalize: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class MLDataset:
    """
    Standard PyTorch-style Dataset abstraction in pure Python/NumPy.
    Demonstrates encapsulation, operator overloading, and indexing protocols.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, config: DatasetConfig | None = None):
        self._validate_inputs(features, targets)
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.config = config or DatasetConfig(
            name="default_dataset",
            feature_dim=features.shape[1] if features.ndim > 1 else 1,
            num_samples=features.shape[0]
        )
        if self.config.normalize:
            self.features = self._normalize(self.features)

    @staticmethod
    def _validate_inputs(features: np.ndarray, targets: np.ndarray) -> None:
        if not isinstance(features, np.ndarray) or not isinstance(targets, np.ndarray):
            raise TypeError("Features and targets must be numpy.ndarray instances.")
        if len(features) != len(targets):
            raise ValueError(
                f"Sample count mismatch: features have {len(features)}, "
                f"targets have {len(targets)}."
            )
        if np.isnan(features).any() or np.isnan(targets).any():
            raise ValueError("NaN values detected in input tensors.")

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        return (x - mean) / (std + eps)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int | slice | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.targets[idx]

    def __repr__(self) -> str:
        return (
            f"MLDataset(name='{self.config.name}', "
            f"samples={len(self)}, dim={self.config.feature_dim})"
        )


def batch_generator(
    dataset: MLDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Memory-efficient mini-batch generator using Python generator semantics.
    
    Args:
        dataset: MLDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to randomly permute indices each epoch.
        drop_last: Whether to drop the final batch if smaller than batch_size.

    Yields:
        Tuples of (batch_features, batch_targets).
    """
    n = len(dataset)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    
    for start_idx in range(0, n, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > n:
            if drop_last:
                break
            end_idx = n
        batch_indices = indices[start_idx:end_idx]
        yield dataset[batch_indices]


def memoize_transformation(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Decorator caching deterministic array transformations using array bytes hashing.
    """
    cache: dict[bytes, np.ndarray] = {}

    def wrapper(arr: np.ndarray) -> np.ndarray:
        key = arr.tobytes()
        if key not in cache:
            cache[key] = fn(arr)
        return cache[key]

    return wrapper
