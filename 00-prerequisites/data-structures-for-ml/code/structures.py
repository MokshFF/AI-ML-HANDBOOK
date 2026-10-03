"""
Data Structures for ML - Algorithmic Implementations from Scratch
Implements contiguous arrays, hash maps with rehashing, streaming ring buffers,
KD-Trees for spatial search, binary heaps for top-k retrieval, and DAGs for computational graphs.
"""

from __future__ import annotations
import math
from typing import Any, Iterator, Generic, TypeVar
import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------
# 1. Contiguous Dynamic Array
# ---------------------------------------------------------

class DynamicArray:
    """
    Simulates dynamic array memory growth (amortized O(1) appends)
    demonstrating cache line locality vs linked structures.
    """
    def __init__(self, initial_capacity: int = 4):
        self._capacity = initial_capacity
        self._size = 0
        self._buffer: list[Any] = [None] * self._capacity

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Any:
        if not (0 <= index < self._size):
            raise IndexError("Index out of bounds.")
        return self._buffer[index]

    def append(self, value: Any) -> None:
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._buffer[self._size] = value
        self._size += 1

    def _resize(self, new_capacity: int) -> None:
        new_buffer = [None] * new_capacity
        for i in range(self._size):
            new_buffer[i] = self._buffer[i]
        self._buffer = new_buffer
        self._capacity = new_capacity


# ---------------------------------------------------------
# 2. Hash Map with Separate Chaining & Load Factor Rehashing
# ---------------------------------------------------------

class ChainedHashMap:
    """
    Hash map with separate chaining collision resolution and dynamic rehashing.
    Essential in ML for feature hashing, categorical embedding lookup tables, and vocabulary mappings.
    """
    def __init__(self, initial_buckets: int = 8, max_load_factor: float = 0.75):
        self.num_buckets = initial_buckets
        self.max_load_factor = max_load_factor
        self.size = 0
        self.buckets: list[list[tuple[Any, Any]]] = [[] for _ in range(self.num_buckets)]

    def _hash(self, key: Any) -> int:
        return hash(key) % self.num_buckets

    def put(self, key: Any, value: Any) -> None:
        b_idx = self._hash(key)
        bucket = self.buckets[b_idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1

        if (self.size / self.num_buckets) > self.max_load_factor:
            self._rehash()

    def get(self, key: Any, default: Any = None) -> Any:
        b_idx = self._hash(key)
        for k, v in self.buckets[b_idx]:
            if k == key:
                return v
        return default

    def _rehash(self) -> None:
        old_buckets = self.buckets
        self.num_buckets *= 2
        self.buckets = [[] for _ in range(self.num_buckets)]
        self.size = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)


# ---------------------------------------------------------
# 3. Streaming Circular Queue (Ring Buffer)
# ---------------------------------------------------------

class RingBuffer(Generic[T]):
    """
    Fixed-size circular queue with O(1) enqueues and dequeues.
    Used for rolling experience replay buffers in reinforcement learning and sliding window statistics.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[T | None] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.count = 0

    def append(self, item: T) -> None:
        """Appends item, overwriting oldest element if full."""
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        else:
            self.head = (self.head + 1) % self.capacity

    def get_all(self) -> list[T]:
        """Returns items in temporal FIFO order."""
        items = []
        for i in range(self.count):
            idx = (self.head + i) % self.capacity
            items.append(self.buffer[idx])  # type: ignore
        return items

    def __len__(self) -> int:
        return self.count


# ---------------------------------------------------------
# 4. KD-Tree for Nearest Neighbor Spatial Queries
# ---------------------------------------------------------

class KDNode:
    def __init__(self, point: np.ndarray, index: int, axis: int, left=None, right=None):
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """
    k-d Tree spatial index for logarithmic nearest-neighbor search: O(log N) average query time.
    """
    def __init__(self, points: np.ndarray):
        self.dim = points.shape[1]
        indices = np.arange(len(points))
        self.root = self._build(points, indices, depth=0)

    def _build(self, points: np.ndarray, indices: np.ndarray, depth: int) -> KDNode | None:
        if len(indices) == 0:
            return None
        axis = depth % self.dim
        sorted_order = np.argsort(points[indices, axis])
        sorted_indices = indices[sorted_order]
        median_idx = len(sorted_indices) // 2

        node_pt_idx = sorted_indices[median_idx]
        return KDNode(
            point=points[node_pt_idx],
            index=node_pt_idx,
            axis=axis,
            left=self._build(points, sorted_indices[:median_idx], depth + 1),
            right=self._build(points, sorted_indices[median_idx + 1:], depth + 1)
        )

    def query_nearest(self, query_pt: np.ndarray) -> tuple[int, float]:
        """Finds nearest point in tree to query_pt. Returns: (point_index, distance)."""
        best = {"index": -1, "dist": float("inf")}

        def search(node: KDNode | None):
            if node is None:
                return
            dist = float(np.linalg.norm(query_pt - node.point))
            if dist < best["dist"]:
                best["dist"] = dist
                best["index"] = node.index

            axis = node.axis
            diff = query_pt[axis] - node.point[axis]
            first = node.left if diff < 0 else node.right
            second = node.right if diff < 0 else node.left

            search(first)
            # Prune branch if plane difference is greater than current best distance
            if abs(diff) < best["dist"]:
                search(second)

        search(self.root)
        return best["index"], best["dist"]


# ---------------------------------------------------------
# 5. Min-Heap for Top-K Retrieval
# ---------------------------------------------------------

class MinHeap:
    """
    Binary Min-Heap array representation.
    Used for extracting top-k items from massive data streams in O(N log K) time.
    """
    def __init__(self):
        self.heap: list[tuple[float, Any]] = []

    def push(self, score: float, item: Any) -> None:
        self.heap.append((score, item))
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> tuple[float, Any]:
        if not self.heap:
            raise IndexError("Pop from empty heap.")
        top = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return top

    def peek(self) -> tuple[float, Any]:
        if not self.heap:
            raise IndexError("Peek from empty heap.")
        return self.heap[0]

    def _sift_up(self, idx: int) -> None:
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx][0] < self.heap[parent][0]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx: int) -> None:
        n = len(self.heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                idx = smallest
            else:
                break

    def __len__(self) -> int:
        return len(self.heap)


def top_k_scores(scores: list[float], k: int) -> list[float]:
    """Retrieves top-k scores in O(N log k) time using a Min-Heap of size k."""
    heap = MinHeap()
    for s in scores:
        if len(heap) < k:
            heap.push(s, s)
        elif s > heap.peek()[0]:
            heap.pop()
            heap.push(s, s)
    return sorted([val for _, val in heap.heap], reverse=True)


# ---------------------------------------------------------
# 6. Directed Acyclic Graph (DAG) for Computational Graphs
# ---------------------------------------------------------

class ComputationalDAG:
    """
    DAG representation for operator dependency tracking in neural network forward/backward passes.
    Supports Kahn's algorithm for topological sorting and cycle detection.
    """
    def __init__(self):
        self.adj: dict[str, list[str]] = {}
        self.in_degree: dict[str, int] = {}

    def add_node(self, node: str) -> None:
        if node not in self.adj:
            self.adj[node] = []
            self.in_degree[node] = 0

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.add_node(from_node)
        self.add_node(to_node)
        self.adj[from_node].append(to_node)
        self.in_degree[to_node] += 1

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm: O(V + E) complexity."""
        in_deg = self.in_degree.copy()
        queue = [node for node, deg in in_deg.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in self.adj[curr]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.adj):
            raise ValueError("Cycle detected in graph; topological sort impossible.")
        return order
