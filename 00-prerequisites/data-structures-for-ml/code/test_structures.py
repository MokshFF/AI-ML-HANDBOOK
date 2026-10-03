"""
Tests for Data Structures for ML.
"""

import numpy as np
import pytest
from structures import (
    DynamicArray,
    ChainedHashMap,
    RingBuffer,
    KDTree,
    MinHeap,
    top_k_scores,
    ComputationalDAG,
)


def test_dynamic_array():
    arr = DynamicArray(initial_capacity=2)
    assert len(arr) == 0

    for i in range(10):
        arr.append(i * 10)

    assert len(arr) == 10
    assert arr[0] == 0
    assert arr[9] == 90

    with pytest.raises(IndexError):
        _ = arr[10]


def test_chained_hash_map():
    hm = ChainedHashMap(initial_buckets=4, max_load_factor=0.75)
    hm.put("learning_rate", 0.001)
    hm.put("batch_size", 64)
    hm.put("optimizer", "adam")

    assert hm.get("learning_rate") == 0.001
    assert hm.get("batch_size") == 64
    assert hm.get("optimizer") == "adam"
    assert hm.get("nonexistent", default=404) == 404

    # Test update
    hm.put("learning_rate", 0.01)
    assert hm.get("learning_rate") == 0.01

    # Insert more elements to trigger rehash
    for i in range(20):
        hm.put(f"param_{i}", i)

    assert hm.get("param_15") == 15
    assert hm.size == 23


def test_ring_buffer():
    rb = RingBuffer[int](capacity=3)
    assert len(rb) == 0

    rb.append(10)
    rb.append(20)
    rb.append(30)
    assert rb.get_all() == [10, 20, 30]

    # Overwrite oldest
    rb.append(40)
    assert len(rb) == 3
    assert rb.get_all() == [20, 30, 40]


def test_kd_tree():
    points = np.array([
        [2.0, 3.0],
        [5.0, 4.0],
        [9.0, 6.0],
        [4.0, 7.0],
        [8.0, 1.0],
        [7.0, 2.0]
    ])
    tree = KDTree(points)

    query = np.array([9.2, 5.8])
    idx, dist = tree.query_nearest(query)

    # Closest should be [9.0, 6.0] at index 2
    assert idx == 2
    assert np.isclose(dist, np.linalg.norm(query - points[2]))


def test_min_heap_and_top_k():
    heap = MinHeap()
    vals = [7, 2, 9, 1, 5]
    for v in vals:
        heap.push(v, f"item_{v}")

    popped = [heap.pop()[0] for _ in range(len(vals))]
    assert popped == [1, 2, 5, 7, 9]

    # Top-k
    scores = [12.5, 99.1, 45.2, 88.0, 2.3, 104.7, 73.1]
    top_3 = top_k_scores(scores, k=3)
    assert top_3 == [104.7, 99.1, 88.0]


def test_computational_dag():
    dag = ComputationalDAG()
    # Linear model: X -> MatMul -> AddBias -> Activation -> Loss
    dag.add_edge("Input_X", "MatMul")
    dag.add_edge("Weights", "MatMul")
    dag.add_edge("MatMul", "AddBias")
    dag.add_edge("Bias", "AddBias")
    dag.add_edge("AddBias", "Activation")
    dag.add_edge("Activation", "Loss")

    order = dag.topological_sort()
    assert order.index("Input_X") < order.index("MatMul")
    assert order.index("MatMul") < order.index("AddBias")
    assert order.index("Activation") < order.index("Loss")

    # Cycle test
    dag_cyclic = ComputationalDAG()
    dag_cyclic.add_edge("A", "B")
    dag_cyclic.add_edge("B", "C")
    dag_cyclic.add_edge("C", "A")

    with pytest.raises(ValueError, match="Cycle detected"):
        dag_cyclic.topological_sort()
