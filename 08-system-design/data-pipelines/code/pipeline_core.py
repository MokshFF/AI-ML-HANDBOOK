"""
Data Pipelines for Machine Learning:
1. Idempotent Event Deduplicator (at-least-once to exactly-once processing).
2. Time-Windowed Stream Aggregator (Tumbling & Sliding windows).
3. Dead-Letter Queue (DLQ) for poisonous record isolation.
4. Historical Data Backfill partition chunker.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# 1. Idempotent Deduplicator
# --------------------------------------------------------------------------- #
class IdempotentDeduplicator:
    """Tracks seen event IDs within a sliding retention window to guarantee exactly-once semantics."""
    def __init__(self, max_history: int = 1000):
        self.seen_ids: Set[str] = set()
        self.history_queue: List[str] = []
        self.max_history = max_history

    def is_duplicate_and_record(self, event_id: str) -> bool:
        if event_id in self.seen_ids:
            return True
        self.seen_ids.add(event_id)
        self.history_queue.append(event_id)
        if len(self.history_queue) > self.max_history:
            oldest = self.history_queue.pop(0)
            self.seen_ids.remove(oldest)
        return False


# --------------------------------------------------------------------------- #
# 2. Time-Windowed Stream Aggregator
# --------------------------------------------------------------------------- #
@dataclass
class WindowAggregate:
    window_start: float
    window_end: float
    count: int
    sum_value: float
    mean_value: float


class SlidingWindowAggregator:
    """Computes rolling aggregations over a sliding temporal window (e.g. 60-second window)."""
    def __init__(self, window_size_seconds: float = 60.0):
        self.window_size = window_size_seconds
        self.events: List[Tuple[float, float]] = []  # (timestamp, numeric_value)

    def add_event(self, timestamp: float, value: float) -> None:
        self.events.append((timestamp, value))

    def compute_window(self, current_time: float) -> WindowAggregate:
        cutoff = current_time - self.window_size
        # Keep only events within window
        valid_events = [val for ts, val in self.events if cutoff <= ts <= current_time]
        count = len(valid_events)
        total = sum(valid_events) if count > 0 else 0.0
        mean = total / count if count > 0 else 0.0
        return WindowAggregate(
            window_start=cutoff,
            window_end=current_time,
            count=count,
            sum_value=round(total, 2),
            mean_value=round(mean, 2)
        )


# --------------------------------------------------------------------------- #
# 3. Dead-Letter Queue (DLQ)
# --------------------------------------------------------------------------- #
@dataclass
class DeadLetterRecord:
    raw_payload: Any
    error_message: str
    timestamp: float = field(default_factory=time.time)


class DeadLetterQueue:
    """Isolates poisoned records failing parsing without halting the pipeline."""
    def __init__(self):
        self.failed_records: List[DeadLetterRecord] = []

    def push_failure(self, raw_payload: Any, error: Exception) -> None:
        self.failed_records.append(DeadLetterRecord(
            raw_payload=raw_payload,
            error_message=f"{type(error).__name__}: {str(error)}"
        ))

    def size(self) -> int:
        return len(self.failed_records)


# --------------------------------------------------------------------------- #
# 4. Historical Backfill Partition Chunker
# --------------------------------------------------------------------------- #
def partition_backfill_ranges(
    start_time: float,
    end_time: float,
    chunk_size_seconds: float = 3600.0
) -> List[Tuple[float, float]]:
    """Splits historical backfill interval into non-overlapping parallel worker chunks."""
    chunks = []
    curr = start_time
    while curr < end_time:
        next_chunk = min(curr + chunk_size_seconds, end_time)
        chunks.append((curr, next_chunk))
        curr = next_chunk
    return chunks
