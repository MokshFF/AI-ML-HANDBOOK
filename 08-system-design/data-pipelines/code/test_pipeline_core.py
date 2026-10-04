import pytest
from pipeline_core import (
    IdempotentDeduplicator,
    SlidingWindowAggregator,
    DeadLetterQueue,
    partition_backfill_ranges
)


def test_idempotent_deduplicator():
    dedup = IdempotentDeduplicator(max_history=5)

    # First arrival -> not duplicate
    assert dedup.is_duplicate_and_record("evt_1") is False

    # Second arrival of same event -> duplicate!
    assert dedup.is_duplicate_and_record("evt_1") is True

    # Other events
    assert dedup.is_duplicate_and_record("evt_2") is False
    assert dedup.is_duplicate_and_record("evt_3") is False


def test_sliding_window_aggregator():
    agg = SlidingWindowAggregator(window_size_seconds=60.0)

    # Events at t=10, 20, 50, 80
    agg.add_event(timestamp=10.0, value=100.0)
    agg.add_event(timestamp=20.0, value=200.0)
    agg.add_event(timestamp=50.0, value=300.0)

    # Window evaluated at t=60: includes t=10, 20, 50 (all in [0, 60])
    w60 = agg.compute_window(current_time=60.0)
    assert w60.count == 3
    assert w60.sum_value == 600.0
    assert w60.mean_value == 200.0

    # Add event at t=80
    agg.add_event(timestamp=80.0, value=400.0)

    # Window evaluated at t=90: window is [30, 90]. Drops t=10 and 20!
    # Includes t=50 (300) and t=80 (400)
    w90 = agg.compute_window(current_time=90.0)
    assert w90.count == 2
    assert w90.sum_value == 700.0
    assert w90.mean_value == 350.0


def test_dead_letter_queue():
    dlq = DeadLetterQueue()
    assert dlq.size() == 0

    try:
        raise ValueError("Malformed JSON payload: missing field 'price'")
    except Exception as e:
        dlq.push_failure(raw_payload="bad_data_123", error=e)

    assert dlq.size() == 1
    rec = dlq.failed_records[0]
    assert "ValueError" in rec.error_message
    assert rec.raw_payload == "bad_data_123"


def test_backfill_partition_chunker():
    # 3.5 hours split into 1-hour chunks -> 4 chunks
    chunks = partition_backfill_ranges(start_time=0.0, end_time=12600.0, chunk_size_seconds=3600.0)
    assert len(chunks) == 4
    assert chunks[0] == (0.0, 3600.0)
    assert chunks[1] == (3600.0, 7200.0)
    assert chunks[2] == (7200.0, 10800.0)
    assert chunks[3] == (10800.0, 12600.0)
