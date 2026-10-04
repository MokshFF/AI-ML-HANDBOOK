"""Tests for End-to-End LLM Gateway."""
from app_engine import TokenBucketRateLimiter, LLMGateway

def test_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate_per_sec=1.0)
    assert limiter.allow_request() is True
    assert limiter.allow_request() is True
    assert limiter.allow_request() is False

def test_gateway_caching():
    gw = LLMGateway()
    r1 = gw.process_prompt("What is my account balance?")
    assert r1["source"] == "GENERATION"
    r2 = gw.process_prompt("What is my account balance?")
    assert r2["source"] == "CACHE"
    assert gw.telemetry["cache_hits"] == 1
