import pytest
import torch
from inference_engine import (
    kv_cache_size_bytes,
    memory_bandwidth_throughput_roofline,
    PagedBlockAllocator,
    speculative_decoding_step,
    quantize_int8_symmetric,
    dequantize_int8_symmetric,
    quantize_int4_blockwise,
    dequantize_int4_blockwise
)


def test_kv_cache_sizing_and_roofline():
    # 7B model: 32 layers, 32 heads (or 4 KV heads for GQA), d_head=128
    # MHA: 32 KV heads
    bytes_mha = kv_cache_size_bytes(batch_size=1, seq_len=2048, num_layers=32, num_kv_heads=32, head_dim=128)
    # 2 * 1 * 2048 * 32 * 32 * 128 * 2 = 1,073,741,824 bytes = 1.0 GiB
    assert bytes_mha == 1024 ** 3

    # GQA (Grouped Query Attention) with 4 KV heads (8x reduction in KV cache)
    bytes_gqa = kv_cache_size_bytes(batch_size=1, seq_len=2048, num_layers=32, num_kv_heads=4, head_dim=128)
    assert bytes_gqa == bytes_mha // 8

    # Memory bandwidth roofline: 7B params (14 GB in fp16), 1000 GB/s bandwidth (A100)
    tok_per_sec = memory_bandwidth_throughput_roofline(7 * 10**9, memory_bandwidth_gb_per_sec=1000.0)
    assert 70.0 < tok_per_sec < 80.0  # ~76.7 tok/s theoretical memory-bound peak


def test_paged_block_allocator():
    allocator = PagedBlockAllocator(num_blocks=10, block_size=16)

    # Allocate for 30 tokens -> needs ceil(30/16) = 2 blocks
    blocks = allocator.allocate("req_1", num_tokens=30)
    assert len(blocks) == 2
    assert allocator.memory_utilization(10) == 0.2

    # Append tokens: token 31, 32 -> stays in block
    b1 = allocator.append_token("req_1", current_token_count=31)
    # token 32 hits boundary -> allocates 3rd block
    b2 = allocator.append_token("req_1", current_token_count=32)
    assert len(allocator.block_table["req_1"]) == 3

    # Free request
    allocator.free("req_1")
    assert allocator.memory_utilization(10) == 0.0
    assert len(allocator.free_blocks) == 10


def test_speculative_decoding_simulation():
    vocab_size = 20

    # Test case where draft and target agree
    def draft_model(ctx):
        logits = torch.zeros(vocab_size)
        logits[5] = 10.0  # predicts token 5
        return logits

    def target_model(ctx):
        logits = torch.zeros(vocab_size)
        logits[5] = 10.0  # also predicts token 5
        return logits

    prefix = [1, 2]
    accepted, n_tokens = speculative_decoding_step(draft_model, target_model, prefix, gamma=3)
    # All 3 accepted + 1 bonus token = 4 tokens generated
    assert n_tokens == 4
    assert accepted[:3] == [5, 5, 5]


def test_int8_quantization():
    weights = torch.randn(64, 64)
    q, scale = quantize_int8_symmetric(weights)
    assert q.dtype == torch.int8
    assert q.max() <= 127 and q.min() >= -128

    w_rec = dequantize_int8_symmetric(q, scale)
    mse = torch.mean((weights - w_rec) ** 2)
    assert mse < 0.001


def test_int4_quantization():
    weights = torch.randn(128, 128)
    q, scales = quantize_int4_blockwise(weights, group_size=32)
    assert q.max() <= 7 and q.min() >= -8

    w_rec = dequantize_int4_blockwise(q, scales, weights.numel(), weights.shape)
    mse = torch.mean((weights - w_rec) ** 2)
    assert mse < 0.01
