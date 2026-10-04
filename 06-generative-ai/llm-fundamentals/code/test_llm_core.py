import math
import pytest
import torch
from llm_core import (
    train_bpe, bpe_encode_word, bpe_decode, sinusoidal_positions, TinyGPT, generate,
    filter_logits, kv_cache_matmul_savings, fit_to_context, parametric_loss, compute_optimal,
)


def test_bpe_roundtrip_and_merges():
    corpus = ["low low low lower lowest", "new newer newest"]
    merges = train_bpe(corpus, 10)
    assert len(merges) > 0
    toks = bpe_encode_word("lowest", merges)
    assert bpe_decode(toks) == "lowest"
    # frequent word compresses to fewer tokens than its characters
    assert len(bpe_encode_word("low", merges)) < len("low") + 1
    # unseen word still decodes (falls back to smaller pieces)
    assert bpe_decode(bpe_encode_word("zzz", merges)) == "zzz"


def test_sinusoidal_properties():
    pe = sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8))
    assert torch.allclose(pe[0, 1::2], torch.ones(8))
    with pytest.raises(ValueError):
        sinusoidal_positions(4, 5)


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    m = TinyGPT(vocab=40, dim=32, heads=4, layers=2)
    prompt = [3, 7, 11, 2]
    a = generate(m, prompt, 12, use_cache=True)
    b = generate(m, prompt, 12, use_cache=False)
    assert a == b
    # logits equal too
    full, _ = m(torch.tensor([a]))
    _, cache = m(torch.tensor([a[:5]]))
    step, _ = m(torch.tensor([[a[5]]]), cache)
    assert torch.allclose(full[0, 5], step[0, 0], atol=1e-5)


def test_kv_cache_work_savings():
    without, with_cache = kv_cache_matmul_savings(10, 5)
    assert with_cache < without


def test_filter_logits_top_k_top_p_temperature():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    p = filter_logits(logits, top_k=2)
    assert (p > 0).sum() == 2 and math.isclose(float(p.sum()), 1.0, rel_tol=1e-5)
    p = filter_logits(logits, top_p=0.5)
    assert (p > 0).sum() >= 1 and p[0] > 0
    sharp = filter_logits(logits, temperature=0.1)
    flat = filter_logits(logits, temperature=5.0)
    assert sharp.max() > flat.max()
    with pytest.raises(ValueError):
        filter_logits(logits, temperature=0.0)


def test_top_p_keeps_smallest_nucleus():
    probs_logits = torch.log(torch.tensor([0.6, 0.3, 0.05, 0.05]))
    p = filter_logits(probs_logits, top_p=0.85)
    assert (p > 0).sum() == 2  # 0.6 + 0.3 = 0.9 >= 0.85


def test_fit_to_context_protects_prefix():
    toks = list(range(20))
    out = fit_to_context(toks, 8, keep_prefix=3)
    assert out == [0, 1, 2, 15, 16, 17, 18, 19] and len(out) == 8
    assert fit_to_context(toks, 50) == toks


def test_scaling_law_compute_optimal():
    c = 1e23
    n, d = compute_optimal(c)
    assert math.isclose(6 * n * d, c, rel_tol=1e-6)
    best = parametric_loss(n, d)
    for f in (0.5, 2.0):
        n2 = n * f
        d2 = c / (6 * n2)
        assert parametric_loss(n2, d2) > best
