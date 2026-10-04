"""
Unit tests for Language Modeling Engine: N-Gram LM, Causal Transformer, Perplexity, and Sampling.
"""

import pytest
import torch
from lm_engine import (
    NgramLanguageModel,
    MiniCausalLM,
    compute_perplexity,
    sample_next_token,
    generate_sequence,
)


def test_ngram_model_scoring_and_perplexity():
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
    ]
    lm = NgramLanguageModel(n=2, alpha=1.0)
    lm.fit(corpus)

    score_cat = lm.score("cat", ("the",))
    score_dog = lm.score("dog", ("the",))
    score_unseen = lm.score("elephant", ("the",))

    assert score_cat > 0
    assert score_dog > 0
    assert score_cat > score_unseen

    ppl = lm.compute_perplexity(["the", "cat", "sat"])
    assert ppl > 1.0


def test_mini_causal_lm_forward_and_loss():
    torch.manual_seed(42)
    model = MiniCausalLM(vocab_size=50, hidden_dim=32, num_layers=2, num_heads=4, intermediate_dim=64)

    input_ids = torch.randint(0, 50, (2, 8))
    targets = torch.randint(0, 50, (2, 8))

    out = model(input_ids, targets=targets)
    assert out["logits"].shape == (2, 8, 50)
    assert out["loss"] is not None
    assert out["loss"].item() > 0

    ppl = compute_perplexity(out["loss"].item())
    assert ppl > 1.0


def test_sampling_strategies():
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])

    # Greedy test (temp=0)
    greedy_token = sample_next_token(logits, temperature=0.0)
    assert greedy_token == 1

    # Top-k test
    sampled_token = sample_next_token(logits, temperature=1.0, top_k=2)
    assert sampled_token in [1, 2]

    # Top-p test
    sampled_p = sample_next_token(logits, temperature=1.0, top_p=0.8)
    assert sampled_p in [1, 2]


def test_autoregressive_generation():
    torch.manual_seed(42)
    model = MiniCausalLM(vocab_size=30, hidden_dim=32, num_layers=1, num_heads=2, intermediate_dim=64)
    prompt = [1, 5, 12]
    new_tokens = 5
    generated = generate_sequence(model, prompt_ids=prompt, max_new_tokens=new_tokens, temperature=0.7, top_k=5)

    assert len(generated) == len(prompt) + new_tokens
    assert generated[:len(prompt)] == prompt
