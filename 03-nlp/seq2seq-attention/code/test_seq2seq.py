import pytest
import torch
from seq2seq_engine import (
    BahdanauAttention,
    LuongAttention,
    Seq2SeqEncoder,
    Seq2SeqAttentionDecoder,
    Seq2SeqModel,
    compute_bleu_score
)


def test_bahdanau_and_luong_attention():
    B = 2
    src_len = 5
    enc_dim = 16
    dec_dim = 16

    dec_hidden = torch.randn(B, dec_dim)
    enc_outputs = torch.randn(B, src_len, enc_dim)

    # 1. Bahdanau Additive Attention
    bahdanau = BahdanauAttention(dec_hidden_dim=dec_dim, enc_hidden_dim=enc_dim)
    context_b, weights_b = bahdanau(dec_hidden, enc_outputs)
    assert context_b.shape == (B, enc_dim)
    assert weights_b.shape == (B, src_len)
    assert torch.allclose(weights_b.sum(dim=-1), torch.ones(B), atol=1e-5)

    # 2. Luong Multiplicative Attention
    luong = LuongAttention(dec_hidden_dim=dec_dim, enc_hidden_dim=enc_dim)
    context_l, weights_l = luong(dec_hidden, enc_outputs)
    assert context_l.shape == (B, enc_dim)
    assert weights_l.shape == (B, src_len)
    assert torch.allclose(weights_l.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_seq2seq_model_forward():
    src_vocab = 30
    tgt_vocab = 35
    embed_dim = 16
    hidden_dim = 32

    encoder = Seq2SeqEncoder(src_vocab, embed_dim, hidden_dim)
    decoder = Seq2SeqAttentionDecoder(tgt_vocab, embed_dim, hidden_dim, hidden_dim)
    model = Seq2SeqModel(encoder, decoder)

    src = torch.randint(0, src_vocab, (4, 8))  # Batch=4, SrcLen=8
    tgt = torch.randint(0, tgt_vocab, (4, 6))  # Batch=4, TgtLen=6

    # Forward with teacher forcing
    outputs = model(src, tgt, teacher_forcing_ratio=0.5)
    assert outputs.shape == (4, 6, tgt_vocab)

    # Backpropagation check
    loss = outputs.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_bleu_score_computation():
    ref = ["the", "quick", "brown", "fox", "jumps"]
    cand_exact = ["the", "quick", "brown", "fox", "jumps"]
    cand_partial = ["the", "fast", "brown", "fox"]
    cand_disjoint = ["completely", "different", "sentence"]

    # Exact match: BLEU = 1.0
    bleu_exact = compute_bleu_score(ref, cand_exact, max_n=2)
    assert pytest.approx(bleu_exact, 1e-4) == 1.0

    # Partial match: 0 < BLEU < 1
    bleu_part = compute_bleu_score(ref, cand_partial, max_n=2)
    assert 0.0 < bleu_part < 1.0

    # Disjoint match: BLEU near 0
    bleu_disjoint = compute_bleu_score(ref, cand_disjoint, max_n=2)
    assert bleu_disjoint < 0.01
