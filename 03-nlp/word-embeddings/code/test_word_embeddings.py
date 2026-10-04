import pytest
import numpy as np
import torch
from embedding_engine import (
    SkipGramNegativeSampling,
    CBOW,
    GloVeLoss,
    find_most_similar,
    solve_analogy
)


def test_skipgram_negative_sampling():
    vocab_size = 50
    embed_dim = 16
    sgns = SkipGramNegativeSampling(vocab_size=vocab_size, embed_dim=embed_dim)

    center = torch.tensor([1, 4, 7])
    context = torch.tensor([2, 5, 8])
    neg_samples = torch.randint(0, vocab_size, (3, 5))

    loss = sgns(center, context, neg_samples)
    assert loss.item() > 0.0
    loss.backward()

    assert sgns.target_embed.weight.grad is not None
    assert sgns.context_embed.weight.grad is not None
    embeddings = sgns.get_embeddings()
    assert embeddings.shape == (vocab_size, embed_dim)


def test_cbow():
    vocab_size = 30
    embed_dim = 12
    cbow = CBOW(vocab_size=vocab_size, embed_dim=embed_dim)

    # Batch=4, Context Window=4 words (2 left, 2 right)
    context_words = torch.randint(0, vocab_size, (4, 4))
    logits = cbow(context_words)
    assert logits.shape == (4, vocab_size)


def test_glove_loss():
    glove = GloVeLoss(x_max=100.0, alpha=0.75)
    batch_size = 5
    embed_dim = 8

    w_i = torch.randn(batch_size, embed_dim)
    w_j = torch.randn(batch_size, embed_dim)
    b_i = torch.zeros(batch_size, 1)
    b_j = torch.zeros(batch_size, 1)
    cooccur = torch.tensor([[10.0], [50.0], [120.0], [1.0], [5.0]])

    loss = glove(w_i, w_j, b_i, b_j, cooccur)
    assert loss.item() >= 0.0


def test_analogy_and_similarity():
    # Setup controlled synthetic embeddings:
    # king = [1, 0], man = [0.5, 0], woman = [0.5, 1], queen = [1, 1]
    embeddings = np.array([
        [1.0, 0.0],  # 0: king
        [0.5, 0.0],  # 1: man
        [0.5, 1.0],  # 2: woman
        [1.0, 1.0],  # 3: queen
    ])
    idx2word = {0: "king", 1: "man", 2: "woman", 3: "queen"}
    word2idx = {w: i for i, w in idx2word.items()}

    # king - man + woman = [1, 0] - [0.5, 0] + [0.5, 1] = [1, 1] (queen)
    analogy_res = solve_analogy("man", "king", "woman", word2idx, idx2word, embeddings, top_k=1)
    assert len(analogy_res) == 1
    assert analogy_res[0][0] == "queen"
