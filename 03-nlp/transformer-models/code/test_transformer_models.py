"""
Unit tests for Transformer Downstream Models (Classification, NER, and QA).
"""

import pytest
import torch
from transformer_tasks import (
    MiniBertModel,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    extract_best_answer_span,
)


@pytest.fixture
def bert_backbone():
    torch.manual_seed(42)
    return MiniBertModel(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        intermediate_dim=64,
        max_seq_len=64,
    )


def test_mini_bert_forward(bert_backbone):
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len))

    hidden, pooled = bert_backbone(input_ids, attention_mask=mask)
    assert hidden.shape == (batch_size, seq_len, 32)
    assert pooled.shape == (batch_size, 32)


def test_sequence_classification(bert_backbone):
    model = BertForSequenceClassification(bert_backbone, num_labels=3)
    input_ids = torch.randint(0, 100, (4, 10))
    labels = torch.tensor([0, 1, 2, 1])

    out = model(input_ids=input_ids, labels=labels)
    assert "logits" in out and "loss" in out
    assert out["logits"].shape == (4, 3)
    assert out["loss"] is not None
    assert out["loss"].item() > 0


def test_token_classification_ner(bert_backbone):
    model = BertForTokenClassification(bert_backbone, num_labels=5)
    input_ids = torch.randint(0, 100, (3, 8))
    labels = torch.tensor([
        [0, 1, 2, -100, -100, -100, -100, -100],
        [0, 3, 4, 1, -100, -100, -100, -100],
        [1, 1, 2, 2, 0, 0, -100, -100],
    ])

    out = model(input_ids=input_ids, labels=labels)
    assert out["logits"].shape == (3, 8, 5)
    assert out["loss"] is not None
    assert not torch.isnan(out["loss"])


def test_question_answering_and_span_extraction(bert_backbone):
    model = BertForQuestionAnswering(bert_backbone)
    input_ids = torch.randint(0, 100, (2, 12))
    start_pos = torch.tensor([2, 5])
    end_pos = torch.tensor([4, 7])

    out = model(input_ids=input_ids, start_positions=start_pos, end_positions=end_pos)
    assert out["start_logits"].shape == (2, 12)
    assert out["end_logits"].shape == (2, 12)
    assert out["loss"] is not None

    # Test extract_best_answer_span
    start_logits = torch.tensor([-5.0, 4.0, 1.0, -2.0, 0.0])
    end_logits = torch.tensor([-5.0, 0.5, 1.0, 4.5, 0.0])
    best_span = extract_best_answer_span(start_logits, end_logits, max_span_length=4)
    assert best_span == (1, 3)
