"""Tests for Document Intelligence Extractor."""
import torch
from document_parser import SAMPLE_INVOICE_TOKENS
from entity_extractor import DocumentEntityExtractor, parse_document_to_json

def test_document_model_forward():
    model = DocumentEntityExtractor(vocab_size=200, embed_dim=16, n_classes=5)
    t_ids = torch.randint(0, 200, (2, 8))
    boxes = torch.randint(0, 1000, (2, 8, 4))
    logits = model(t_ids, boxes)
    assert logits.shape == (2, 8, 5)

def test_parse_document_to_json():
    model = DocumentEntityExtractor()
    res = parse_document_to_json(SAMPLE_INVOICE_TOKENS, model)
    assert isinstance(res, dict)
