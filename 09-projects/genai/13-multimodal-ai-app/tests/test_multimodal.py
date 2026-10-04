"""Tests for Multimodal Engine."""
import torch
from multimodal_engine import VisionEncoder, TextEncoder, MultimodalEngine

def test_encoders():
    v = VisionEncoder(embed_dim=16)
    t = TextEncoder(vocab_size=50, embed_dim=16)
    
    img = torch.randn(2, 3, 32, 32)
    tokens = torch.randint(0, 50, (2, 4))
    
    v_out = v(img)
    t_out = t(tokens)
    assert v_out.shape == (2, 16)
    assert t_out.shape == (2, 16)

def test_multimodal_similarity():
    engine = MultimodalEngine()
    sim = engine.compute_similarity(torch.randn(2, 3, 32, 32), torch.randint(0, 100, (2, 4)))
    assert sim.shape == (2, 2)
