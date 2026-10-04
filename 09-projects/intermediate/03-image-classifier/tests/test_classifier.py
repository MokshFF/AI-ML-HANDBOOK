"""Tests for PyTorch Image Classifier."""
import torch
from dataset import SyntheticImageDataset
from classifier import ConvNet

def test_dataset_output():
    ds = SyntheticImageDataset(n_samples=20)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label.item() < 4

def test_convnet_forward():
    model = ConvNet(in_channels=3, n_classes=4)
    model.eval()
    dummy_input = torch.randn(8, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (8, 4)
    assert not torch.isnan(output).any()
