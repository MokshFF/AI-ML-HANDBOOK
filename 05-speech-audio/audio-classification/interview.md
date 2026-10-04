# Audio Classification & Speaker Verification - Technical Interview Preparation

A curated question bank covering 2D vs. 1D audio architectures, temporal pooling strategies, metric learning objectives (Additive Angular Margin / ArcFace), and biometric evaluation (EER).

---

## 1. Architectural & Theoretical Foundations

### Q1: Why are 2D convolutions on log-Mel spectrograms preferred over 1D convolutions on raw waveforms (e.g. WaveNet / SincNet) for audio classification?
- **Domain Transformation Efficiency**:
  Raw audio at 16 kHz has 16,000 values per second. A 1D ConvNet must learn massive receptive fields ($> 10,000$ samples) and high filter redundancy simply to reconstruct basic Fourier-like sinusoidal basis functions.
- **2D Spectrogram Advantages**:
  1. *Dimensionality Reduction*: STFT with hop length 160 reduces 16,000 samples to 100 spectral frames per second ($160\times$ temporal reduction).
  2. *Visual Analogy*: Formant frequencies, harmonic spacing, and pitch tracks form 2D geometric patterns in the time-frequency plane. Standard 2D vision backbones (ResNet, EfficientNet) can be applied with minor modifications and transfer learning.

---

### Q2: How does Temporal Statistics Pooling work in x-vector architectures (Snyder et al.), and why is it essential for variable-duration audio?
- **The Variable Duration Problem**:
  Speaker enrollment utterances vary in duration (e.g., 2 seconds vs. 15 seconds). A standard dense network cannot accept variable-length time sequences.
- **Statistics Pooling Mechanism**:
  Across all temporal frame representations $\{\mathbf{h}_1, \dots, \mathbf{h}_T\}$ produced by the encoder:
  1. Compute temporal mean: $\boldsymbol{\mu} = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t$.
  2. Compute temporal standard deviation: $\boldsymbol{\sigma} = \sqrt{\frac{1}{T} \sum_{t=1}^T (\mathbf{h}_t - \boldsymbol{\mu})^2}$.
  3. Concatenate: $\mathbf{h}_{\text{pooled}} = [\boldsymbol{\mu}; \boldsymbol{\sigma}] \in \mathbb{R}^{2d}$.
  This maps any arbitrary-length utterance into a fixed-length summary vector that captures both average vocal tract characteristics ($\boldsymbol{\mu}$) and speaking dynamics/variability ($\boldsymbol{\sigma}$).

---

### Q3: What is the Equal Error Rate (EER), and how is it computed from the ROC / DET curve?
- **Definitions**:
  - $\text{FAR}(\theta) = \frac{\text{False Impostor Accepts}}{\text{Total Impostor Trials}}$
  - $\text{FRR}(\theta) = \frac{\text{False Genuine Rejects}}{\text{Total Genuine Trials}}$
- **Trade-off Dynamics**:
  As verification decision threshold $\theta$ increases:
  - $\text{FAR}(\theta)$ decreases (system becomes stricter; fewer impostors enter).
  - $\text{FRR}(\theta)$ increases (more genuine users are rejected).
- **The EER**:
  The specific threshold $\theta^*$ where $\text{FAR}(\theta^*) = \text{FRR}(\theta^*)$ is the Equal Error Rate (EER). It provides a single operating-point-independent benchmark metric to compare speaker verification models.

---

## 2. Whiteboard Coding Drills

### Q4: Implement Triplet Margin Loss in PyTorch with Euclidean distance.
```python
import torch
import torch.nn.functional as F

def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2):
    """
    anchor, positive, negative: (B, D) tensors
    """
    d_pos = torch.norm(anchor - positive, p=2, dim=-1)
    d_neg = torch.norm(anchor - negative, p=2, dim=-1)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()
```
