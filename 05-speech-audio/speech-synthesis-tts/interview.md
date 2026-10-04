# Speech Synthesis (TTS) - Technical Interview Preparation

A curated question bank covering autoregressive vs. non-autoregressive acoustic models, attention alignment failures, duration prediction, and neural vocoder architectures (HiFi-GAN).

---

## 1. Architectural & Theoretical Foundations

### Q1: What causes attention alignment failures (word skipping and word repetition) in autoregressive TTS models like Tacotron 2, and how does FastSpeech eliminate them?
- **Tacotron 2 Failure Modes**:
  Tacotron 2 uses content-based or location-sensitive attention between the phoneme encoder and acoustic decoder.
  - *Word Skipping*: If the decoder's attention weights jump forward prematurely across phoneme tokens, entire syllables or words are omitted from the synthesized speech.
  - *Word Repeating*: If attention weights get stuck in a local minimum on a particular token, the decoder enters an infinite repetition loop, repeating a phrase continuously.
- **How FastSpeech Eliminates Them**:
  FastSpeech discards soft attention alignments entirely. It uses a **Length Regulator** driven by an explicit **Duration Predictor**. Each phoneme is deterministically expanded into an exact integer number of frames ($d_i$). Because the sequence of frames is hard-expanded strictly in chronological order, skipping or looping words is mathematically impossible.

---

### Q2: Why is a vocoder necessary? Why can't we simply take the Inverse Short-Time Fourier Transform (iSTFT) of the predicted Mel spectrogram?
- **The Phase Discard Problem**:
  The STFT computes complex numbers $X(f, t) = |X| e^{j \phi}$. The spectrogram retains only the magnitude $|X|$ and discards the phase $\phi$.
  - While human hearing is relatively insensitive to static phase shifts of individual sinusoids, the relative phase differences across adjacent frequencies determine temporal wave interference patterns (e.g. glottal closure instants in speech).
  - Furthermore, Mel filterbanks compress hundreds of FFT bins into 80 bins, making direct inversion mathematically underdetermined.
- **Neural Vocoders (HiFi-GAN / WaveNet)**:
  Neural vocoders are trained specifically to generate both plausible high-resolution time-domain samples and phase coherence conditioned on the low-resolution Mel spectrogram.

---

### Q3: How does Multi-Period Discriminator (MPD) in HiFi-GAN capture pitch harmonics?
- **The Challenge in Audio GANs**:
  Speech audio contains periodic signals with various fundamental frequencies ($f_0 \in [80, 400]\text{ Hz}$). A standard 1D convolutional discriminator has difficulty evaluating periodic continuity across distant samples.
- **The MPD Architecture**:
  HiFi-GAN reshapes the 1D audio waveform into 2D matrices of dimensions $(T/p, p)$ using different prime periods $p \in [2, 3, 5, 7, 11]$.
  - By applying 2D convolutions over these reshaped matrices, the discriminator evaluates adjacent periodic cycles directly as neighboring rows, efficiently enforcing phase and harmonic consistency across periods.

---

## 2. Whiteboard Coding Drills

### Q4: Implement a Length Regulator in PyTorch that expands phoneme embeddings based on an integer duration tensor.
```python
import torch

def length_regulator(x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, D)
    durations: (B, T)
    """
    batch_size, seq_len, d_dim = x.shape
    expanded_list = []
    
    for b in range(batch_size):
        reps = []
        for t in range(seq_len):
            d = max(1, int(durations[b, t].item()))
            reps.append(x[b, t].unsqueeze(0).expand(d, -1))
        expanded_list.append(torch.cat(reps, dim=0))
        
    max_len = max(item.size(0) for item in expanded_list)
    out = torch.zeros(batch_size, max_len, d_dim, device=x.device)
    for b, item in enumerate(expanded_list):
        out[b, :item.size(0)] = item
        
    return out
```
