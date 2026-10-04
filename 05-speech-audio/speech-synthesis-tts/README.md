# Speech Synthesis (TTS): FastSpeech, Vocoders & Mel-Cepstral Distortion

A comprehensive mathematical and architectural guide to Text-to-Speech (TTS) synthesis, comparing autoregressive (Tacotron 2) and non-autoregressive (FastSpeech) acoustic models, Length Regulation duration mechanisms, neural vocoders (HiFi-GAN, WaveNet), and Mel-Cepstral Distortion (MCD) evaluation.

---

## 1. The Two-Stage Text-to-Speech Pipeline

Directly mapping raw text characters to high-frequency audio waveforms (e.g. 24,000 samples per second) has an intractable sequence length disparity ($1\text{ s of speech} \approx 15\text{ text characters vs. } 24,000\text{ audio samples}$). Modern systems decouple the problem into two distinct stages:

```
Text: "Hello world"
         │
    [Text Normalization & Grapheme-to-Phoneme (G2P)]
         │
 Phonemes: /h eh l ow w er l d/
         │
┌────────▼─────────────────────────────────────────────────┐
│ STAGE 1: Acoustic Model (FastSpeech 2 / Tacotron 2)       │
│ - Phoneme Encoder                                        │
│ - Duration / Pitch / Energy Predictor                    │
│ - Length Regulator (Expands Phonemes -> Acoustic Frames) │
│ - Mel Spectrogram Decoder                                │
└────────┬─────────────────────────────────────────────────┘
         │
 Intermediate Representation: Log-Mel Spectrogram (80 bins x T_mel)
         │
┌────────▼─────────────────────────────────────────────────┐
│ STAGE 2: Neural Vocoder (HiFi-GAN / WaveNet / WaveGlow)   │
│ - Inverts time-frequency magnitude to time-domain audio  │
│ - Generates phase information                            │
└────────┬─────────────────────────────────────────────────┘
         │
 Output Waveform x(t) at 24 kHz (Audible Speech)
```

---

## 2. Acoustic Modeling: Tacotron 2 vs. FastSpeech

### 2.1 Autoregressive: Tacotron 2 (Shen et al., 2018)
- Employs an attention-based recurrent decoder generating Mel spectrogram frames one by one autoregressively ($m_t = f(m_{<t}, \mathbf{h}_{\text{text}})$).
- **Failure Modes**: Slow sequential inference, attention alignment errors (skipping words or catastrophic repeating loops), and exposure bias.

### 2.2 Non-Autoregressive: FastSpeech (Ren et al., 2019)
- Generates all Mel frames in parallel using a feedforward Transformer architecture.
- **Length Regulator**: A dedicated duration predictor predicts the exact number of acoustic frames each phoneme should occupy:
  $$\hat{d}_i = \text{round}(\text{DurationPredictor}(\mathbf{h}_i))$$
  The Length Regulator expands each phoneme representation $\mathbf{h}_i$ by repeating it $\hat{d}_i$ times, perfectly bridging the length gap between phoneme sequence length $N$ and acoustic frame length $T_{\text{mel}}$.
- **Benefits**: $270\times$ faster synthesis, zero skipped words, and fine-grained controllable speed and pitch.

---

## 3. Vocoding: Inverting Spectrograms to Waveforms

A log-Mel spectrogram captures energy magnitude but completely discards phase information. Without phase, an inverse Fourier transform cannot reconstruct a clean, natural acoustic wave.

1. **Griffin-Lim Algorithm (Griffin & Lim, 1984)**:
   A classical iterative algorithm that estimates phase by alternating between STFT and iSTFT projections, replacing the magnitude at each step with the ground-truth target. Slower and produces robotic, metallic artifacts.
2. **Autoregressive Neural Vocoders (WaveNet; van den Oord et al., 2016)**:
   Predicts audio sample by sample conditioned on dilated causal convolutions: $P(x_t \mid x_{<t}, \mathbf{M})$. High fidelity, but computationally expensive.
3. **GAN-Based Vocoders (HiFi-GAN; Kong et al., 2020)**:
   Uses transposed convolutions with Multi-Period Discriminators (MPD) and Multi-Scale Discriminators (MSD). Synthesizes ultra-high-fidelity speech faster than real time on consumer CPUs.

---

## 4. Evaluation: Mel-Cepstral Distortion (MCD)

Objective evaluation compares the synthesized Mel-frequency cepstrum against a parallel human reference:
$$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \frac{1}{T} \sum_{t=1}^T \sqrt{\sum_{d=1}^D (c_d^{\text{pred}}(t) - c_d^{\text{ref}}(t))^2} \quad [\text{in dB}]$$
Subjective quality is benchmarked via **Mean Opinion Score (MOS)** on a scale from 1 (unacceptable) to 5 (excellent).

---

## 5. Implementation Blueprint

- [`code/tts_engine.py`](code/tts_engine.py): Pure PyTorch/SciPy implementations of `LengthRegulator`, `MiniAcousticTTS`, `griffin_lim_reconstruction`, and `compute_mcd`.
- [`code/test_tts.py`](code/test_tts.py): Unit tests for length regulation expansion, acoustic forward loss, Griffin-Lim phase retrieval, and MCD computation.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring duration regulation, acoustic text-to-mel generation, and Griffin-Lim waveform inversion.
