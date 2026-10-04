# Audio Fundamentals & DSP - Technical Interview Preparation

A curated question bank covering sampling theorems, windowing functions, time-frequency resolution trade-offs, Mel filterbanks, and the rationale behind MFCCs.

---

## 1. Core Mathematical & Conceptual Foundations

### Q1: What is the Heisenberg-Gabor Uncertainty Principle in audio signal processing, and how does it manifest in STFT window length selection?
- **The Uncertainty Principle**:
  $$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
  One cannot simultaneously achieve arbitrarily high time resolution ($\Delta t$) and arbitrarily high frequency resolution ($\Delta f$).
- **Window Length Trade-Off**:
  - *Short Window (e.g. 5 ms, 80 samples at 16 kHz)*: High temporal precision (excellent for capturing fast transient events like plosive consonants /p/, /t/, /k/), but poor frequency resolution (frequency bins are wide; cannot resolve individual harmonic pitches).
  - *Long Window (e.g. 50 ms, 800 samples)*: High frequency precision (clear harmonic lines for vowel pitch tracking), but temporal smearing (cannot localize transient boundary clicks).
  - *Standard compromise*: A window of $20 - 25\text{ ms}$ with a $10\text{ ms}$ frame shift (hop).

---

### Q2: Why apply a window function (e.g., Hann or Hamming) before computing the FFT, rather than using a rectangular window?
- **Spectral Leakage**:
  The Discrete Fourier Transform assumes that the finite $N$-sample slice is infinitely periodic. If the signal has non-zero values at the boundaries, cutting it abruptly with a rectangular window introduces sharp step discontinuities at the ends.
- **Sidelobe Suppression**:
  In the frequency domain, a rectangular window corresponds to convolution with a sinc function whose high sidelobes fall off slowly ($-13\text{ dB}$ for the first sidelobe), causing energy from strong frequencies to leak into distant frequency bins (spectral leakage).
- **Hann Window**:
  Tapers the signal smoothly to zero at the frame boundaries ($w[n] \to 0$ as $n \to 0, N-1$). The first sidelobe is suppressed to $-31.5\text{ dB}$, eliminating spurious high-frequency spectral artifacts.

---

### Q3: Why are Log-Mel Spectrograms preferred over MFCCs in modern deep learning (Whisper, Conformer), whereas MFCCs were dominant in classical GMM-HMM systems?
- **Historical GMM-HMM Era**:
  Gaussian Mixture Models used diagonal covariance matrices for computational tractability. Diagonal covariance assumes all input feature dimensions are uncorrelated. Raw filterbank energies are highly correlated across adjacent frequency bands, violating the diagonal covariance assumption. Applying the DCT produces orthogonal, decorrelated **MFCCs**, making diagonal GMMs effective.
- **Deep Learning Era (CNNs & Transformers)**:
  1. *Spatial Locality*: A log-Mel spectrogram retains 2D spatial-temporal topology: adjacent bins are locally correlated in both time and frequency, which is exactly the structure 2D CNNs and Vision/Audio Transformers exploit.
  2. *Loss of Information in DCT*: The DCT discards phase and compresses non-linear relationships. Deep neural networks learn superior non-linear representations directly from raw log-Mel energies without manual linear DCT projections.

---

## 2. Whiteboard Coding Drills

### Q4: Implement a pure NumPy function to convert linear frequency in Hz to Mel scale and back.
```python
import numpy as np

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
```
