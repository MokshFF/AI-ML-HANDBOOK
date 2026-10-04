# Audio Fundamentals: Sampling, STFT Spectrograms, Mel Scale & MFCCs

A rigorous mathematical and digital signal processing guide to continuous audio waveforms, discrete sampling, windowed Short-Time Fourier Transforms (STFT), auditory Mel frequency filterbanks, and Mel-Frequency Cepstral Coefficients (MFCCs).

---

## 1. Acoustic Waveforms & Discrete Sampling

Continuous sound is a longitudinal pressure wave propagating through air. In digital audio, this continuous physical signal $x(t)$ is discretized at uniform sampling intervals $T_s = 1 / f_s$:
$$x[n] = x(n T_s), \quad n \in \mathbb{Z}$$

### The Nyquist-Shannon Sampling Theorem
To completely reconstruct a bandlimited continuous signal without aliasing (frequency folding), the sampling rate $f_s$ must be strictly greater than twice the maximum frequency present in the signal ($f_{\max}$):
$$f_s > 2 f_{\max} \implies f_{\text{Nyquist}} = \frac{f_s}{2}$$
- **Speech standard**: $16\text{ kHz}$ ($f_{\text{Nyquist}} = 8\text{ kHz}$ captures standard human vocal tract formants).
- **High-fidelity audio/music**: $44.1\text{ kHz}$ or $48\text{ kHz}$ ($f_{\text{Nyquist}} \approx 22\text{ kHz}$ covers the upper limit of human hearing, $\sim 20\text{ kHz}$).

---

## 2. Short-Time Fourier Transform (STFT) & Spectrograms

Because audio signals are non-stationary (spectral content evolves rapidly over time), a global Fourier Transform loses temporal localization. The STFT resolves this by sliding a finite-length window $w[n]$ across time:
$$\mathbf{X}(m, \omega) = \sum_{n=-\infty}^\infty x[n] w[n - m R] e^{-j \omega n}$$
where $m$ is the frame index, $R$ is the frame shift / hop length (e.g., 10 ms = 160 samples at 16 kHz), and $w$ is typically a **Hann window**:
$$w[n] = 0.5 \left( 1 - \cos\left( \frac{2\pi n}{N-1} \right) \right)$$

### Linear Power Spectrogram
$$P(m, k) = |\mathbf{X}(m, k)|^2, \quad k \in \left[0, \frac{N_{\text{FFT}}}{2}\right]$$
Converted to logarithmic decibels (dB) to match logarithmic human auditory perception:
$$\text{dB}(m, k) = 10 \log_{10}\left( \frac{\max(\epsilon, P(m, k))}{P_{\text{ref}}} \right)$$

```
Continuous Audio Waveform x(t)
            │
    [Discrete Sampling at fs = 16 kHz]
            │
   [Framing & Hann Windowing (Win: 25ms, Hop: 10ms)]
            │
       [Fast Fourier Transform (FFT)]
            │
  Linear Power Spectrogram P(f, t)  (FFT bins x Time frames)
            │
  [Mel Triangular Filterbank Warping (40-80 bins)]
            │
   Log-Mel Spectrogram S_mel(m, t)  (Mel bins x Time frames)
            │
  [Discrete Cosine Transform (DCT-II)]
            │
 Mel-Frequency Cepstral Coefficients (MFCCs: 13-20 coeffs)
```

---

## 3. The Perceptual Mel Scale & Filterbanks

Human pitch perception is non-linear: our cochlea distinguishes small pitch differences much more accurately at low frequencies (e.g. 200 Hz vs 300 Hz) than at high frequencies (e.g. 5000 Hz vs 5100 Hz).

### The Mel Formula (Stevens, Volkmann & Newman, 1937)
$$m = 2595 \log_{10}\left( 1 + \frac{f}{700} \right)$$
$$f = 700 \left( 10^{m / 2595} - 1 \right)$$
Triangular overlapping filterbanks $H_m(k)$ are spaced linearly below 1,000 Hz and logarithmically above 1,000 Hz:
$$\mathbf{M}(m, t) = \sum_{k=0}^{N_{\text{FFT}}/2} H_m(k) P(k, t)$$

---

## 4. Mel-Frequency Cepstral Coefficients (MFCCs)

Adjacent Mel filterbank channels are highly correlated because individual vocal tract resonances (formants) span multiple triangular filters. Applying the **Type-II Discrete Cosine Transform (DCT)** decorrelates these energies and compacts information into a small number of cepstral coefficients:
$$c_n = \sum_{m=1}^M \log(\mathbf{M}_m) \cos\left[ \frac{\pi n}{M} \left(m - \frac{1}{2}\right) \right], \quad n \in [0, N_{\text{ceps}}-1]$$
- $c_0$: Proportional to total frame energy.
- $c_1$: Spectral tilt / slope.
- Lower coefficients ($c_1 - c_{12}$): Capture overall vocal tract shape (phonetic content).
- Higher coefficients: Represent fine pitch harmonics and excitation glottal pulses (often discarded to make speech recognition speaker-invariant).

---

## 5. Implementation Blueprint

- [`code/audio_dsp.py`](code/audio_dsp.py): NumPy/SciPy implementations of `generate_synthetic_tone`, `compute_stft_spectrogram`, `amplitude_to_db`, `hz_to_mel`, `create_mel_filterbank`, `compute_mel_spectrogram`, and `compute_mfcc`.
- [`code/test_audio_fundamentals.py`](code/test_audio_fundamentals.py): Unit tests verifying sampling arithmetic, STFT shapes, and DCT properties.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring time-domain signals, STFT spectrograms, Mel banks, and MFCC matrices.
