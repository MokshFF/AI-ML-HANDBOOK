"""
Audio Digital Signal Processing (DSP): Waveforms, STFT Spectrograms, Mel Filterbanks, and MFCCs.

Implements pure NumPy and SciPy audio feature extraction:
- generate_synthetic_tone: Generates pure or multi-harmonic continuous audio waveforms.
- compute_stft_spectrogram: Computes Short-Time Fourier Transform (STFT) magnitude and power spectrograms.
- hz_to_mel & mel_to_hz: Auditory frequency warping functions.
- create_mel_filterbank: Triangular overlapping filterbank matrix.
- compute_mel_spectrogram: Converts audio waveform to logarithmic Mel spectrogram.
- compute_mfcc: Applies Discrete Cosine Transform (DCT-II) to extract Mel-Frequency Cepstral Coefficients.
"""

from typing import Tuple
import numpy as np
import scipy.signal
import scipy.fftpack


def generate_synthetic_tone(
    frequencies: Tuple[float, ...] = (440.0,),
    duration: float = 1.0,
    sample_rate: int = 16000,
    noise_level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 1D synthetic discrete waveform composed of sinusoidal tones plus optional noise.
    Returns:
        time_axis: 1D array of time steps in seconds.
        waveform: 1D float32 audio signal normalized to [-1, 1].
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    signal = np.zeros_like(t)
    for f in frequencies:
        signal += np.sin(2.0 * np.pi * f * t)

    if noise_level > 0.0:
        signal += noise_level * np.random.randn(*signal.shape).astype(np.float32)

    # Normalize amplitude
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / max_amp

    return t, signal


def compute_stft_spectrogram(
    waveform: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes discrete Short-Time Fourier Transform (STFT) using a Hann window.
    Returns:
        frequencies: (n_fft // 2 + 1,)
        times: (num_frames,)
        power_spec: (n_fft // 2 + 1, num_frames) magnitude squared
    """
    # SciPy STFT
    frequencies, times, zxx = scipy.signal.stft(
        waveform,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window="hann",
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(zxx)
    power_spec = magnitude**2
    return frequencies, times, power_spec.astype(np.float32)


def amplitude_to_db(power: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Converts power spectrogram to Decibel (dB) scale:
    dB = 10 * log10(max(amin, power) / ref)
    """
    clipped = np.maximum(amin, power)
    return (10.0 * np.log10(clipped / ref)).astype(np.float32)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """
    Converts linear frequency (Hz) to perceptual Mel scale (Stevens et al.):
    m = 2595 * log10(1 + f / 700)
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """
    Converts Mel scale back to linear frequency (Hz):
    f = 700 * (10^(m / 2595) - 1)
    """
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    num_mel_bins: int = 40,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Constructs a triangular Mel filterbank matrix of shape (num_mel_bins, n_fft // 2 + 1).
    """
    num_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2.0, num_fft_bins)

    # Linearly spaced points in Mel scale
    min_mel = hz_to_mel(np.array(f_min))
    max_mel = hz_to_mel(np.array(f_max))
    mel_points = np.linspace(min_mel, max_mel, num_mel_bins + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin indices corresponding to Hz points
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_mel_bins, num_fft_bins), dtype=np.float32)
    for m in range(1, num_mel_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        # Up-slope
        for k in range(f_m_minus, f_m):
            if k < num_fft_bins:
                filterbank[m - 1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
        # Down-slope
        for k in range(f_m, f_m_plus):
            if k < num_fft_bins:
                filterbank[m - 1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)

    return filterbank


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    num_mel_bins: int = 40,
) -> np.ndarray:
    """
    Computes Log-Mel Spectrogram (dB) for an input audio waveform.
    Returns:
        log_mel_spec: (num_mel_bins, num_frames) in dB scale.
    """
    _, _, power_spec = compute_stft_spectrogram(waveform, n_fft, hop_length, win_length)
    filterbank = create_mel_filterbank(sample_rate, n_fft, num_mel_bins, f_min=0.0, f_max=sample_rate / 2.0)
    mel_power = np.dot(filterbank, power_spec)
    log_mel = amplitude_to_db(mel_power, ref=np.max(mel_power) if np.max(mel_power) > 0 else 1.0)
    return log_mel


def compute_mfcc(
    log_mel_spectrogram: np.ndarray,
    num_mfcc: int = 13,
) -> np.ndarray:
    """
    Applies Type-II Discrete Cosine Transform (DCT) across Mel bins to decorrelate filterbank energies.
    Args:
        log_mel_spectrogram: (num_mel_bins, num_frames)
    Returns:
        mfcc: (num_mfcc, num_frames)
    """
    # Apply DCT along frequency axis (axis 0)
    mfcc = scipy.fftpack.dct(log_mel_spectrogram, type=2, axis=0, norm="ortho")
    return mfcc[:num_mfcc, :].astype(np.float32)
