"""
Unit tests for Audio Digital Signal Processing & Feature Extraction.
"""

import numpy as np
import pytest
from audio_dsp import (
    generate_synthetic_tone,
    compute_stft_spectrogram,
    amplitude_to_db,
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    compute_mel_spectrogram,
    compute_mfcc,
)


def test_synthetic_tone_generation():
    sr = 16000
    duration = 0.5
    t, waveform = generate_synthetic_tone(frequencies=(440.0, 880.0), duration=duration, sample_rate=sr)

    assert len(t) == int(sr * duration)
    assert len(waveform) == int(sr * duration)
    assert np.max(np.abs(waveform)) <= 1.0


def test_stft_and_amplitude_to_db():
    _, waveform = generate_synthetic_tone(frequencies=(440.0,), duration=0.2, sample_rate=16000)
    freqs, times, power_spec = compute_stft_spectrogram(waveform, n_fft=512, hop_length=160, win_length=400)

    # FFT bins = n_fft // 2 + 1 = 257
    assert power_spec.shape[0] == 257
    assert len(freqs) == 257
    assert len(times) == power_spec.shape[1]

    db_spec = amplitude_to_db(power_spec)
    assert db_spec.shape == power_spec.shape
    assert not np.isnan(db_spec).any()


def test_hz_mel_inversion():
    hz_orig = np.array([100.0, 1000.0, 4000.0, 8000.0])
    mels = hz_to_mel(hz_orig)
    hz_recon = mel_to_hz(mels)
    assert np.allclose(hz_orig, hz_recon, atol=1e-3)


def test_mel_filterbank_and_spectrogram():
    sr = 16000
    n_fft = 512
    num_mel = 40
    filterbank = create_mel_filterbank(sample_rate=sr, n_fft=n_fft, num_mel_bins=num_mel)

    assert filterbank.shape == (num_mel, n_fft // 2 + 1)
    # Filterbank weights non-negative
    assert (filterbank >= 0.0).all()

    _, waveform = generate_synthetic_tone(frequencies=(500.0,), duration=0.2, sample_rate=sr)
    mel_spec = compute_mel_spectrogram(waveform, sample_rate=sr, n_fft=n_fft, num_mel_bins=num_mel)

    assert mel_spec.shape[0] == num_mel
    assert not np.isnan(mel_spec).any()


def test_compute_mfcc():
    sr = 16000
    _, waveform = generate_synthetic_tone(frequencies=(300.0, 1200.0), duration=0.2, sample_rate=sr)
    mel_spec = compute_mel_spectrogram(waveform, sample_rate=sr, num_mel_bins=40)
    mfcc = compute_mfcc(mel_spec, num_mfcc=13)

    assert mfcc.shape[0] == 13
    assert mfcc.shape[1] == mel_spec.shape[1]
    assert not np.isnan(mfcc).any()
