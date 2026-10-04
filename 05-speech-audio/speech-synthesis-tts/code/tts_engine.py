"""
Speech Synthesis (TTS) Engine: Acoustic Modeling, FastSpeech Length Regulation, and Vocoding.

Implements from scratch using pure PyTorch and SciPy:
- GraphemeTokenizer: Simple phonetic vocabulary lookup.
- LengthRegulator: Expands phoneme embeddings according to duration values (FastSpeech mechanism).
- MiniAcousticTTS: Non-autoregressive text-to-mel spectrogram acoustic network.
- griffin_lim_reconstruction: Phase reconstruction algorithm converting magnitude spectrograms to waveforms.
- compute_mcd: Mel-Cepstral Distortion metric measuring spectral synthesis fidelity.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthRegulator(nn.Module):
    """
    Expands encoder phoneme representations to match temporal audio frames
    based on predicted or ground-truth duration values.
    """
    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_text, D) phoneme representations
            durations: (B, T_text) integer number of frames each phoneme lasts
        Returns:
            expanded: (B, sum(durations), D)
        """
        batch_size = x.size(0)
        expanded_batch = []

        for b in range(batch_size):
            expanded_tokens = []
            for t in range(x.size(1)):
                d = max(1, int(durations[b, t].item()))
                rep = x[b, t].unsqueeze(0).expand(d, -1)
                expanded_tokens.append(rep)
            expanded_batch.append(torch.cat(expanded_tokens, dim=0))

        # Pad to max length in batch
        max_len = max(item.size(0) for item in expanded_batch)
        d_dim = x.size(-1)
        out = torch.zeros(batch_size, max_len, d_dim, device=x.device)
        for b, item in enumerate(expanded_batch):
            out[b, :item.size(0), :] = item

        return out


class MiniAcousticTTS(nn.Module):
    """
    Non-autoregressive Text-to-Mel Acoustic Model (FastSpeech-style).
    Text Tokens -> Phoneme Encoder -> Length Regulator -> Mel Decoder -> Spectrogram.
    """
    def __init__(self, vocab_size: int = 30, embed_dim: int = 32, num_mels: int = 40):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.length_regulator = LengthRegulator()
        self.duration_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.ReLU(),  # Durations must be non-negative
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_mels),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Phoneme encoding
        x = self.embed(token_ids)
        x = self.encoder(x)

        # Predict durations
        pred_durations = self.duration_predictor(x).squeeze(-1)

        durations_to_use = target_durations if target_durations is not None else torch.clamp(torch.round(pred_durations), min=1)
        durations_to_use = durations_to_use.long()

        # Expand via Length Regulation
        expanded = self.length_regulator(x, durations_to_use)

        # Decode to Mel spectrogram: (B, T_mel, num_mels)
        mel_out = self.decoder(expanded)

        loss = None
        if target_mel is not None:
            # Align lengths for loss calculation
            min_len = min(mel_out.size(1), target_mel.size(1))
            mel_loss = F.l1_loss(mel_out[:, :min_len, :], target_mel[:, :min_len, :])
            loss = mel_loss

        return {
            "mel_out": mel_out,
            "pred_durations": pred_durations,
            "loss": loss,
        }


def griffin_lim_reconstruction(
    magnitude: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_iter: int = 16,
) -> np.ndarray:
    """
    Griffin-Lim phase retrieval algorithm.
    Iteratively estimates STFT phase from magnitude spectrogram and inverts to waveform via iSTFT.
    Args:
        magnitude: (n_fft // 2 + 1, num_frames)
    Returns:
        reconstructed_waveform: 1D float32 array
    """
    # Initialize with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft_estimate = magnitude * angles

    for _ in range(n_iter):
        # iSTFT to time domain
        _, waveform = scipy.signal.istft(
            stft_estimate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window="hann",
        )
        # STFT back to frequency domain
        _, _, stft_new = scipy.signal.stft(
            waveform,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window="hann",
        )
        # Update phase, replace magnitude with ground truth
        angles = np.exp(1j * np.angle(stft_new))
        stft_estimate = magnitude * angles

    _, final_waveform = scipy.signal.istft(
        stft_estimate,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window="hann",
    )
    return final_waveform.astype(np.float32)


def compute_mcd(mel_pred: np.ndarray, mel_target: np.ndarray) -> float:
    """
    Computes Mel-Cepstral Distortion (MCD) in decibels (dB):
    MCD = (10 * sqrt(2) / ln(10)) * mean_t( sqrt( sum_d (c_pred(d, t) - c_target(d, t))^2 ) )
    """
    min_t = min(mel_pred.shape[1], mel_target.shape[1])
    diff = mel_pred[:, :min_t] - mel_target[:, :min_t]
    frame_dist = np.sqrt(np.sum(diff**2, axis=0))
    scale = (10.0 * np.sqrt(2.0)) / np.log(10.0)
    return float(scale * np.mean(frame_dist))
