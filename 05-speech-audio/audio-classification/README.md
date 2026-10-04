# Audio Classification & Speaker Recognition

A comprehensive engineering guide to classifying acoustic events, environmental audio scenes, and biometric speaker verification using 2D Spectrogram Convolutional Networks (PANNs), speaker embeddings (d-vectors / x-vectors), and margin-based metric learning.

---

## 1. Acoustic Scene & Event Classification

Audio classification predicts categorical sound labels (e.g., speech, dog bark, engine noise, music genre) from an input audio clip.

### 2D Spectrogram ConvNet Framework
Rather than processing 1D raw waveforms directly, audio is transformed into a log-Mel time-frequency spectrogram $\mathbf{S} \in \mathbb{R}^{F \times T}$. The spectrogram is treated as a 1-channel 2D image $\mathbf{X} \in \mathbb{R}^{B \times 1 \times F \times T}$.
1. **2D Convolutions**: Learn local acoustic spectro-temporal patterns (formant sweeps, pitch harmonics, transient clicks).
2. **Frequency & Time Pooling**: Progressively downsamples both spectral bins ($F$) and temporal frames ($T$).
3. **Global Average Pooling**: Collapses remaining spatial dimensions into a fixed-length representation vector $\mathbf{h} \in \mathbb{R}^d$, regardless of input audio duration.
4. **Classification Layer**: $\mathbf{z} = \mathbf{W} \mathbf{h} + \mathbf{b}$, optimized via Categorical Cross-Entropy.

```
Waveform x(t) ──> STFT + Mel Bank ──> Log-Mel Spectrogram (1 x F x T)
                                                │
                                    [2D Conv + BN + ReLU]
                                                │
                                        [MaxPool 2x2]
                                                │
                                    [2D Conv + BN + ReLU]
                                                │
                                        [MaxPool 2x2]
                                                │
                              [Global Average Pooling (GAP)]
                                                │
                          Latent Audio Embedding h in R^d (d-vector)
                                       ┌────────┴────────┐
                                       │                 │
                             [Linear Classification]   [L2 Normalization]
                                       │                 │
                                Class Logits z        Unit Vector e in S^(d-1)
                                       │                 │
                                Cross-Entropy        Cosine Verification
```

---

## 2. Speaker Recognition: Identification vs. Verification

- **Speaker Identification (Closed-Set 1-to-N)**: Given an audio clip, classify which of $N$ known enrolled individuals produced the speech.
- **Speaker Verification (Open-Set 1-to-1)**: Given a claimed identity and an audio clip, verify whether the speaker is genuinely the claimed person or an impostor. Requires generalized metric embeddings.

### Speaker Embeddings: d-vectors & x-vectors
1. Pass audio through a deep neural network (TDNN or 2D CNN).
2. Extract the activation vector prior to the final classification layer.
3. Project onto the unit hypersphere:
   $$\mathbf{e} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2} \in \mathbb{S}^{d-1}$$
4. Verification score is the cosine similarity between the enrolled speaker template $\mathbf{e}_{\text{enrolled}}$ and test sample $\mathbf{e}_{\text{test}}$:
   $$\text{score} = \mathbf{e}_{\text{enrolled}}^\top \mathbf{e}_{\text{test}}$$

---

## 3. Metric Learning: Triplet Margin Loss

To learn robust open-set speaker embeddings, the network is trained using triplets comprising an **Anchor** ($a$), a **Positive** ($p$; same speaker), and a **Negative** ($n$; impostor speaker):
$$\mathcal{L}(a, p, n) = \max\left( 0, \, \|\mathbf{e}_a - \mathbf{e}_p\|_2^2 - \|\mathbf{e}_a - \mathbf{e}_n\|_2^2 + \alpha \right)$$
where $\alpha > 0$ enforces an explicit geometric margin separating genuine utterances from impostor utterances.

---

## 4. Biometric Evaluation: FAR, FRR & EER

Given verification threshold $\theta$:
- **False Acceptance Rate (FAR)**: Proportion of impostor pairs incorrectly accepted ($\text{score} \ge \theta$).
- **False Rejection Rate (FRR)**: Proportion of genuine speaker pairs incorrectly rejected ($\text{score} < \theta$).
- **Equal Error Rate (EER)**: The operational threshold where $\text{FAR}(\theta) = \text{FRR}(\theta)$. Lower is better.

---

## 5. Implementation Blueprint

- [`code/audio_classifier.py`](code/audio_classifier.py): PyTorch implementations of `SpectrogramCNNClassifier`, `SpeakerEmbeddingModel`, `compute_triplet_loss`, and `evaluate_verification`.
- [`code/test_audio_classification.py`](code/test_audio_classification.py): Unit tests for CNN feature shapes, L2 normalization, triplet margin arithmetic, and error rate metrics.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring audio classification forward passes, speaker embedding extraction, and verification evaluations.
