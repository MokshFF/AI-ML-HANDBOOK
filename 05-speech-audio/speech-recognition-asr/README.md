# Automatic Speech Recognition (ASR): Acoustic Models, CTC Loss & WER Evaluation

A comprehensive architectural and mathematical guide to Automatic Speech Recognition (ASR), Connectionist Temporal Classification (CTC), dynamic programming alignment marginalization, best-path greedy decoding, and Levenshtein edit-distance error metrics (WER / CER).

---

## 1. The Speech Recognition Alignment Problem

Automatic Speech Recognition maps a sequence of continuous acoustic observations $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ to a discrete text transcription $\mathbf{Y} = (y_1, \dots, y_U)$.
- **Temporal Mismatch**: Audio is sampled at high temporal resolution (e.g., 100 frames/sec for 10 ms hop), so $T \approx 1,000$ for a 10-second sentence. However, the transcript has only $U \approx 30$ words or $150$ characters ($T \gg U$).
- **Unknown Latent Alignment**: We do not possess ground-truth timestamps indicating exactly which audio frame corresponds to which character or phoneme.

---

## 2. Connectionist Temporal Classification (CTC; Graves et al., 2006)

CTC introduces a special **blank token** (`_` or index 0) and defines a collapsing function $\mathcal{B}: \Omega^T \to \Omega^{\le T}$ that:
1. Replaces consecutive identical tokens with a single token.
2. Removes all blank tokens.
   $$\mathcal{B}(\text{"c c \_ a a \_ t"}) = \text{"cat"}$$
   $$\mathcal{B}(\text{"h e l \_ l o"}) = \text{"hello"} \quad (\text{blank allows consecutive duplicate letters})$$

### Marginal Probability via Forward-Backward Algorithm
The conditional probability of target transcript $\mathbf{Y}$ is the sum of probabilities of all valid alignments $\pi \in \mathcal{B}^{-1}(\mathbf{Y})$:
$$P(\mathbf{Y} \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} P(\pi \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x}_t)$$
$$\mathcal{L}_{\text{CTC}} = - \log P(\mathbf{Y} \mid \mathbf{X})$$
This sum is computed efficiently in $O(T \cdot U)$ time using the **forward-backward dynamic programming** trellis.

```
Acoustic Features (Spectrogram)
              │
  [1D Temporal Conv Downsampling]  (Reduces T -> T / 2)
              │
  [Deep Bidirectional GRU / Conformer]
              │
  [Linear Projection to Vocab + Blank]
              │
 Frame-wise Log-Probabilities P(pi_t | x)  (T_out x B x Vocab)
              │
 ┌────────────┴────────────┐
 │                         │
[CTC Loss (Training)]   [Greedy / Beam Search (Inference)]
 │                         │
Marginalize Valid Paths   Collapse Repeated & Strip Blanks
                           │
                         Decoded Text: "hello world"
```

---

## 3. Decoding Algorithms

1. **Greedy (Best-Path) Decoding**:
   $$\pi^* = \arg\max_\pi \prod_{t=1}^T P(\pi_t \mid \mathbf{x}_t), \quad \hat{\mathbf{Y}} = \mathcal{B}(\pi^*)$$
   Fast ($O(T)$), but ignores path merging (multiple distinct paths that collapse to the same text could have higher joint probability).
2. **CTC Prefix Beam Search**:
   Maintains top-$K$ candidate text prefixes, accumulating probability mass across all paths that map to each prefix, optionally integrated with an external n-gram language model:
   $$\text{Score}(\mathbf{Y}) = \log P_{\text{CTC}}(\mathbf{Y} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{Y}) + \beta |\mathbf{Y}|$$

---

## 4. Evaluation Metrics: WER and CER

The Levenshtein minimum edit distance computes the minimum number of insertions ($I$), deletions ($D$), and substitutions ($S$) required to transform the hypothesis string into the reference string:
$$\text{WER} = \frac{S_{\text{word}} + D_{\text{word}} + I_{\text{word}}}{N_{\text{word}}}$$
$$\text{CER} = \frac{S_{\text{char}} + D_{\text{char}} + I_{\text{char}}}{N_{\text{char}}}$$

---

## 5. Modern Architectures: Wav2Vec 2.0, Conformer & Whisper

- **Wav2Vec 2.0 (Baevski et al., 2020)**: Self-supervised masked latent representation learning directly from raw waveforms, fine-tuned with CTC.
- **Conformer (Gulati et al., 2020)**: Interleaves multi-head self-attention with depthwise separable convolutions to capture both global context and local acoustic features.
- **Whisper (Radford et al., 2022)**: Weakly supervised encoder-decoder Transformer trained on 680,000 hours of multilingual audio using sequence-to-sequence autoregressive text generation.

---

## 6. Implementation Blueprint

- [`code/ctc_engine.py`](code/ctc_engine.py): Pure PyTorch implementations of `MiniAcousticEncoder`, `ctc_greedy_decode`, `compute_edit_distance`, `compute_wer`, and `compute_cer`.
- [`code/test_asr.py`](code/test_asr.py): Unit tests verifying temporal downsampling, PyTorch CTC loss computation, collapsing logic, and Levenshtein distance.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab demonstrating acoustic model forward pass, CTC loss, greedy decoding, and error rate computation.
