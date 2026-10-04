# Automatic Speech Recognition (ASR) - Technical Interview Preparation

A curated question bank covering CTC alignment mechanics, the conditional independence assumption of CTC, beam search decoding with language models, and modern self-supervised ASR (Wav2Vec2 / Whisper).

---

## 1. Architectural & Theoretical Foundations

### Q1: What is the "Conditional Independence Assumption" in CTC, why is it a theoretical limitation, and how do RNN-Transducers (RNN-T) or Autoregressive ASR models solve it?
- **The CTC Conditional Independence Assumption**:
  At each time step $t$, CTC assumes that the probability of label $\pi_t$ is strictly independent of past and future labels $\pi_{<t}, \pi_{>t}$, conditioned only on the full input sequence $\mathbf{X}$:
  $$P(\pi \mid \mathbf{X}) = \prod_{t=1}^T P(\pi_t \mid \mathbf{X})$$
- **Why It Is a Limitation**:
  In natural human language, tokens are heavily interdependent (e.g., after the letters `"c"`, `"a"`, the probability of `"t"` or `"r"` is dramatically higher than `"q"`). Because CTC cannot condition predictions on previous output tokens, it behaves solely as an acoustic model and cannot function as an implicit language model. It requires an external n-gram language model during beam search to correct spelling.
- **The Solution**:
  - **RNN-Transducer (Graves, 2012)**: Adds a Prediction Network (RNN/LM) that conditions next token prediction jointly on acoustic features and previous non-blank label history: $P(y_u \mid \mathbf{h}_t^{\text{enc}}, \mathbf{h}_{u-1}^{\text{pred}})$.
  - **Encoder-Decoder (Whisper / LAS)**: Employs an autoregressive Transformer decoder with cross-attention over encoder states, naturally modeling full language dependencies.

---

### Q2: Why is the "blank token" necessary in CTC? Why not just collapse duplicate consecutive characters without it?
- **The Problem of Double Letters**:
  Words like `"letter"`, `"hello"`, or `"book"` contain consecutive identical characters.
- **Without Blank**:
  If an acoustic model outputs the sequence of frames `['l', 'l', 'e', 'e', 't', 't', 't', 't', 'e', 'r']` representing the word `"letter"`, simple deduplication would collapse it into `"leter"`. There would be no mathematical way to distinguish a prolonged single `"t"` from a genuine double `"tt"`.
- **With Blank**:
  The model outputs `['l', 'e', 't', '_', 't', 'e', 'r']`. The intervening blank token separates the two distinct phonetic instances of `'t'`, allowing the collapsing rule to produce the exact target `"letter"`.

---

### Q3: How does self-supervised speech pre-training (Wav2Vec 2.0 / HuBERT) work, and why does it require quantization of continuous audio representations?
- **The Problem**:
  Unlike NLP where text is discrete words/subwords, speech is a continuous, unsegmented waveform without natural discrete token boundaries. Masked language modeling (like BERT) cannot be directly applied because predicting raw continuous float vectors with cross-entropy is impossible.
- **Wav2Vec 2.0 Solution**:
  1. *Feature Encoder*: CNN downsamples raw waveform into continuous latent vectors $\mathbf{z}_t$.
  2. *Vector Quantization (Gumbel-Softmax)*: Discretizes $\mathbf{z}_t$ into discrete codebook entries $\mathbf{q}_t$.
  3. *Context Network*: Masked Transformer processes latent sequence $\mathbf{z}_t$ and predicts the quantized codebook entry $\mathbf{q}_t$ of masked time steps via contrastive loss.

---

## 2. Whiteboard Coding Drills

### Q4: Implement greedy CTC decoding in pure Python.
```python
from typing import List

def ctc_greedy_decode(frame_predictions: List[int], blank_idx: int = 0) -> List[int]:
    """
    Collapses repeated adjacent tokens and strips blank tokens.
    """
    collapsed = []
    prev_tok = None
    for tok in frame_predictions:
        if tok != prev_tok:
            if tok != blank_idx:
                collapsed.append(tok)
            prev_tok = tok
    return collapsed
```
