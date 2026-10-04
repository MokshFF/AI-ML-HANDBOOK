# Transformer Models & Downstream Heads - Technical Interview Preparation

A curated question bank covering BERT pre-training, fine-tuning dynamics, subword handling in NER, span prediction in QA, and sequence classification pooling.

---

## 1. Architectural & Training Concepts

### Q1: Why does BERT use the `[CLS]` token for sequence-level tasks rather than mean-pooling all token representations?
- **Theoretical Design**:
  In a deep Transformer with multi-head self-attention, every token attends to every other token at each layer. By pre-training with Next Sentence Prediction (NSP), the `[CLS]` token at position 0 is explicitly trained to aggregate global sequence-level properties.
- **Mean-Pooling vs. [CLS]**:
  - `[CLS]` provides a clean, single-vector bottleneck directly tied to the dense pooler layer ($\tanh(\mathbf{W} \mathbf{h}_{[\text{CLS}]} + \mathbf{b})$).
  - However, empirical research (e.g., Sentence-BERT; Reimers & Gurevych, 2019) demonstrated that for *unsupervised semantic textual similarity (STS)* without fine-tuning, mean-pooling across all token hidden states often yields superior semantic embeddings compared to raw `[CLS]`.

---

### Q2: How should subword tokenization (WordPiece/BPE) be handled during Token Classification (NER)?
- **The Problem**:
  Named entity boundaries are defined at the word level, but subword tokenizers split words into multiple pieces (e.g., `"Washington"` $\to$ `["Wash", "##ing", "##ton"]`).
- **Standard Solutions**:
  1. **First-Token Labeling**: Assign the true entity label (e.g., `B-LOC`) to the first subword token (`"Wash"`), and assign a special mask label (conventionally `-100` in PyTorch) to the remaining continuation subwords (`"##ing"`, `"##ton"`). PyTorch's `CrossEntropyLoss(ignore_index=-100)` ignores these positions during gradient backpropagation.
  2. **Subword Replication**: Assign the label (or its continuation tag `I-LOC`) to all subwords. In practice, first-token labeling is preferred because it prevents models from being over-penalized for complex morphology.

---

### Q3: How does Extractive Question Answering handle unanswerable questions (SQuAD 2.0)?
- **SQuAD 2.0 Formulation**:
  Some questions cannot be answered from the provided passage. The model must produce a "no-answer" prediction.
- **Mechanism**:
  The `[CLS]` token (index 0) serves as the null answer candidate.
  - If the highest scoring span is $(0, 0)$, or if $\max_{(i, j)} (s_i + e_j) - (s_0 + e_0) < \tau$ (where $\tau$ is a tunable threshold), the model predicts that the question is unanswerable.

---

## 2. Whiteboard Coding Drills

### Q4: Implement the optimal span extraction algorithm for Extractive QA given start and end logit tensors.
```python
import torch

def extract_best_answer_span(start_logits: torch.Tensor, end_logits: torch.Tensor, max_span_length: int = 15):
    """
    Finds (i, j) maximizing start_probs[i] + end_probs[j] subject to 0 <= i <= j < N
    and (j - i + 1) <= max_span_length.
    """
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    
    n = len(start_probs)
    best_score = -float("inf")
    best_span = (0, 0)
    
    for i in range(n):
        for j in range(i, min(n, i + max_span_length)):
            score = start_probs[i].item() + end_probs[j].item()
            if score > best_score:
                best_score = score
                best_span = (i, j)
                
    return best_span
```
