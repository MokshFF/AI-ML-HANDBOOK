# Transformer Downstream Architectures: BERT, NER & Extractive QA

A rigorous conceptual, mathematical, and hands-on guide to downstream NLP architectures built upon the Transformer encoder framework, with dedicated heads for sequence classification, token classification (Named Entity Recognition), and extractive question answering.

---

## 1. Architectural Foundations: BERT & Encoder Backbones

Bidirectional Encoder Representations from Transformers (BERT; Devlin et al., 2018) revolutionized natural language processing by pre-training deep bidirectional representations using masked language modeling (MLM) and next-sentence prediction (NSP).

### Input Representation Triplet
For each token at sequence position $i$, the final input embedding $\mathbf{e}_i \in \mathbb{R}^{d}$ is the element-wise sum of three independent embedding lookups followed by Layer Normalization:
$$\mathbf{e}_i = \text{LayerNorm}(\mathbf{e}_{\text{word}}(x_i) + \mathbf{e}_{\text{pos}}(i) + \mathbf{e}_{\text{seg}}(s_i))$$
- **Token Embedding** ($\mathbf{e}_{\text{word}}$): WordPiece subword vocabulary lookup.
- **Position Embedding** ($\mathbf{e}_{\text{pos}}$): Learned or sinusoidal coordinate encoding position $i \in [0, N-1]$.
- **Segment Embedding** ($\mathbf{e}_{\text{seg}}$): Distinguishes Sentence A from Sentence B ($s_i \in \{0, 1\}$).

```
Tokens:      [CLS]    Machine    learning    is    fun    [SEP]
Segments:      0         0          0         0     0       0
Positions:     0         1          2         3     4       5
                 \       |          |         |     |       /
               Element-wise Sum + LayerNorm + Dropout
                                 │
              ┌──────────────────┴──────────────────┐
              │    Transformer Encoder Layer x L    │
              │  - Multi-Head Self-Attention        │
              │  - Residual + LayerNorm             │
              │  - Position-wise FFN (GELU)         │
              │  - Residual + LayerNorm             │
              └──────────────────┬──────────────────┘
                                 │
           Sequence Hidden Output: H in R^(B x N x d)
```

---

## 2. Downstream Task Heads

### 2.1 Sequence Classification (Sentiment & Intent)
For document-level or sentence-level prediction, BERT designates the first token `[CLS]` as the summary representation.
1. **Pooler Layer**:
   $$\mathbf{h}_{\text{pool}} = \tanh(\mathbf{W}_{\text{pool}} \mathbf{h}_{[\text{CLS}]} + \mathbf{b}_{\text{pool}}), \quad \mathbf{W}_{\text{pool}} \in \mathbb{R}^{d \times d}$$
2. **Classification Projection**:
   $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_c \mathbf{h}_{\text{pool}} + \mathbf{b}_c), \quad \mathbf{W}_c \in \mathbb{R}^{C \times d}$$
3. **Objective**: Categorical Cross-Entropy across classes $C$.

### 2.2 Token Classification (Named Entity Recognition - NER)
For sequence labeling (e.g., BIO tagging for Persons, Locations, Organizations), classification is performed at every token position:
$$\mathbf{z}_i = \mathbf{W}_{\text{ner}} \mathbf{h}_i + \mathbf{b}_{\text{ner}}, \quad \mathbf{W}_{\text{ner}} \in \mathbb{R}^{K \times d}, \quad \forall i \in [1, N]$$
$$\mathcal{L}_{\text{NER}} = - \frac{1}{\sum_{i} \mathbb{I}(y_i \ne -100)} \sum_{i: y_i \ne -100} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{k=1}^K \exp(z_{i, k})} \right)$$
Subword pieces following the head token and padding tokens are masked with label `-100` so they do not skew gradient updates.

### 2.3 Extractive Question Answering (SQuAD)
Given a packed sequence `[CLS] Question [SEP] Context Paragraph [SEP]`, the model identifies the continuous text span within the Context that answers the query.
1. The model projects each token's hidden state $\mathbf{h}_i$ to two scalar logits using linear parameter vectors $\mathbf{w}_s, \mathbf{w}_e \in \mathbb{R}^d$:
   $$s_i = \mathbf{w}_s^\top \mathbf{h}_i, \quad e_i = \mathbf{w}_e^\top \mathbf{h}_i$$
2. **Span Probabilities**:
   $$P_{\text{start}}(i) = \frac{\exp(s_i)}{\sum_k \exp(s_k)}, \quad P_{\text{end}}(j) = \frac{\exp(e_j)}{\sum_k \exp(e_k)}$$
3. **Span Selection Rule**: Find pair $(i^*, j^*)$ that maximizes $s_i + e_j$ subject to $i^* \le j^*$ and $j^* - i^* + 1 \le L_{\max}$.

---

## 3. Implementation Blueprint

All modules are implemented in pure PyTorch in [`code/transformer_tasks.py`](code/transformer_tasks.py):
- `MiniBertModel`: Self-contained multi-head self-attention transformer backbone.
- `BertForSequenceClassification`: Pooled classifier head with dropout.
- `BertForTokenClassification`: Linear per-token entity classifier with index masking.
- `BertForQuestionAnswering`: Dual start/end boundary logit head with `extract_best_answer_span()`.

---

## 4. Verification & Testing

Run unit tests verifying output shapes, masking behavior, and span decoding:
```bash
python -m pytest code/test_transformer_models.py -v
```
Interactive demonstrations and forward/backward gradient flows are provided in [`notebook.ipynb`](notebook.ipynb).
