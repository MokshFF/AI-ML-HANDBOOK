"""
Transformer Downstream Architectures: Sequence Classification, NER, and Extractive QA.

Implements lightweight, pure PyTorch transformer encoder modules and task-specific heads:
- MiniBertEncoder: Multi-head self-attention + MLP with residual connections and layer norm.
- BertForSequenceClassification: Sentence / text-level sentiment & intent classification.
- BertForTokenClassification: Per-token classification for Named Entity Recognition (NER).
- BertForQuestionAnswering: Span boundary prediction for extractive Question Answering (SQuAD-style).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBertEmbeddings(nn.Module):
    """
    Constructs token, position, and token-type (segment) embeddings.
    """
    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int = 512, type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = words + positions + token_types
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MiniBertLayer(nn.Module):
    """
    Single Transformer encoder block: Multi-head self-attention + Feedforward network.
    """
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Multi-Head Self-Attention
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: (batch_size, 1, 1, seq_len) with 0 for valid and large negative for masked
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Add & Norm
        normed1 = self.norm1(hidden_states + self.dropout1(self.out_proj(context)))

        # FFN + Add & Norm
        output = self.norm2(normed1 + self.dropout2(self.ffn(normed1)))
        return output


class MiniBertModel(nn.Module):
    """
    Lightweight BERT backbone returning sequence representations and pooled [CLS] vector.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_dim: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = MiniBertEmbeddings(vocab_size, hidden_dim, max_seq_len=max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList([
            MiniBertLayer(hidden_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare attention mask for multi-head attention: (B, 1, 1, S)
        ext_mask = None
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        hidden = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=ext_mask)

        # Pooled output from first token [CLS] (index 0)
        cls_token = hidden[:, 0, :]
        pooled = self.pooler(cls_token)
        return hidden, pooled


class BertForSequenceClassification(nn.Module):
    """
    Classification head on top of BERT pooled [CLS] representation.
    """
    def __init__(self, bert: MiniBertModel, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embeddings.word_embeddings.embedding_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}


class BertForTokenClassification(nn.Module):
    """
    Token-level classification head for Named Entity Recognition (NER) and POS tagging.
    """
    def __init__(self, bert: MiniBertModel, num_labels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embeddings.word_embeddings.embedding_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            # Flatten to compute cross entropy, ignoring padded tokens (label == -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss}


class BertForQuestionAnswering(nn.Module):
    """
    Extractive QA head predicting start and end span token logits.
    """
    def __init__(self, bert: MiniBertModel):
        super().__init__()
        self.bert = bert
        self.qa_outputs = nn.Linear(bert.embeddings.word_embeddings.embedding_dim, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)      # (batch_size, seq_len)

        loss = None
        if start_positions is not None and end_positions is not None:
            # If positions fall outside seq_len, clamp to seq_len - 1
            max_len = input_ids.size(1)
            start_positions = start_positions.clamp(0, max_len - 1)
            end_positions = end_positions.clamp(0, max_len - 1)
            
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2.0

        return {"start_logits": start_logits, "end_logits": end_logits, "loss": loss}


def extract_best_answer_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_span_length: int = 15,
) -> Tuple[int, int]:
    """
    Finds the optimal start and end index (i, j) maximizing score[i] + score[j]
    subject to i <= j and (j - i + 1) <= max_span_length.
    """
    start_probs = F.softmax(start_logits, dim=-1).squeeze().tolist()
    end_probs = F.softmax(end_logits, dim=-1).squeeze().tolist()

    if isinstance(start_probs, float):
        return 0, 0

    best_score = -float("inf")
    best_span = (0, 0)

    seq_len = len(start_probs)
    for i in range(seq_len):
        for j in range(i, min(seq_len, i + max_span_length)):
            score = start_probs[i] + end_probs[j]
            if score > best_score:
                best_score = score
                best_span = (i, j)

    return best_span
