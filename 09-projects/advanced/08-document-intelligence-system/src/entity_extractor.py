"""Multimodal Text + Spatial Layout Token Classifier."""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from document_parser import SAMPLE_INVOICE_TOKENS, ENTITY_MAP, REV_ENTITY_MAP

class DocumentEntityExtractor(nn.Module):
    def __init__(self, vocab_size: int = 500, embed_dim: int = 32, n_classes: int = 5):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        # Spatial 2D coordinates: x0, y0, x1, y1 (normalized to 1000)
        self.x_embed = nn.Embedding(1001, embed_dim)
        self.y_embed = nn.Embedding(1001, embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )

    def forward(self, token_ids: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, SeqLen)
        boxes: (B, SeqLen, 4) -> [x0, y0, x1, y1]
        """
        t_emb = self.text_embed(token_ids)
        x0_emb = self.x_embed(boxes[:, :, 0].clamp(0, 1000))
        y0_emb = self.y_embed(boxes[:, :, 1].clamp(0, 1000))
        x1_emb = self.x_embed(boxes[:, :, 2].clamp(0, 1000))
        y1_emb = self.y_embed(boxes[:, :, 3].clamp(0, 1000))
        
        fused = torch.cat([t_emb, x0_emb, y0_emb, x1_emb, y1_emb], dim=-1)
        logits = self.fusion(fused)
        return logits

def parse_document_to_json(tokens: List[Dict[str, Any]], model: DocumentEntityExtractor) -> Dict[str, str]:
    vocab = {t["text"]: (hash(t["text"]) % 490 + 1) for t in tokens}
    t_ids = torch.tensor([[vocab[t["text"]] for t in tokens]])
    boxes = torch.tensor([[t["box"] for t in tokens]])
    
    with torch.no_grad():
        logits = model(t_ids, boxes)
        preds = logits.squeeze(0).argmax(dim=-1).tolist()
        
    extracted = {}
    for t, pred_label_idx in zip(tokens, preds):
        label = REV_ENTITY_MAP.get(pred_label_idx, "O")
        if label != "O":
            if label not in extracted:
                extracted[label] = []
            extracted[label].append(t["text"])
            
    return {k: " ".join(v) for k, v in extracted.items()}

if __name__ == "__main__":
    model = DocumentEntityExtractor()
    model.eval()
    result = parse_document_to_json(SAMPLE_INVOICE_TOKENS, model)
    print("Parsed Extracted Document Entities:")
    print(result)
