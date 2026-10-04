# Cheat Sheet: Transformer Architecture & Mechanics

| Component | Standard Formulation | Modern Production Variant | Rationale |
| :--- | :--- | :--- | :--- |
| **Attention** | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | Reduces KV cache memory by $4-8\times$ |
| **Positional Encoding** | Sinusoidal absolute embeddings | Rotary Position Embedding (RoPE) | Relative position invariance; context extension via YaRN |
| **Normalization** | Post-LayerNorm ($x = \text{LN}(x + F(x))$) | Pre-RMSNorm ($x = x + F(\text{RMSNorm}(x))$) | Stable gradient propagation; $15\%$ faster execution |
| **Activation** | ReLU or standard GELU | SwiGLU: $\text{Swish}(x W) \cdot (x V)$ | Superior downstream perplexity at equal parameter count |
| **Attention Execution** | Materialized $\mathcal{O}(N^2)$ memory matrix | FlashAttention-2 / FlashAttention-3 | IO-aware SRAM tiling; zero HBM quadratic memory |
| **Feed-Forward Expansion** | Hidden dim $4 d_{model}$ | Hidden dim $\frac{8}{3} d_{model}$ (SwiGLU) | Preserves FLOP parity with gated linear units |
