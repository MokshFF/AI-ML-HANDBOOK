# Cheat Sheet: Large Language Models (LLMs)

| Phase / Topic | Core Concepts | Industry Standards | Failure Modes / Mitigations |
| :--- | :--- | :--- | :--- |
| **Pre-Training** | Chinchilla Scaling: $D \approx 20 \times P$ (20 tokens per param) | LLaMA-3 (15T tokens on 8B model) | Loss spikes; gradient clipping, loss checkpointing |
| **Fine-Tuning (PEFT)** | LoRA: $\Delta W = B \cdot A$ with rank $r \in [8, 64]$ | QLoRA (NF4 base weights + LoRA) | Catastrophic forgetting; add replay regularizers |
| **Alignment** | DPO: Closed-form policy preference loss | DPO replacing complex RLHF PPO | Policy collapse under low $\beta$; use length normalization |
| **Inference Engine** | Continuous batching, PagedAttention, Speculative decoding | vLLM, TensorRT-LLM, TGI | KV cache OOM; enforce strict max context limits |
| **Quantization** | Weight-Only (AWQ/GPTQ INT4), Weight-Activation (FP8) | FP8 on Hopper (H100); INT4 on edge | Activation outliers; smoothquant / AWQ protection |
