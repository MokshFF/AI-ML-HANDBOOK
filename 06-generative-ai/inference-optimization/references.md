# LLM Inference Optimization References & Seminal Papers

### Seminal Papers
- **Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)** (Kwon et al., 2023)  
  *Introduced PagedAttention and near-zero memory fragmentation for LLM serving.*  
  [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023)  
  *Foundational formulation of speculative decoding and lossless rejection sampling.*  
  [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., 2023)  
  *Pioneered activation-aware 4-bit weight quantization.*  
  [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023)  
  *Introduced Grouped-Query Attention bridging MHA quality and MQA memory speed.*  
  [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023)  
  [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

### Frameworks & Serving Engines
- **vLLM Project**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **TensorRT-LLM (NVIDIA)**: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **TGI (Hugging Face Text Generation Inference)**: [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
