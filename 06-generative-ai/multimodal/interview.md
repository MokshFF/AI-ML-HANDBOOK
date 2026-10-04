# Multimodal Generative AI Interview Questions & Answers

### Q1: How does the LLaVA architecture connect vision models to language models without retraining the entire LLM?
**Answer:**
LLaVA (Large Language and Vision Assistant) adopts a modular prefix-projection architecture:
1. **Vision Encoder**: Uses a frozen vision backbone (e.g., CLIP ViT-L/14) to extract grid patch tokens $[B, N_{\text{patches}}, D_{\text{vision}}]$.
2. **Projection Matrix / MLP**: A lightweight adapter (typically a 2-layer MLP with GeLU activation) projects the visual patch representations into the exact hidden dimension of the LLM $[B, N_{\text{patches}}, D_{\text{LLM}}]$.
3. **Unified Sequence**: The projected image tokens are prepended to the prompt token embeddings $[X_v, X_t]$.
4. **Two-Stage Training**:
   - *Stage 1 (Feature Alignment)*: Freeze both the vision encoder and LLM; train only the projection MLP on image-caption pairs so image tokens look like words to the LLM.
   - *Stage 2 (Visual Instruction Tuning)*: Unfreeze the LLM and fine-tune both the projector and LLM weights on conversational visual instruction data.

---

### Q2: What is Classifier-Free Guidance (CFG) in diffusion models, and why is it preferred over classifier guidance?
**Answer:**
- **Classifier Guidance** (Dhariwal & Nichol, 2021) required training a separate noisy image classifier $p_\phi(y | x_t)$ to guide the diffusion gradient $\nabla_{x_t} \log p_\phi(y | x_t)$. This classifier had to be trained on all noise levels and added significant complexity.
- **Classifier-Free Guidance (CFG)** (Ho & Salimans, 2022) trains a single conditional diffusion model $\epsilon_\theta(x_t, c)$, where the conditioning $c$ is randomly dropped (replaced with $\emptyset$) during 10-20% of training steps.
- At inference time, the model evaluates both conditional and unconditional noise predictions and combines them:
  $$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$
- When $s > 1.0$, the vector pointing from unconditional to conditional is amplified, pushing the sample toward high-likelihood regions of the conditional distribution without requiring any auxiliary classifier.

---

### Q3: What is the "Any-to-Any" multimodal paradigm, and how does it differ from dual-encoder models?
**Answer:**
- **Dual-Encoder Models (e.g., CLIP)**: Encode images and text via separate towers into a single shared vector space. They are optimized for similarity, retrieval, and zero-shot classification, but cannot generate new text or images.
- **Any-to-Any Models (e.g., Gemini, Chameleon, GPT-4o)**: Treat text, audio, images, and video as unified token streams. Modalities are tokenized (e.g., discrete image tokens via VQ-GAN or continuous patch embeddings) and processed jointly by an autoregressive Transformer. They can accept any combination of modalities as input and generate any combination as output.

---

### Q4: How does temporal downsampling work in Audio-Language models?
**Answer:**
Audio signals have extremely high sampling rates (e.g., 16 kHz = 16,000 samples/sec). Even after computing a Mel-spectrogram with a hop length of 10 ms, 10 seconds of audio yields 1,000 frames. Feeding 1,000 tokens into an LLM context window for a brief audio clip is inefficient.
Audio-language models (e.g., Whisper + LLM, AudioPaLM) use:
1. **1D Strided Convolutions or Pooling**: Applying convolutions with stride 2 or 4 reduces temporal resolution by $2\times - 8\times$.
2. **Q-Former / Perceiver Resampler**: Uses a fixed number of learnable latent queries to cross-attend to the audio frames, producing a compact fixed-length token sequence (e.g., 64 tokens) regardless of audio duration.

---

### Q5: What are the main challenges when implementing Multimodal RAG?
**Answer:**
1. **Cross-Modal Embeddings**: Joint embedding models (like CLIP) often exhibit "modality gap" where image embeddings cluster in a different sub-manifold than text embeddings, hurting cross-modal retrieval calibration.
2. **Document Parsing**: Extracting tables, charts, and diagrams from PDFs requires specialized visual document understanding models (DocVQA) rather than plain text extraction.
3. **Context Window & Compute Cost**: Passing multiple high-resolution images into a VLM context window consumes thousands of equivalent token slots, increasing latency and TTFT (time-to-first-token).
4. **Resolution Trade-offs**: Fixed patch grids can blur small text inside diagrams or screenshots, necessitating dynamic high-resolution patching (e.g., AnyRes).
