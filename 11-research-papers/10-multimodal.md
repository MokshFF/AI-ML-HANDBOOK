# Seminal Research Papers: Multimodal Foundation Models

Cross-modal representation learning, dual-encoders, and vision-language instruction tuning.

---

## 1. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- **Title**: Learning Transferable Visual Models From Natural Language Supervision
- **Authors**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, et al.
- **Year**: 2021
- **Link**: https://arxiv.org/abs/2103.00020
- **Problem**: Computer vision models were constrained to fixed discrete label sets (ImageNet 1k), unable to generalize to open-vocabulary concepts.
- **Main Idea**: Pre-train an image encoder and text encoder jointly on 400 million internet image-text pairs using symmetric InfoNCE contrastive loss.
- **Key Contribution**: Enabled zero-shot image classification competitive with supervised models simply by querying text prompt embeddings.
- **Important Architecture/Math**:
  $$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}} \right), \quad \mathcal{L}_{i\to t} = -\sum_{i} \log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_j \exp(I_i \cdot T_j / \tau)}$$
- **Why It Matters**: Created the standard multimodal embedding foundation for text-to-image diffusion, visual search, and zero-shot perception.
- **Prerequisites**: Contrastive representation learning, InfoNCE loss, Vision Transformers.
- **Suggested Follow-up Papers**: *Visual Instruction Tuning (LLaVA)* (Liu et al., 2023); *BLIP-2* (Li et al., 2023).
