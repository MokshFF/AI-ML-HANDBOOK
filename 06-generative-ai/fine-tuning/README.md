# Fine-Tuning LLMs

> **Last reviewed:** 2026-10. Training recipes, library APIs (e.g. Hugging Face PEFT/TRL) and best practices change often. This module teaches the stable mathematics and mechanisms; consult current docs before running real jobs. Everything here is small, CPU-only and vendor-neutral.

## Learning objectives
Explain and implement transfer learning for LLMs, LoRA, adapters, QLoRA-style quantization, supervised fine-tuning (SFT), and understand the objectives behind RLHF and DPO.

## 1. When to fine-tune
Try in this order: better prompts ([`../prompt-engineering/`](../prompt-engineering/)), retrieval for knowledge ([`../rag/`](../rag/)), then fine-tuning for **behaviour, format, tone, domain skills, or distilling a larger model**. Fine-tuning is a poor way to inject volatile facts. Always evaluate against the untuned baseline ([`../evaluation/`](../evaluation/)).

## 2. Transfer learning and full fine-tuning
Pretrained weights are a strong initialisation. Full fine-tuning updates every parameter: best capacity, but needs memory for weights, gradients and optimiser state (for Adam roughly 16 bytes/parameter in mixed precision, before activations), stores a full copy per task, and risks **catastrophic forgetting**.

## 3. Parameter-efficient fine-tuning (PEFT)
Freeze the base model; train a tiny number of parameters.
- **Adapters** (Houlsby et al., 2019): small bottleneck MLPs inserted in each block, $x+W_{up}\,\sigma(W_{down}x)$ with zero-init so training starts at the base model. Adds inference latency unless fused.
- **Prefix/prompt tuning**: learn virtual tokens or per-layer key/value prefixes.
- **LoRA** (Hu et al., 2021): the weight update is low-rank:
$$W'=W+\frac{\alpha}{r}BA,\quad B\in\mathbb{R}^{d_{out}\times r},\ A\in\mathbb{R}^{r\times d_{in}},\ r\ll\min(d_{in},d_{out})$$
$B=0$ at init so the model starts unchanged. Trainable parameters per layer: $r(d_{in}+d_{out})$ instead of $d_{in}d_{out}$. After training, $BA$ can be **merged** into $W$ for zero inference overhead, or kept separate to hot-swap many adapters on one base model. Key hyperparameters: rank $r$, scaling $\alpha$, target modules (attention projections, often MLPs too), dropout, learning rate (typically higher than full fine-tuning).

## 4. QLoRA
QLoRA (Dettmers et al., 2023) backpropagates through a **frozen 4-bit-quantized** base model into LoRA adapters kept in higher precision. Ingredients: 4-bit NormalFloat (NF4) data type, **blockwise** absmax quantization, **double quantization** of the scale constants, and paged optimizers. It cuts memory enough to fine-tune large models on a single GPU. `peft_lib.QLoRALinear` demonstrates the blockwise principle with NF-*style* quantile levels; it is **not** bit-compatible with NF4 and omits double quantization and fused kernels. Quantization adds error; always evaluate the final model.

## 5. Supervised fine-tuning (instruction tuning)
SFT minimises next-token cross-entropy on (prompt, response) pairs, usually **only on response tokens** (prompt labels set to `-100`). Data quality dominates quantity: deduplicate, filter, balance, hold out an eval split, and use the model's chat template consistently. Watch for overfitting and for training on leaked test data.

## 6. Preference optimisation: RLHF and DPO (concepts)
- **RLHF** (Christiano et al., 2017; Ouyang et al., 2022): (1) SFT, (2) train a **reward model** from human comparisons with the Bradley-Terry loss $-\log\sigma(r_c-r_r)$, (3) optimise the policy (e.g. PPO) to maximise reward with a **KL penalty** to the reference model, $r-\beta\,\mathrm{KL}(\pi\|\pi_{ref})$, to limit reward hacking and drift.
- **DPO** (Rafailov et al., 2023) reparameterises that objective so the policy is trained directly on preference pairs:
$$\mathcal{L}_{DPO}=-\log\sigma\Big(\beta\big[(\log\pi_\theta(y_c|x)-\log\pi_\theta(y_r|x))-(\log\pi_{ref}(y_c|x)-\log\pi_{ref}(y_r|x))\big]\Big)$$
Simpler and more stable than PPO-based RLHF, but still depends on preference-data quality and $\beta$. Related methods (IPO, KTO, ORPO, GRPO and others) are active research; verify against the literature.
Safety implications are covered in [`../safety-alignment/`](../safety-alignment/).

## Common mistakes
- Fine-tuning to add facts that change weekly.
- Tuning hyperparameters on the test set.
- Wrong chat template or training on prompt tokens unintentionally.
- Merging LoRA into a quantized base without checking numerics.
- Ignoring regressions on general capabilities and safety after tuning.

## Code and notebook
- [`code/peft_lib.py`](code/peft_lib.py): `LoRALinear`, `apply_lora`/`merge_lora`, `BottleneckAdapter`, blockwise quantization + `QLoRALinear`, SFT masking, `dpo_loss`, reward/KL/PPO helpers.
- [`code/test_peft_lib.py`](code/test_peft_lib.py): 11 tests (merge exactness, frozen base, quantization error ordering, DPO $=\ln 2$ at the reference, ...).
- [`notebook.ipynb`](notebook.ipynb): runnable demos.
