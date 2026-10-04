# Fine-Tuning - Interview Questions

### 1. Why does LoRA work, and how many parameters does it add?
Task-specific weight updates empirically have low intrinsic rank, so $\Delta W=BA$ with small $r$ is expressive enough. It adds $r(d_{in}+d_{out})$ parameters per adapted matrix (e.g. $r=8$, $d=4096$: ~65k vs ~16.8M).

### 2. Why initialise $B$ to zero and $A$ randomly?
So $\Delta W=0$ at step 0 (the model starts exactly as the base) while gradients still flow to both factors (zero-init of both would give zero gradients).

### 3. How do you serve many LoRA adapters?
Keep one frozen base model in memory and swap/batch adapters per request (unmerged), or merge for a single-tenant deployment to eliminate overhead.

### 4. What does QLoRA change vs. LoRA?
The frozen base is stored in 4-bit (NF4, blockwise, double-quantized scales) and dequantized on the fly; adapters and gradients stay in higher precision. Much less memory; some quantization error and speed overhead.

### 5. RAG vs. fine-tuning?
RAG for dynamic, attributable knowledge; fine-tuning for behaviour/format/skills. See [`../rag/`](../rag/).

### 6. Explain DPO and how it differs from PPO-based RLHF.
DPO trains the policy directly on chosen/rejected pairs using a closed-form reparameterisation of the KL-regularised reward objective: no reward model, no sampling loop. PPO RLHF trains a reward model then optimises with on-policy rollouts; more moving parts, more flexibility.

### 7. What is reward hacking and what limits it?
The policy exploits flaws in the proxy reward. Mitigations: KL penalty to the reference, reward-model ensembles, diverse human data, regular re-collection, adversarial evaluation.

### 8. Your fine-tuned model got worse on general tasks. Why and how to fix it?
Catastrophic forgetting/overfitting. Use lower LR/fewer epochs, PEFT, mix in general data, early stopping on a broad eval suite.

### 9. How do you build a good SFT dataset?
Diverse, high-quality, deduplicated, consistent format and template, response-only loss, decontaminated against evals, with a held-out split and human review samples.

### Coding drill
Write a `LoRALinear` whose merged output equals its unmerged output (see the test), then compute trainable parameter counts for a 7-layer toy model.
