# LLM Fundamentals - Interview Questions

### 1. Why does BPE avoid out-of-vocabulary tokens, and what are its downsides?
It keeps single characters/bytes in the vocabulary, so any string can be segmented. Downsides: frequency-driven merges are not linguistically aligned, some languages/scripts become token-expensive, and numbers/rare strings split unpredictably.

### 2. Why is positional information needed, and what are the main schemes?
Attention treats the input as a set. Sinusoidal/learned absolute embeddings add position to inputs; RoPE rotates query/key pairs so scores depend on relative offsets; ALiBi adds a distance-proportional bias to scores. Extrapolation to longer contexts than seen in training differs by scheme.

### 3. Walk through one decoding step with a KV cache. Why is it exact?
For the new token compute $q,k,v$; append $k,v$ to cached keys/values per layer; attend $q$ over all cached keys. Causal masking means earlier positions never depended on later ones, so their keys/values are unchanged and caching reproduces full recomputation exactly (the repo test asserts identical logits).

### 4. Temperature, top-k, top-p: what does each control and when does each fail?
Temperature changes sharpness; top-k fixes the candidate count (too many tokens in confident contexts, too few in flat ones); top-p adapts the count to the distribution. All are heuristics; none fix factual errors.

### 5. Estimate KV-cache memory for a model.
$2\times L\times n_{kv}\times d_{head}\times T\times B\times \text{bytes}$. Grouped-query attention lowers $n_{kv}$; quantizing the cache lowers bytes. See [`../inference-optimization/`](../inference-optimization/).

### 6. What do scaling laws say, and what are their limits?
Loss is predictable from $N$, $D$, compute. Limits: fits are specific to data/architecture, say little about downstream/emergent behaviour, and ignore inference cost, data quality, and data exhaustion.

### 7. Why can a long context window still fail?
Compute/memory growth, positional extrapolation, and attention dilution; evidence shows degraded use of mid-context information. Retrieval and structure often beat blindly stuffing context.

### 8. Pretraining vs. fine-tuning vs. in-context learning?
Pretraining learns general next-token statistics; fine-tuning updates weights for a behaviour/domain; in-context learning conditions behaviour on prompt examples with no weight update.

### Coding drill
Implement `top_p_filter(probs, p)` so it keeps the *smallest* set with cumulative mass $\ge p$ (hint: compare the cumulative mass *before* each token with $p$). See `filter_logits`.
