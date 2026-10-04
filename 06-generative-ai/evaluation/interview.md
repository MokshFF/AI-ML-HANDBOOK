# Generative AI Evaluation Interview Questions & Answers

### Q1: Why are classical NLP metrics like BLEU and ROUGE often insufficient for evaluating modern LLM generations?
**Answer:**
1. **Surface Form Rigidity**: BLEU and ROUGE evaluate strictly exact n-gram and substring overlaps. A model response that paraphrases using synonyms or reorders clauses will score near 0 despite being semantically identical.
2. **Failure to Detect Hallucinations**: A response can achieve high ROUGE overlap with the source context by repeating words, yet introduce a critical factual falsehood (e.g., negation flips like "did" vs "did not").
3. **Open-Ended Generations**: For creative writing, reasoning explanations, or conversational chat, there is no single "gold reference" text against which to compare.
4. **Modern Alternatives**: LLM-as-a-judge, semantic embedding similarity (BERTScore), claim-level natural language inference (NLI) entailment, and task-specific execution verifiers (e.g., unit test execution for code).

---

### Q2: What is "LLM-as-a-Judge", what are its main failure modes, and how do you mitigate them?
**Answer:**
**LLM-as-a-Judge** uses a strong LLM to evaluate the outputs of other models against rubrics or in pairwise head-to-head battles.
**Major Failure Modes & Mitigations**:
1. **Position Bias**:
   - *Failure*: Judges favor whichever response is presented first (Response A).
   - *Mitigation*: Run two trials swapping presentation order $(A, B)$ and $(B, A)$. Declare a winner only if consistent; otherwise record a tie.
2. **Verbosity Bias**:
   - *Failure*: Models favor longer, ornate responses over concise answers.
   - *Mitigation*: Include explicit negative scoring in rubrics for unnecessary filler text; normalize or match response lengths.
3. **Self-Preference Bias**:
   - *Failure*: A model tends to rate generations from its own architecture or fine-tuning lineage higher.
   - *Mitigation*: Anonymize outputs, use an ensemble of distinct judge model architectures, or calibrate with human gold labels.
4. **Rubric Ambiguity**:
   - *Failure*: Loose criteria lead to high variance.
   - *Mitigation*: Few-shot exemplar scoring and discrete 1-5 scales with descriptive anchor definitions for each score.

---

### Q3: How do you measure hallucination in Retrieval-Augmented Generation (RAG)?
**Answer:**
Hallucination in RAG is evaluated via **Faithfulness / Groundedness**:
1. **Claim Extraction**: Decompose the generated response into atomic factual propositions using sentence splitting or an extraction prompt.
2. **Context Entailment Verification**: For each atomic claim, evaluate whether it is logically entailed by the retrieved context chunks (using an NLI model or cross-encoder).
3. **Metric Calculation**:
   $$\text{Faithfulness} = \frac{\text{Number of Supported Claims}}{\text{Total Claims Extracted}}$$
   $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
4. **Citation Precision/Recall**: Verify that every cited passage actually supports the associated claim, and that no ungrounded claims are made without citations.

---

### Q4: Explain the difference between MRR and nDCG for retrieval evaluation.
**Answer:**
- **MRR (Mean Reciprocal Rank)**:
  - Considers only the position of the **single first relevant item**.
  - Formula: $\text{MRR} = \frac{1}{\text{rank}_1}$.
  - Best for: Known-item search or single-fact question answering where the user only needs one correct answer.
- **nDCG (Normalized Discounted Cumulative Gain)**:
  - Considers **all relevant items** retrieved up to position $k$, accounting for both their degree of relevance (binary or graded) and their position in the ranked list.
  - Applies a logarithmic discount penalty $\frac{1}{\log_2(i + 1)}$ to penalize relevant items placed deeper down.
  - Best for: Multi-document retrieval, search engines, and multi-hop RAG where multiple documents are required to form a complete answer.

---

### Q5: Why is Cohen's Kappa preferred over raw percentage agreement for human evaluation?
**Answer:**
Raw percentage agreement does not account for agreements that occur purely by chance. For example, if two annotators evaluate whether responses contain toxicity, and 95% of all samples are non-toxic, two annotators who guess blindly or default to "non-toxic" would show 90%+ raw agreement.
**Cohen's Kappa ($\kappa$)** corrects for chance:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is observed agreement and $p_e$ is hypothetical chance agreement computed from the marginal distributions. A value of $\kappa = 0$ indicates agreement no better than random guessing.
