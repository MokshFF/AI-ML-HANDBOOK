# Cheat Sheet: Natural Language Processing

| Concept | Key Mechanics | Use Cases | Pitfalls / Edge Cases |
| :--- | :--- | :--- | :--- |
| **BPE Tokenization** | Greedy frequency merge of adjacent bytes | GPT, LLaMA tokenizers | Whitespace sensitivity; arithmetic token fragmentation |
| **TF-IDF** | $\text{TF}(t, d) \times \log((1+N)/(1+\text{DF}(t)))$ | Baseline text classification, BM25 retrieval | Completely ignores word ordering and semantics |
| **Word2Vec (Skip-gram)** | Predict context words given center word via negative sampling | Static lexical representations | Polysemy (e.g. "bank" financial vs river bank) |
| **BERT (Masked LM)** | Bidirectional encoder trained on $15\%$ masked tokens | Classification, NER, extractive QA | Cannot generate text autoregressively |
| **ROUGE (1/2/L)** | N-gram overlap between candidate and reference | Summarization evaluation | Rewards exact surface matches, not factual accuracy |
| **BLEU** | Modified n-gram precision with brevity penalty | Machine translation | Weak correlation with human fluency on single sentences |
| **Perplexity (PPL)** | $\exp\left(-\frac{1}{N} \sum \log P(w_t \mid w_{<t})\right)$ | Language model quality comparison | Dependent on tokenizer vocabulary size; cannot cross-compare |
