# Text Preprocessing - Curated References & Bibliography

A curated collection of foundational publications, authoritative textbooks, and standard NLP engineering documentation on text preprocessing.

---

## 1. Landmark Research Papers

### 1.1 Stemming & Morphological Rules
- **An Algorithm for Suffix Stripping (Porter Stemmer)**
  - *Author*: Martin F. Porter (Program: Electronic Library and Information Systems, 1980).
  - *Significance*: The canonical rule-based English suffix stripping algorithm widely used in information retrieval.
- **WordNet: An Electronic Lexical Database**
  - *Author*: George A. Miller (MIT Press, 1998).
  - *Significance*: Hand-crafted semantic and morphological network providing ground truth for English lemmatization.

### 1.2 Subword Tokenization Milestones
- **Neural Machine Translation of Rare Words with Subword Units (BPE)**
  - *Authors*: Rico Sennrich, Barry Haddow, Alexandra Birch (ACL, 2016).
  - *Significance*: Introduced Byte-Pair Encoding (BPE) to NLP, solving the open-vocabulary problem in neural language generation.
- **Japanese and Korean Voice Search (WordPiece)**
  - *Authors*: Mike Schuster, Kaisuke Nakajima (ICASSP, 2012).
  - *Significance*: Introduced WordPiece tokenization, later standardized in Google BERT.
- **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (SentencePiece / Unigram)**
  - *Author*: Taku Kudo (ACL, 2018).
  - *Significance*: Formulated the Unigram language model for subword segmentation and language-independent SentencePiece tokenization.

---

## 2. Textbooks & Engineering Documentation

- **Speech and Language Processing (3rd ed. draft, Chapters 2 & 6)** by Daniel Jurafsky and James H. Martin (Stanford University).
- **Introduction to Information Retrieval** by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (Cambridge University Press).
- **Hugging Face Tokenizers Documentation**: [huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)
- **Scikit-Learn Feature Extraction (`CountVectorizer`, `TfidfVectorizer`)**: [scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
