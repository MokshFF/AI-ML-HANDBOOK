"""
Text Preprocessing & Classical NLP Vectorizers from Scratch.
Implements:
1. Regex & Rule-based Tokenizer.
2. Lightweight Suffix Stemmer (Porter-style rules).
3. Bag-of-Words (CountVectorizer).
4. TF-IDF Vectorizer with smooth IDF and L2 normalization.
"""

from __future__ import annotations
import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set


class SimpleTokenizer:
    """
    Rule-based and regex text tokenizer with lowercasing, punctuation stripping,
    and optional stopword removal.
    """
    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "is", "it",
        "of", "for", "with", "as", "by", "that", "this", "be", "are", "was"
    }

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = False):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        # Extract word alphanumeric sequences
        tokens = re.findall(r"\b\w+\b", text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens


class PorterStemmerMini:
    """
    Lightweight rule-based suffix stemmer implementing fundamental Porter Phase 1 rules:
    - sses -> ss
    - ies -> i
    - ss -> ss
    - s -> ""
    - eed -> ee (if stem length > 1)
    - ed / ing -> stripped
    """
    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif not word.endswith("ss") and word.endswith("s"):
            word = word[:-1]

        # Step 1b: eed, ed, ing
        if word.endswith("eed"):
            if len(word[:-3]) > 1:
                word = word[:-1]
        elif word.endswith("ed"):
            stem = word[:-2]
            if len(stem) > 1 and any(c in "aeiou" for c in stem):
                word = stem
                if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                    word = word[:-1]
        elif word.endswith("ing"):
            stem = word[:-3]
            if len(stem) > 1 and any(c in "aeiou" for c in stem):
                word = stem
                if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                    word = word[:-1]

        # Step 1c: y -> i
        if word.endswith("y") and len(word) > 2 and any(c in "aeiou" for c in word[:-1]):
            word = word[:-1] + "i"

        return word


class BagOfWords:
    """
    CountVectorizer from scratch.
    Constructs vocabulary and transforms text documents into sparse/dense count vectors.
    """
    def __init__(self, max_features: Optional[int] = None):
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.tokenizer = SimpleTokenizer(lowercase=True, remove_stopwords=False)

    def fit(self, documents: List[str]) -> BagOfWords:
        word_counts: Dict[str, int] = {}
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1

        # Sort by frequency descending, then alphabetically
        sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        if self.max_features is not None:
            sorted_words = sorted_words[:self.max_features]

        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted_words)}
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=np.int32)
        for row_idx, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            for t in tokens:
                if t in self.vocabulary_:
                    col_idx = self.vocabulary_[t]
                    matrix[row_idx, col_idx] += 1
        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


class TfidfVectorizerScratch:
    """
    TF-IDF Vectorizer from scratch matching scikit-learn standard smooth formulation:
    TF(t, d) = count(t, d)
    IDF(t) = log((1 + N) / (1 + df(t))) + 1
    TF-IDF = TF * IDF
    Normalized using L2 Euclidean vector norm: v / ||v||_2
    """
    def __init__(self, smooth_idf: bool = True, sublinear_tf: bool = False):
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self.tokenizer = SimpleTokenizer(lowercase=True, remove_stopwords=False)

    def fit(self, documents: List[str]) -> TfidfVectorizerScratch:
        N = len(documents)
        doc_tokens = [set(self.tokenizer.tokenize(d)) for d in documents]

        # Vocabulary
        all_words = sorted(list(set(t for dt in doc_tokens for t in dt)))
        self.vocabulary_ = {w: i for i, w in enumerate(all_words)}
        V = len(self.vocabulary_)

        # Document frequencies
        df = np.zeros(V, dtype=np.float64)
        for dt in doc_tokens:
            for w in dt:
                if w in self.vocabulary_:
                    df[self.vocabulary_[w]] += 1.0

        if self.smooth_idf:
            self.idf_ = np.log((1.0 + N) / (1.0 + df)) + 1.0
        else:
            self.idf_ = np.log(N / df) + 1.0

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        N = len(documents)
        V = len(self.vocabulary_)
        tf = np.zeros((N, V), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            for t in tokens:
                if t in self.vocabulary_:
                    tf[i, self.vocabulary_[t]] += 1.0

        if self.sublinear_tf:
            # 1 + log(TF) if TF > 0
            tf = np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1e-12)), 0.0)

        # TF-IDF
        tfidf = tf * self.idf_

        # L2 Normalization per document row
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return tfidf / norms

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)
