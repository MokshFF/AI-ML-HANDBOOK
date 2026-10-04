import pytest
import numpy as np
from text_processor import SimpleTokenizer, PorterStemmerMini, BagOfWords, TfidfVectorizerScratch


def test_simple_tokenizer():
    tok = SimpleTokenizer(lowercase=True, remove_stopwords=False)
    tokens = tok.tokenize("Deep Learning and Natural Language Processing, in 2026!")
    assert tokens == ["deep", "learning", "and", "natural", "language", "processing", "in", "2026"]

    tok_stop = SimpleTokenizer(lowercase=True, remove_stopwords=True)
    tokens_no_stop = tok_stop.tokenize("This is a test of the system.")
    assert "is" not in tokens_no_stop
    assert "a" not in tokens_no_stop
    assert "test" in tokens_no_stop and "system" in tokens_no_stop


def test_porter_stemmer():
    stemmer = PorterStemmerMini()
    assert stemmer.stem("running") == "run"
    assert stemmer.stem("cats") == "cat"
    assert stemmer.stem("ponies") == "poni"
    assert stemmer.stem("caresses") == "caress"


def test_bag_of_words():
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log"
    ]
    bow = BagOfWords()
    counts = bow.fit_transform(corpus)
    assert counts.shape == (2, len(bow.vocabulary_))
    # "the" appears twice in doc 0
    the_idx = bow.vocabulary_["the"]
    assert counts[0, the_idx] == 2
    assert counts[1, the_idx] == 2


def test_tfidf_vectorizer():
    corpus = [
        "quantum physics and computing",
        "classical mechanics and physics",
        "quantum entanglement and teleportation"
    ]
    tfidf = TfidfVectorizerScratch(smooth_idf=True)
    X = tfidf.fit_transform(corpus)
    assert X.shape == (3, len(tfidf.vocabulary_))

    # L2 norm of each document vector should be 1.0
    row_norms = np.linalg.norm(X, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)

    # Word "and" appears in all 3 documents, so its IDF should be lower than "teleportation"
    and_idx = tfidf.vocabulary_["and"]
    teleport_idx = tfidf.vocabulary_["teleportation"]
    assert tfidf.idf_[and_idx] < tfidf.idf_[teleport_idx]
