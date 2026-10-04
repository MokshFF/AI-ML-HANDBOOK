import math
import numpy as np
import pytest
from rag_pipeline import *

DOC = """# Billing
## Refunds
Customers may request a refund within 30 days of purchase. Refunds are issued to the original payment method.
## Invoices
Invoices are emailed monthly. Error code ERR-4012 means the invoice address is missing.
# Security
## Passwords
Passwords must be rotated every 90 days and must contain at least twelve characters.
"""


def test_parse_markdown_keeps_heading_path():
    secs = dict(parse_markdown(DOC))
    assert "Billing > Refunds" in secs and "Security > Passwords" in secs
    assert "refund" in secs["Billing > Refunds"].lower()


def test_chunk_words_overlap_and_validation():
    text = " ".join(f"w{i}" for i in range(25))
    ch = chunk_words(text, size=10, overlap=3)
    assert ch[0].split()[-3:] == ch[1].split()[:3]
    assert " ".join(ch).count("w24") >= 1 and ch[-1].endswith("w24")
    with pytest.raises(ValueError):
        chunk_words(text, size=5, overlap=5)
    assert chunk_words("", 5, 1) == []


def test_chunk_sentences_never_splits_sentence():
    text = "One two three. Four five six seven. Eight nine. Ten eleven twelve thirteen."
    ch = chunk_sentences(text, max_words=7, overlap_sentences=1)
    assert all(c.endswith(".") for c in ch)
    assert ch[1].startswith(ch[0].split(". ")[-1][:4])  # last sentence repeated


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = e.embed("refund policy"), e.embed("refund policy")
    assert np.allclose(a, b) and math.isclose(float(np.linalg.norm(a)), 1.0, rel_tol=1e-5)
    assert float(e.embed("refund policy") @ e.embed("refund policy window")) > float(e.embed("refund policy") @ e.embed("quantum chromodynamics"))


def _setup():
    chunks = ingest_markdown("handbook", DOC, size=30, overlap=5, extra_meta={"dept": "x"})
    store = VectorStore(HashingEmbedder(512))
    store.add(chunks)
    return chunks, store, BM25(chunks)


def test_metadata_filtering():
    chunks, store, bm = _setup()
    hits = store.search("rotated characters", k=3, where={"section": "Security > Passwords"})
    assert hits and all(c.metadata["section"] == "Security > Passwords" for c, _ in hits)
    assert store.search("refund", k=3, where={"source": "nope"}) == []
    assert bm.search("refund", k=3, where={"section": ["Billing > Refunds"]})


def test_bm25_exact_identifier_and_hybrid():
    chunks, store, bm = _setup()
    top = bm.search("ERR-4012", k=1)[0][0]
    assert "ERR-4012" in top.text
    hyb = hybrid_search(store, bm, "what does ERR-4012 mean", k=2)
    assert any("ERR-4012" in c.text for c in hyb)


def test_rrf_math():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60))
    assert math.isclose(fused["a"], 1 / 61 + 1 / 62)
    assert fused["a"] > fused["b"] and fused["c"] > fused["b"]


def test_mmr_promotes_diversity():
    q = np.array([1.0, 0.0])
    cands = np.array([[1.0, 0.0], [0.999, 0.045], [0.6, 0.8]])
    cands = cands / np.linalg.norm(cands, axis=1, keepdims=True)
    assert mmr(q, cands, 2, lam=1.0) == [0, 1]       # pure relevance
    assert mmr(q, cands, 2, lam=0.3)[1] == 2         # diversity wins


def test_rerank_multi_query_and_parent_expansion():
    chunks, store, bm = _setup()
    r = rerank("password rotated 90 days", chunks)
    assert "Passwords" in r[0].metadata["section"]
    mq = multi_query_search(["refund window", "money back policy refund"], lambda q: bm.search(q, 5) and [c for c, _ in bm.search(q, 5)], k=3)
    assert mq and "refund" in mq[0].text.lower()
    parents = {"handbook#s0": "full parent text"}
    out = expand_to_parents([c for c in chunks if c.metadata["parent_id"] == "handbook#s0"] * 2, parents)
    assert out == [("handbook#s0", "full parent text")]


def test_context_budget_and_citations():
    chunks, _, _ = _setup()
    ctx, used = build_context(chunks, max_words=40)
    assert ctx.startswith("[1] (source: handbook)") and sum(len(c.text.split()) for c in used) <= 40
    assert "Context:" in rag_prompt("q?", ctx)
    ans = "Refunds take 30 days [1]. Policy is strict [5][2]."
    assert extract_citations(ans) == {1, 2, 5}
    assert invalid_citations(ans, n_sources=2) == {5}


def test_groundedness_proxy():
    src = ["Customers may request a refund within 30 days of purchase."]
    assert groundedness("Customers may request a refund within 30 days.", src) == 1.0
    assert groundedness("Dragons guard the premium tier vaults.", src) == 0.0
    assert groundedness("", src) == 0.0


def test_retrieval_metrics():
    ret = ["d3", "d1", "d9", "d2"]
    rel = {"d1", "d2"}
    assert recall_at_k(ret, rel, 2) == 0.5 and recall_at_k(ret, rel, 4) == 1.0
    assert reciprocal_rank(ret, rel) == 0.5 and reciprocal_rank(["x"], rel) == 0.0
    gains = {"d1": 3, "d2": 1}
    assert math.isclose(ndcg_at_k(["d1", "d2"], gains, 2), 1.0)
    assert ndcg_at_k(["d2", "d1"], gains, 2) < 1.0
