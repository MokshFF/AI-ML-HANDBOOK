"""Tests for RAG Chatbot Pipeline."""
from rag_pipeline import DocumentChunker, MockDenseEmbedder, InMemoryVectorStore, RAGChatbot

def test_chunker():
    text = "word " * 300
    chunks = DocumentChunker.chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1

def test_rag_pipeline():
    embedder = MockDenseEmbedder(dim=16)
    store = InMemoryVectorStore(embedder)
    store.add_documents([
        {"id": "d1", "title": "MLOps", "content": "Model registry and monitoring ensure reliable ML deployments."},
        {"id": "d2", "title": "Security", "content": "Role based access control limits unauthorized data exposure."}
    ])
    bot = RAGChatbot(store)
    res = bot.answer_query("How does model registry help?")
    assert len(res["citations"]) == 2
    assert "MLOps" in res["answer"] or "Security" in res["answer"]
