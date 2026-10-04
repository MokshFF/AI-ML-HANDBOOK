"""Production RAG Pipeline with Exact Citation Tracking."""
import math
import numpy as np
from typing import List, Dict, Any

class DocumentChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += (chunk_size - overlap)
        return chunks

class MockDenseEmbedder:
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Deterministic semantic hash embedding for reproducibility
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=float)
        for w in words:
            h = hash(w)
            idx = abs(h) % self.dim
            vec[idx] += (1.0 if h > 0 else -1.0)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

class InMemoryVectorStore:
    def __init__(self, embedder: MockDenseEmbedder):
        self.embedder = embedder
        self.docs = []  # List[Dict[str, Any]]
        self.matrix = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            chunks = DocumentChunker.chunk_text(doc["content"])
            for idx, ch in enumerate(chunks):
                vec = self.embedder.embed(ch)
                self.docs.append({
                    "doc_id": doc["id"],
                    "chunk_id": f"{doc['id']}_c{idx}",
                    "text": ch,
                    "title": doc.get("title", "Untitled"),
                    "vector": vec
                })
        self.matrix = np.array([d["vector"] for d in self.docs])

    def query(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed(query_text)
        sims = self.matrix @ q_vec
        ranked_indices = np.argsort(sims)[::-1][:top_k]
        return [self.docs[i] for i in ranked_indices]

class RAGChatbot:
    def __init__(self, vector_store: InMemoryVectorStore):
        self.store = vector_store

    def answer_query(self, query: str) -> Dict[str, Any]:
        retrieved = self.store.query(query, top_k=2)
        citations = [{"source": r["title"], "chunk_id": r["chunk_id"]} for r in retrieved]
        
        # Grounded answer synthesis
        context_snippets = " ".join([r["text"] for r in retrieved])
        response = (
            f"Based on [{retrieved[0]['title']}], the system finds that: "
            f"{retrieved[0]['text'][:100]}... "
            f"Sources cited: {', '.join(c['chunk_id'] for c in citations)}."
        )
        return {
            "query": query,
            "answer": response,
            "citations": citations,
            "retrieved_context": context_snippets
        }

if __name__ == "__main__":
    embedder = MockDenseEmbedder()
    store = InMemoryVectorStore(embedder)
    
    docs = [
        {"id": "doc1", "title": "Kubernetes Deployment Guide", "content": "Deployments manage ReplicaSets which ensure the desired pod count is maintained in production clusters."},
        {"id": "doc2", "title": "Kafka Architecture Overview", "content": "Apache Kafka provides high-throughput distributed commit logs partitioned across broker nodes."}
    ]
    store.add_documents(docs)
    bot = RAGChatbot(store)
    result = bot.answer_query("How do replica sets work?")
    print("RAG Response:", result["answer"])
