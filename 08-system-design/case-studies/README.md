# Machine Learning System Design: Practical Case Studies

Comprehensive production case studies detailing end-to-end architectures, scale assumptions, data flows, model selections, APIs, failure modes, trade-offs, and cost analyses for modern AI systems.

---

## 1. Case Study Index

Every case study in this directory is structured into 15 standard engineering sections with Mermaid architectural diagrams:

1. [**01. Personalized Recommendation System**](01-recommendation-system.md)  
   *Multi-stage recommendation cascade (Two-Tower retrieval, DLRM fine-ranking, and diversity reranking) at 100k QPS.*

2. [**02. Real-Time Payment Fraud Detection System**](02-fraud-detection.md)  
   *Sub-30ms payment scoring engine combining deterministic rules with compiled GBDT models (Treelite).*

3. [**03. Hybrid Web Search Engine**](03-search-engine.md)  
   *Hybrid search combining BM25 sparse inverted indexes, dense ANN vector search, and cross-encoder neural reranking.*

4. [**04. Enterprise Image Classification Platform**](04-image-classification-platform.md)  
   *High-throughput visual taxonomy classification platform utilizing ConvNeXt/ViT and TensorRT INT8 serving.*

5. [**05. Real-Time Video Object Detection & Tracking**](05-real-time-object-detection.md)  
   *Edge-to-cloud multi-camera perception pipeline with NVIDIA DeepStream, YOLOv8, and ByteTrack.*

6. [**06. Enterprise Knowledge RAG System**](06-rag-system.md)  
   *Role-based access-controlled (RBAC) retrieval-augmented generation engine with verifiable citation attribution.*

7. [**07. Enterprise Customer Support Chatbot**](07-enterprise-chatbot.md)  
   *Stateful, multi-turn autonomous support agent with LangGraph orchestration, function calling, and human escalation.*

8. [**08. Intelligent Document Processing (IDP) Platform**](08-document-intelligence-system.md)  
   *Multimodal document understanding (LayoutLMv3) extracting structured JSON entities from invoices and forms.*

9. [**09. High-Throughput LLM Inference Service**](09-llm-inference-service.md)  
   *Distributed vLLM serving cluster with PagedAttention, continuous batching, and speculative decoding.*

10. [**10. Autonomous AI Agent Platform**](10-ai-agent-platform.md)  
    *Multi-agent software engineering execution platform with Firecracker microVM sandboxes and MCP integrations.*

11. [**11. Multimodal Content Moderation Platform**](11-content-moderation-system.md)  
    *Defense-in-depth safety platform combining perceptual hashing (PDQ/PhotoDNA) with multimodal toxicity classifiers.*

---

## 2. Universal 15-Point System Design Template

Every case study adheres to this rigorous framework:
1. **Requirements**: Problem statement and business objectives.
2. **Functional Requirements**: Core capabilities and user-facing features.
3. **Non-Functional Requirements**: Latency SLAs, availability, throughput, consistency.
4. **Scale Assumptions**: QPS, daily volume, DAU, storage footprints.
5. **Architecture**: End-to-end component topology with Mermaid diagram.
6. **Data Flow**: Step-by-step request and data processing lifecycle.
7. **Model Choice**: Algorithms, architectures, objective functions, and trade-offs.
8. **Storage**: Vector stores, feature stores, document databases, and object lakes.
9. **APIs**: REST / gRPC contract specifications.
10. **Training Pipeline**: Offline continuous training, labeling, validation gates.
11. **Serving Architecture**: Real-time deployment, autoscaling, caching, hardware allocation.
12. **Monitoring**: Telemetry, latency percentiles, data/concept drift tracking.
13. **Failure Modes**: Outage fallbacks, degraded operation policies, disaster recovery.
14. **Trade-Offs**: Architectural compromises (e.g. latency vs accuracy, cost vs speed).
15. **Cost Considerations**: Hardware instance sizing, token budgets, and cost optimizations.

---

## 3. Directory Structure

```
08-system-design/case-studies/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
├── 01-recommendation-system.md
├── 02-fraud-detection.md
├── 03-search-engine.md
├── 04-image-classification-platform.md
├── 05-real-time-object-detection.md
├── 06-rag-system.md
├── 07-enterprise-chatbot.md
├── 08-document-intelligence-system.md
├── 09-llm-inference-service.md
├── 10-ai-agent-platform.md
├── 11-content-moderation-system.md
└── code/
    ├── case_study_sim.py
    └── test_case_study_sim.py
```
