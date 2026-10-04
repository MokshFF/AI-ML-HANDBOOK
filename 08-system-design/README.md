# Machine Learning System Design (`08-system-design`)

## Overview
Architectural frameworks, scale constraints, distributed systems, and real-world industrial case studies.

## Subtopics & Navigation
| Directory | Topic | Scope |
| :--- | :--- | :--- |
| [`architecture-patterns/`](./architecture-patterns/) | **Architecture Patterns** | Online vs batch inference, lambda/kappa architectures, feature stores, and caching layers. |
| [`data-pipelines/`](./data-pipelines/) | **Data Pipelines** | Stream processing, batch ETL, message queues (Kafka), and scalable data validation. |
| [`case-studies/`](./case-studies/) | **Case Studies** | 11 comprehensive production case studies (RecSys, Fraud, Search, Vision, RAG, Chatbot, IDP, LLM Serving, Agents, Moderation). |
| [`scaling-infrastructure/`](./scaling-infrastructure/) | **Scaling Infrastructure** | Distributed training, multi-GPU orchestration, model parallelism, tensor parallelism, and horizontal auto-scaling. |

### Featured Case Studies in [`case-studies/`](./case-studies/)
1. [Recommendation System](./case-studies/01-recommendation-system.md) - Two-Tower retrieval and DLRM ranking at 100k QPS
2. [Payment Fraud Detection](./case-studies/02-fraud-detection.md) - Sub-30ms fraud scoring with compiled GBDTs
3. [Hybrid Search Engine](./case-studies/03-search-engine.md) - BM25 lexical + dense ANN + cross-encoder reranking
4. [Image Classification Platform](./case-studies/04-image-classification-platform.md) - High-throughput ConvNeXt/ViT with TensorRT
5. [Real-Time Object Detection](./case-studies/05-real-time-object-detection.md) - DeepStream, YOLOv8, and ByteTrack video perception
6. [Enterprise RAG System](./case-studies/06-rag-system.md) - RBAC-filtered retrieval with verifiable citation attribution
7. [Enterprise Chatbot](./case-studies/07-enterprise-chatbot.md) - LangGraph orchestration, function calling, and escalation
8. [Document Intelligence (IDP)](./case-studies/08-document-intelligence-system.md) - LayoutLMv3 multimodal entity extraction
9. [LLM Inference Service](./case-studies/09-llm-inference-service.md) - vLLM, PagedAttention, continuous batching, speculative decoding
10. [Autonomous AI Agent Platform](./case-studies/10-ai-agent-platform.md) - Firecracker microVM sandboxes, multi-agent workflows, MCP
11. [Content Moderation System](./case-studies/11-content-moderation-system.md) - Perceptual hashing (PDQ/PhotoDNA) + multimodal toxicity filters

## Standard Directory Schema
Every topic directory in this module follows our standard five-component structure:
- `README.md` - Module introduction, learning objectives, and concept matrix
- `notebook.ipynb` - Reproducible, runnable interactive notebook
- `code/` - Clean, modular Python scripts and helper utilities
- `interview.md` - Technical screening questions, edge cases, and design discussions
- `references.md` - Research papers, textbooks, and documentation

## Prerequisites
Before beginning this module, review:
- Foundational math and coding prerequisites in [`../00-prerequisites/`](../00-prerequisites/)
- The end-to-end learning pathways defined in [`../ROADMAP.md`](../ROADMAP.md)
