# Generative AI & LLMs (`06-generative-ai`)

## Overview
Large language models, prompting, retrieval-augmented generation (RAG), parameter-efficient fine-tuning, and autonomous agents.

## Subtopics & Navigation
| Directory | Topic | Scope |
| :--- | :--- | :--- |
| [`llm-fundamentals/`](./llm-fundamentals/) | **LLM Fundamentals** | Autoregressive generation, sampling methods (temperature, top-p, top-k, repetition penalty), and scaling laws. |
| [`prompt-engineering/`](./prompt-engineering/) | **Prompt Engineering** | System prompts, zero-shot/few-shot prompting, Chain-of-Thought (CoT), Tree of Thoughts, and structured outputs. |
| [`rag/`](./rag/) | **RAG** | Retrieval-Augmented Generation, chunking strategies, vector databases, hybrid search, and re-ranking pipelines. |
| [`fine-tuning/`](./fine-tuning/) | **Fine Tuning** | Parameter-Efficient Fine-Tuning (PEFT, LoRA, QLoRA), instruction tuning, dataset preparation, and memory optimization. |
| [`agents/`](./agents/) | **Agents** | ReAct framework, tool calling, memory architectures, multi-agent collaboration, and execution environments. |
| [`evaluation/`](./evaluation/) | **Evaluation** | LLM-as-a-judge, benchmark suites (MMLU, GSM8K, HumanEval), automated evaluation metrics, and hallucination detection. |
| [`multimodal/`](./multimodal/) | **Multimodal** | Vision-language models (CLIP, LLaVA), audio-text models, interleaved multimodal inputs, and cross-attention. |
| [`safety-alignment/`](./safety-alignment/) | **Safety Alignment** | RLHF, DPO, constitutional AI, prompt injection defenses, guardrails, and toxicity filtering. |
| [`inference-optimization/`](./inference-optimization/) | **Inference Optimization** | KV-cache optimization, quantization (INT8, INT4, AWQ, GPTQ), vLLM, speculative decoding, and batching. |

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
