# AI, ML & GenAI Engineering Handbook (`ai-ml-handbook`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Validation Gate](https://img.shields.io/badge/Validation-100%25%20Passing-brightgreen.svg)](./scripts/validate_repo.py)
[![Unit Tests](https://img.shields.io/badge/Tests-76%20Passing-success.svg)](./PROJECT_STATUS.md)
[![Project Status](https://img.shields.io/badge/Status-Complete-success.svg)](./PROJECT_STATUS.md)

A comprehensive, engineering-first knowledge base, reference architecture, and curriculum designed for software engineers, machine learning practitioners, and researchers navigating classical ML, deep neural networks, distributed systems, and Generative AI.

---

## Table of Contents
1. [Project Vision & Philosophy](#1-project-vision--philosophy)
2. [Repository Statistics](#2-repository-statistics)
3. [Repository Architecture & Module Index](#3-repository-architecture--module-index)
4. [Standard Topic & Project Schemas](#4-standard-topic--project-schemas)
5. [Learning Pathways](#5-learning-pathways)
6. [Installation & Setup](#6-installation--setup)
7. [Testing & Validation](#7-testing--validation)
8. [Current Engineering Status & Roadmap](#8-current-engineering-status--roadmap)
9. [Contribution Guidelines](#9-contribution-guidelines)
10. [License & Citations](#10-license--citations)

---

## 1. Project Vision & Philosophy

Modern AI education often bifurcates into high-level, code-free conceptual summaries or scattered, unmaintained notebooks that break across library updates. The `ai-ml-handbook` bridges this divide through four guiding engineering tenets:

- **Mathematical Rigor with Code**: Every formula is matched with a clean, vectorized Python/PyTorch implementation.
- **Production-Oriented Thinking**: Focus on system trade-offs, inference latency, memory footprint, and serving constraints.
- **Deterministic Standard Schema**: Every module adheres to a strict, predictable folder structure (Theory, Lab, Standalone Code, Interview Drills, Literature References).
- **Lightweight & Reproducible**: Avoids committing multi-gigabyte checkpoints or proprietary cloud dependencies; uses deterministic datasets that execute reliably on modern multi-core laptops and workstations.

---

## 2. Repository Statistics

Empirical codebase metrics verified by continuous integration gates:

| Metric | Verified Count | Scope & Details |
| :--- | :--- | :--- |
| **Major Modules** | **13** | Modules `00` through `12` covering prerequisites to resources |
| **Curriculum Topics** | **55** | Individual deep-dive topics with 5-part architecture schemas |
| **End-to-End Projects** | **16** | Fully runnable projects across Beginner, Intermediate, Advanced, Research, GenAI, MLOps |
| **Jupyter Notebooks** | **71** | Verified interactive notebooks executed headlessly in CI |
| **Automated Unit Tests** | **76** | Pytest suites verifying algorithm outputs, APIs, and pipelines |
| **System Design Case Studies** | **11** | Full-scale 15-section industrial system designs with Mermaid diagrams |
| **Cheat Sheets** | **10** | High-yield reference tables (ML, DL, NLP, Transformers, RAG, LLMOps, System Design) |
| **Seminal Research Papers** | **22** | Annotated landmark papers across 11 domains with 11-point architectural breakdowns |
| **Source Python Files** | **169** | Modular production code, tests, and utility tools |
| **Documentation Guides** | **270** | Curated markdown files, theoretical guides, and interview prep |

---

## 3. Repository Architecture & Module Index

```
ai-ml-handbook/
├── 00-prerequisites/       # Python, Linear Algebra, Calculus, Probability, Data Structures
├── 01-machine-learning/     # Classical ML: Supervised, Unsupervised, Ensembles, Features, RL
├── 02-deep-learning/        # Neural Nets: Fundamentals, CNN, RNN/LSTM, Transformers, GNN
├── 03-nlp/                  # Computational Linguistics, Embeddings, Language Models
├── 04-computer-vision/      # Image Processing, Detection, Segmentation, Vision Transformers
├── 05-speech-audio/         # Signal Processing, ASR, TTS, Audio Classification
├── 06-generative-ai/        # LLMs, Prompting, RAG, Fine-Tuning, Agents, Optimization
├── 07-mlops/                # Tracking, Data Versioning, Serving, CI/CD, LLMOps
├── 08-system-design/        # Real-World Architectures, Pipelines, Case Studies, Scaling
├── 09-projects/             # 16 End-to-End Projects (Beginner, Intermediate, Advanced, GenAI, MLOps)
├── 10-interview-prep/       # Screening Q&A, 10 High-Yield Cheat Sheets, Coding Solutions
├── 11-research-papers/      # Landmark Research Annotated Compendiums across 11 AI Domains
├── 12-resources/            # Authoritative Bibliography: Books, Courses, Tools, Benchmarks
├── _templates/              # Standard Reusable Topic Templates
├── scripts/                 # Automated Repository Integrity & Validation Scripts
└── .github/                 # Automated CI Workflows & Quality Gates
```

### Complete Module Index

| Module | Title | Scope & Key Concepts |
| :--- | :--- | :--- |
| [`00-prerequisites/`](./00-prerequisites/) | **Prerequisites** | Linear algebra, multivariate calculus, optimization, probability distributions, data structures |
| [`01-machine-learning/`](./01-machine-learning/) | **Classical ML** | Supervised (Ridge, Lasso, Trees, SVM, GBDT), Unsupervised (K-Means, GMM, PCA, DBSCAN), Ensembles |
| [`02-deep-learning/`](./02-deep-learning/) | **Deep Learning** | Autograd, backprop, optimizers (AdamW), CNNs, RNNs, self-attention, normalization, regularization |
| [`03-nlp/`](./03-nlp/) | **NLP** | Tokenization (BPE/WordPiece), word vectors, sequence models, transformer encoders/decoders |
| [`04-computer-vision/`](./04-computer-vision/) | **Computer Vision** | Filtering, transfer learning, object detection (YOLO/DETR), semantic segmentation, ViT |
| [`05-speech-audio/`](./05-speech-audio/) | **Speech & Audio** | Spectrograms, MFCCs, acoustic modeling, Whisper ASR, TTS synthesis |
| [`06-generative-ai/`](./06-generative-ai/) | **Generative AI** | Autoregressive decoding, Prompting, RAG pipelines, PEFT/LoRA, autonomous agents |
| [`07-mlops/`](./07-mlops/) | **MLOps** | Experiment tracking, model registry, inference servers (FastAPI/Triton), drift monitoring (KS/PSI) |
| [`08-system-design/`](./08-system-design/) | **ML System Design** | Feature stores, streaming architectures, distributed scaling, 11 production case studies |
| [`09-projects/`](./09-projects/) | **Projects** | 16 runnable projects across Beginner, Intermediate, Advanced, Research, GenAI, MLOps |
| [`10-interview-prep/`](./10-interview-prep/) | **Interview Prep** | Theory compendiums, 10 high-yield cheat sheets, and practical coding solutions |
| [`11-research-papers/`](./11-research-papers/) | **Research Papers** | 22 seminal landmark papers annotated across 11 domains with 11-point architectural breakdowns |
| [`12-resources/`](./12-resources/) | **Curated Resources** | Authoritative books, academic courses, official documentation, and benchmark datasets |

---

## 4. Standard Topic & Project Schemas

### Standard Topic Schema (`00` to `08`, `10`)
Every learning topic adheres to a deterministic 5-component structure:
- `README.md`: Concept breakdown, mathematical formulas, and matrix comparison.
- `notebook.ipynb`: Verified, runnable interactive Jupyter lab.
- `code/`: Clean, modular Python scripts and helper utilities.
- `interview.md`: Technical screening questions and trade-off analysis.
- `references.md`: Authoritative research citations and documentation.

### Standard Project Schema (`09-projects/`)
Every project adheres strictly to an industry microservice structure:
- `README.md`: 12 standard engineering sections (Problem, Motivation, Dataset, Architecture, Pipeline, Technologies, Installation, Usage, Evaluation, Results, Limitations, Future Work).
- `src/`: Production Python implementation.
- `notebooks/`: Interactive walkthrough notebook (`notebook.ipynb`).
- `tests/`: Automated unit tests (`test_*.py` with `conftest.py`).
- `requirements.txt`: Minimal reproducible dependencies.
- `.env.example`: Configuration templates.
- `Dockerfile`: Multi-stage container specification where applicable.

---

## 5. Learning Pathways

Depending on your background and target career objectives, follow our curated pathways in [`ROADMAP.md`](./ROADMAP.md):

1. **Beginner → Machine Learning Engineer**:
   - Foundations ([`00-prerequisites`](./00-prerequisites/)) $\to$ Classical ML ([`01-machine-learning`](./01-machine-learning/)) $\to$ MLOps ([`07-mlops`](./07-mlops/)) $\to$ Beginner Projects ([`09-projects/beginner`](./09-projects/beginner/)) $\to$ Coding Drills ([`10-interview-prep/coding-questions`](./10-interview-prep/coding-questions/)).
2. **Beginner → Deep Learning Engineer**:
   - Foundations $\to$ Deep Learning ([`02-deep-learning`](./02-deep-learning/)) $\to$ Computer Vision ([`04-computer-vision`](./04-computer-vision/)) $\to$ Intermediate Projects ([`09-projects/intermediate`](./09-projects/intermediate/)) $\to$ DL Theory QA ([`10-interview-prep/deep-learning-questions.md`](./10-interview-prep/deep-learning-questions.md)).
3. **ML Engineer → LLM Engineer**:
   - Transformers ([`02-deep-learning/attention-transformers`](./02-deep-learning/attention-transformers/)) $\to$ GenAI ([`06-generative-ai`](./06-generative-ai/)) $\to$ LLMOps ([`07-mlops/llmops`](./07-mlops/llmops/)) $\to$ GenAI Projects ([`09-projects/genai`](./09-projects/genai/)) $\to$ GenAI Q&A ([`10-interview-prep/genai-llm-questions.md`](./10-interview-prep/genai-llm-questions.md)).
4. **Beginner → Generative AI Engineer**:
   - Python for ML $\to$ GenAI Fundamentals $\to$ RAG $\to$ Agents $\to$ Multimodal $\to$ GenAI Projects.
5. **AI Research Path**:
   - Mathematical Foundations $\to$ Seminal Papers ([`11-research-papers`](./11-research-papers/)) $\to$ Architecture Re-implementation ([`09-projects/research`](./09-projects/research/)).
6. **ML System Design Path**:
   - System Design Patterns ([`08-system-design`](./08-system-design/)) $\to$ Production Case Studies $\to$ System Design Blueprints ([`10-interview-prep/system-design-questions.md`](./10-interview-prep/system-design-questions.md)).

---

## 6. Installation & Setup

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/avars/ai-ml-handbook.git
cd ai-ml-handbook

# 2. Create isolated virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Testing & Validation

Execute the repository quality and integrity test suite locally:

```bash
# 1. Validate complete repository architecture, topic files, and relative links
python scripts/validate_repo.py

# 2. Run unit test suite across all modules (76 tests)
python -m pytest 07-mlops 08-system-design 09-projects 10-interview-prep -v
```

---

## 8. Current Engineering Status & Roadmap

- **Completed Modules**: All 13 major modules (`00` through `12`) are complete, validated, and tested.
- **Detailed Audit & Status Report**: Read [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) for transparent documentation of completed components, intentional architectural boundaries, and known limitations.
- **Full Roadmap & Pathway Breakdown**: Read [`ROADMAP.md`](./ROADMAP.md) for sequence guides and curriculum blueprints.

---

## 9. Contribution Guidelines

Contributions, corrections, and new project modules are welcomed! Before opening a pull request:
1. Review [`CONTRIBUTING.md`](./CONTRIBUTING.md) for coding style, topic schemas, and PR templates.
2. Ensure `python scripts/validate_repo.py` passes with zero errors.
3. Ensure all unit tests pass with `pytest`.

---

## 10. License & Citations

Distributed under the terms of the [MIT License](./LICENSE). When citing this repository in academic or professional projects:
```bibtex
@misc{aimlhandbook2026,
  author = {AI-ML Handbook Contributors},
  title = {AI, ML & GenAI Engineering Handbook: Production Architectures, Foundations, and Projects},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\\url{https://github.com/avars/ai-ml-handbook}}
}
```
