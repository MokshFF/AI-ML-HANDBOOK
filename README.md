# AI, ML & GenAI Engineering Handbook (`ai-ml-handbook`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Repository Structure Validation](https://github.com/avars/ai-ml-handbook/actions/workflows/repo-validation.yml/badge.svg)](./.github/workflows/repo-validation.yml)
[![Code Quality](https://github.com/avars/ai-ml-handbook/actions/workflows/python-lint.yml/badge.svg)](./.github/workflows/python-lint.yml)

A structured, engineering-first knowledge base, reference architecture, and curriculum designed for software engineers, machine learning practitioners, and researchers navigating modern AI, Deep Learning, and Generative AI systems.

---

## 1. Project Vision

Modern AI education often bifurcates into either high-level, code-free conceptual summaries or scattered, unmaintained tutorial notebooks. The goal of `ai-ml-handbook` is to bridge this gap by delivering:

- **Mathematical Foundations with Code**: Every formula accompanied by clean, vectorized implementations.
- **Production-Oriented Thinking**: Focus on system trade-offs, inference latency, memory footprint, and serving constraints.
- **Standardized Architecture**: Every topic follows a deterministic five-component structure (Theory, Lab, Standalone Code, Interview Drills, Literature References).
- **Zero-Bloat Engineering**: Minimal external dependencies, reproducible execution environments, and rigorous quality assurance.

---

## 2. Table of Contents & Repository Architecture

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
├── 09-projects/             # Hands-on Guided Projects, Microservices, and Capstones
├── 10-interview-prep/       # Algorithm Implementations, Cheatsheets, System Design Mocks
├── 11-research-papers/      # Reading Guides, Foundational Papers, Landmark Breakthroughs
├── 12-resources/            # Curated Bibliography, Datasets, Frameworks
├── _templates/              # Standard Reusable Topic Templates
└── .github/                 # Automated CI Workflows & Issue/PR Templates
```

### Module Index

| Module | Title | Scope |
| :--- | :--- | :--- |
| [`00-prerequisites/`](./00-prerequisites/) | **Prerequisites** | Linear algebra, multivariate calculus, probability & statistics, Python vectorization |
| [`01-machine-learning/`](./01-machine-learning/) | **Classical ML** | Supervised, unsupervised, tree ensembles (XGBoost/LightGBM), time-series, recommender systems |
| [`02-deep-learning/`](./02-deep-learning/) | **Deep Learning** | Backprop from scratch, CNNs, RNNs, self-attention, normalization, regularization |
| [`03-nlp/`](./03-nlp/) | **Natural Language Processing** | Tokenization (BPE/SentencePiece), word vectors, seq2seq, transformer encoders/decoders |
| [`04-computer-vision/`](./04-computer-vision/) | **Computer Vision** | Classical image filtering, transfer learning, object detection (YOLO/DETR), segmentation, ViT |
| [`05-speech-audio/`](./05-speech-audio/) | **Speech & Audio** | Spectrograms, MFCCs, acoustic modeling, Whisper ASR, TTS synthesis |
| [`06-generative-ai/`](./06-generative-ai/) | **Generative AI & LLMs** | Autoregressive decoding, Prompting, RAG pipelines, PEFT/LoRA, autonomous agents |
| [`07-mlops/`](./07-mlops/) | **MLOps & Engineering** | Experiment tracking, model registry, inference servers (FastAPI/Triton), drift monitoring |
| [`08-system-design/`](./08-system-design/) | **ML System Design** | Feature stores, streaming architectures, distributed training, latency vs. throughput trade-offs |
| [`09-projects/`](./09-projects/) | **Projects & Capstones** | End-to-end production microservices and applied reference implementations |
| [`10-interview-prep/`](./10-interview-prep/) | **Interview Preparation** | Whiteboard coding drills, theory Q&A, and system design interview templates |
| [`11-research-papers/`](./11-research-papers/) | **Research Papers** | Paper reading frameworks, annotated seminal papers, and modern breakthrough summaries |
| [`12-resources/`](./12-resources/) | **Curated Resources** | Authoritative books, benchmark databases, and developer tooling compendiums |

---

## 3. Topic Architecture Standard

Every subtopic throughout modules `00` to `12` adheres strictly to a standardized structure:

```
topic-name/
├── README.md         # Concept breakdown, formulas, and execution matrix
├── notebook.ipynb    # Verified, runnable interactive Jupyter lab
├── code/             # Reusable Python scripts and utilities
├── interview.md      # Technical interview questions and trade-off analysis
└── references.md     # Authoritative research citations and documentation
```

A reusable blueprint is maintained at [`_templates/topic-template/`](./_templates/topic-template/).

---

## 4. Learning Pathways

Depending on your background and target career objectives, follow our curated pathways in [`ROADMAP.md`](./ROADMAP.md):

1. **Beginner → Machine Learning Engineer**: Focus on Python, statistics, supervised/unsupervised models, feature engineering, and MLOps.
2. **Beginner → Deep Learning Engineer**: Neural network mechanics, computer vision, transformers, and regularization dynamics.
3. **ML Engineer → LLM Engineer**: Transformer architectures, autoregressive sampling, RAG, PEFT/LoRA fine-tuning, and inference engines.
4. **Beginner → Generative AI Engineer**: Prompt engineering, vector search, multi-agent frameworks, and LLMOps.
5. **AI Research Path**: Mathematical foundations, seminal literature reading, and architecture re-implementation.
6. **ML System Design Path**: Distributed systems, high-throughput model serving, streaming pipelines, and industrial case studies.

---

## 5. How to Use This Repository

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/avars/ai-ml-handbook.git
   cd ai-ml-handbook
   ```

2. **Create and activate an isolated virtual environment**:
   ```bash
   # On macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install baseline dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify repository integrity**:
   ```bash
   python scripts/validate_repo.py
   ```

---

## 6. Notebook & Interactive Environment Philosophy

- **Self-Contained Execution**: Notebooks are designed to run cleanly top-to-bottom.
- **Lightweight by Default**: We avoid committing massive datasets or heavy checkpoints into git history. Small benchmark subsets and deterministic synthetic data are used for demonstration.
- **Dual Form**: Every interactive notebook (`notebook.ipynb`) is complemented by standalone Python code in `code/` for modular reuse in production pipelines.

---

## 7. Technology Stack

- **Languages & Runtimes**: Python 3.10+, JupyterLab
- **Data & Math**: NumPy, Pandas, SciPy, Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, TorchVision, TorchAudio
- **NLP & LLMs**: Hugging Face Transformers, Tokenizers, Datasets, Accelerate
- **Quality & CI**: Pytest, Ruff, Black, GitHub Actions

---

## 8. Current Status & Future Roadmap

- **Current Status (Phase 1)**: Core architecture established. Directory hierarchies, templates, continuous integration workflows, requirements, and standard structure validated.
- **Phase 2 (Next)**: Mathematical foundations and classical ML algorithm implementations.
- **Phase 3**: Deep Learning and Computer Vision implementations.
- **Phase 4**: Generative AI, RAG, and LLM fine-tuning labs.
- **Phase 5**: MLOps pipelines and full system design case studies.

For detailed milestone breakdowns, see [`ROADMAP.md`](./ROADMAP.md).

---

## 9. Contributing & Community

Contributions are welcomed! Before opening a pull request, please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) to understand our coding standards, topic templates, and verification steps.

---

## 10. License

This repository is distributed under the terms of the [MIT License](./LICENSE).
