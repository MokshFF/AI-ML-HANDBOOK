# Hands-on Engineering Projects (`09-projects`)

## Overview
Comprehensive, production-grade end-to-end machine learning, deep learning, Generative AI, and MLOps projects. Every project is genuinely runnable, thoroughly tested, and equipped with reproducible source code, notebooks, unit tests, and container configurations.

## Project Taxonomy & Navigation
| Category | Directory | Description | Projects |
| :--- | :--- | :--- | :--- |
| **Beginner** | [`beginner/`](./beginner/) | Tabular regression, classification, EDA, and feature engineering | House Price Prediction, Customer Churn |
| **Intermediate** | [`intermediate/`](./intermediate/) | Vision, NLP, recommendation engines, and time-series forecasting | Image Classifier, Sentiment Engine, RecSys, Demand Forecaster |
| **Advanced** | [`advanced/`](./advanced/) | Object detection with NMS and multimodal document understanding | Anchor-based Object Detector, Document Intelligence (IDP) |
| **Research** | [`research/`](./research/) | Foundational generative algorithms from scratch | Denoising Diffusion Probabilistic Model (DDPM) |
| **GenAI** | [`genai/`](./genai/) | Production RAG, agents, multimodal alignment, interview trainer | RAG Chatbot, Interview Trainer, Multi-Tool Agent, Vision-Language App, LLM Gateway |
| **MLOps** | [`mlops/`](./mlops/) | Production microservices, serving, and continuous drift monitoring | Production Serving API, Data Drift Monitoring Pipeline |

## Standard Project Schema
Every project repository adheres strictly to this structure:
```
<project-name>/
├── README.md            # Problem, motivation, architecture, pipeline, usage, evaluation
├── src/                 # Modular Python production source code
├── notebooks/           # Interactive, runnable Jupyter notebook (notebook.ipynb)
├── tests/               # Automated unit tests (pytest)
├── requirements.txt     # Minimal reproducible dependencies
├── .env.example         # Environment configuration template
└── Dockerfile           # Production container specification (where applicable)
```

## Universal Project Documentation Framework
Each project's `README.md` details:
1. **Problem**: Formal definition of the technical problem.
2. **Motivation**: Business and engineering rationale.
3. **Dataset**: Feature schema and input data distributions.
4. **Architecture**: Mermaid topology diagram illustrating components.
5. **Pipeline**: Step-by-step data and modeling lifecycle.
6. **Technologies**: Core frameworks and libraries.
7. **Installation**: Setup and dependency management instructions.
8. **Usage**: CLI execution and API interaction commands.
9. **Evaluation**: Mathematical definitions of evaluation metrics.
10. **Results**: Verified empirical performance on test distributions (marked pending where appropriate).
11. **Limitations**: Inherent technical boundaries and assumptions.
12. **Future Improvements**: Roadmap for production enhancement.
