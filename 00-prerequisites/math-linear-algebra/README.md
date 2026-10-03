# Math Linear Algebra

## Overview
Vectors, matrices, linear transformations, eigenvalues, eigenvectors, SVD, and matrix decompositions fundamental to ML algorithms.

## Learning Objectives
By completing this topic module, you will be able to:
- Explain core theoretical foundations, assumptions, and mathematical formulations.
- Implement key algorithms from scratch as well as using production-grade libraries.
- Diagnose and debug common issues such as numerical instability, over-fitting, and data leakage.
- Evaluate trade-offs between computational complexity, latency, memory consumption, and predictive performance.
- Formulate answers to relevant technical and conceptual interview questions.

## Directory Structure
- [`notebook.ipynb`](./notebook.ipynb): Interactive Jupyter notebook providing self-contained, reproducible walkthroughs.
- [`code/`](./code/): Reusable Python modules, scripts, and helper functions.
- [`interview.md`](./interview.md): Curated technical interview questions, conceptual drills, and trade-off analyses.
- [`references.md`](./references.md): Seminal papers, official documentation, authoritative textbooks, and external resources.

## Quick Start
1. Ensure your local virtual environment is activated and dependencies are installed:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Launch the interactive notebook:
   ```bash
   jupyter lab notebook.ipynb
   ```
3. Run standalone scripts in [`code/`](./code/):
   ```bash
   python -m code.<script_name>
   ```

## Key Concepts Matrix
| Concept | Description | Typical Use Case | Trade-offs |
| :--- | :--- | :--- | :--- |
| **Core Representation** | Primary mathematical or data abstraction | Problem formulation | Expressiveness vs. complexity |
| **Optimization Goal** | Objective or loss function minimized/maximized | Training & convergence | Convexity vs. local minima |
| **Inference Mechanism** | Forward evaluation / prediction pass | Production serving | Latency vs. precision |
